from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from comfy_env import install as comfy_install
from comfy_env.detection import get_recommended_cuda_version
from comfy_env.environment.cache import get_local_env_path
from comfy_env.packages.cuda_wheels import CUDA_TORCH_MAP, get_wheel_url


NODE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = NODE_DIR / "nodes" / "comfy-env.toml"

# Prebuilt wheels for platforms not covered by the cuda-wheels index.
# Currently used for aarch64 Linux (DGX Spark / GH200) where cuda-wheels has
# no linux_aarch64 wheels for torch-scatter / torch-sparse / torch-cluster /
# detectron2. MANIFEST.txt lists each wheel with its SHA256.
_LOCAL_WHEELS_DIR = NODE_DIR / "wheels"
_LOCAL_WHEELS_MANIFEST = _LOCAL_WHEELS_DIR / "MANIFEST.txt"

# Packages whose project name contains a hyphen. pip's importlib metadata adapter
# (_compat.parse_name_and_version_from_info_directory) splits the dist-info dir name
# on the FIRST hyphen, so it parses e.g. "torch-sparse-0.6.18+cu128torch2.8" as
# name="torch" / version="sparse-0.6.18+cu128torch2.8" and crashes with InvalidVersion.
# Renaming the dir to use an underscore between project and version sidesteps this.
_HYPHENATED_TORCH_PACKAGES = ("torch-scatter", "torch-sparse", "torch-cluster")

# Extend comfy_env's built-in CUDA→torch mapping. Upstream only ships 12.4/12.8,
# so cu126/cu129/cu130 systems (DGX Spark, Blackwell datacentre cards, etc.) fall
# back to torch 2.8 and fail at "pytorch.org/whl/cu130 → torch==2.8.*" because
# cu130 only ships torch 2.9+. Mutating the dict in-place propagates to every
# importer (comfy_env.install holds the same dict reference).
_EXTRA_CUDA_TORCH = {
    "12.6": "2.8",
    "12.9": "2.9",
    "13.0": "2.9",
}
for _cuda_ver, _torch_ver in _EXTRA_CUDA_TORCH.items():
    CUDA_TORCH_MAP.setdefault(_cuda_ver, _torch_ver)

# torch ↔ torchvision compatibility table. Used when monkey-patching comfy_env's
# _install_via_pixi (whose own torchvision_map only covers 2.4 / 2.8). Without
# this, cu130 + torch 2.9 would resolve torchvision==0.23.* which doesn't exist
# on the cu130 PyTorch index.
_TORCH_TORCHVISION_MAP = {
    "2.4": "0.19",
    "2.5": "0.20",
    "2.6": "0.21",
    "2.7": "0.22",
    "2.8": "0.23",
    "2.9": "0.24",
    "2.10": "0.25",
    "2.11": "0.26",
}


def _is_known_uv_metadata_failure(exc: Exception) -> bool:
    text = str(exc)
    return (
        ".dist-info" in text
        and "torch-" in text
        and "expected version to start with a number" in text
    )


def _is_aarch64_linux() -> bool:
    return sys.platform.startswith("linux") and platform.machine().lower() in {"aarch64", "arm64"}


def _check_platform_supported() -> None:
    # cuda-wheels (https://pozzettiandrea.github.io/cuda-wheels/) only ships
    # win_amd64 + linux_x86_64 wheels. For aarch64 Linux (DGX Spark / GH200) we
    # ship prebuilt CUDA 13 wheels in wheels/ alongside this script and force
    # COMFY_ENV_CUDA_VERSION=13.0 so comfy_env pulls cu130 torch/torchvision
    # from the official PyTorch index (which has aarch64 builds).
    if _is_aarch64_linux():
        if not _LOCAL_WHEELS_DIR.is_dir() or not list(_LOCAL_WHEELS_DIR.glob("*.whl")):
            raise RuntimeError(
                f"aarch64 Linux requires prebuilt wheels in {_LOCAL_WHEELS_DIR}/. "
                "Build them on a DGX Spark / GH200 following the procedure in "
                "the project README and place the resulting torch_scatter, "
                "torch_sparse, torch_cluster, detectron2 .whl files plus a "
                "MANIFEST.txt into that directory, then re-run."
            )
        # comfy_env's GPU detection caps at 12.8 even for compute >= 10. Force
        # cu130 so the pixi step pulls the aarch64 cu130 torch/torchvision.
        os.environ.setdefault("COMFY_ENV_CUDA_VERSION", "13.0")
        return
    if sys.platform == "darwin":
        raise RuntimeError(
            "macOS is not supported: bpy and detectron2 do not ship wheels for "
            "this combination."
        )


def _parse_local_wheel_manifest() -> dict:
    # Returns { wheel_filename: sha256 }, or {} if no MANIFEST is present.
    if not _LOCAL_WHEELS_MANIFEST.is_file():
        return {}
    out = {}
    for line in _LOCAL_WHEELS_MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: "<file>.whl  sha256=<hex>  size=<bytes>"
        parts = line.split()
        if not parts or not parts[0].endswith(".whl"):
            continue
        sha = ""
        for p in parts[1:]:
            if p.startswith("sha256="):
                sha = p.split("=", 1)[1]
                break
        out[parts[0]] = sha
    return out


def _verify_local_wheel(wheel_path: Path, expected_sha: str) -> None:
    if not expected_sha:
        return
    actual = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    if actual != expected_sha:
        raise RuntimeError(
            f"SHA256 mismatch for {wheel_path.name}: "
            f"expected {expected_sha}, got {actual}. "
            f"Re-build or re-download the wheel."
        )


def _resolve_local_wheel(package: str) -> Path:
    # package is e.g. "torch-scatter" or "detectron2". Match either form
    # of the project name (hyphen or underscore) since wheel filenames use
    # underscores per PEP 427.
    candidates = list(_LOCAL_WHEELS_DIR.glob(f"{package.replace('-', '_')}-*.whl"))
    candidates += list(_LOCAL_WHEELS_DIR.glob(f"{package}-*.whl"))
    if not candidates:
        raise RuntimeError(
            f"No bundled wheel found for {package} in {_LOCAL_WHEELS_DIR}/. "
            f"Expected a file like {package.replace('-', '_')}-<version>+cu130torch2.9-cp311-cp311-linux_aarch64.whl"
        )
    wheel = candidates[0]
    expected_sha = _parse_local_wheel_manifest().get(wheel.name, "")
    _verify_local_wheel(wheel, expected_sha)
    return wheel


def _python_in_pixi_env(build_dir: Path) -> Path:
    return build_dir / ".pixi" / "envs" / "default" / ("python.exe" if sys.platform == "win32" else "bin/python")


def _rename_broken_torch_dist_info(site_packages: Path) -> None:
    for pkg in _HYPHENATED_TORCH_PACKAGES:
        underscore_pkg = pkg.replace("-", "_")
        for broken in site_packages.glob(f"{pkg}-*.dist-info"):
            fixed = broken.with_name(underscore_pkg + broken.name[len(pkg):])
            if fixed.exists():
                # The underscore-named dir already exists (e.g. created by a prior
                # rename before the wheel was re-extracted). They are duplicates;
                # drop the hyphen-named one so uv/pip don't choke on it later.
                shutil.rmtree(broken)
                print(f"[install workaround] Removed duplicate {broken.name} (kept {fixed.name})")
                continue
            broken.rename(fixed)
            print(f"[install workaround] Renamed {broken.name} -> {fixed.name}")


def _install_wheel_direct(wheel_url: str, site_packages: Path) -> None:
    # Download and extract the wheel directly. We avoid `pip install` because pip's
    # post-install summary reads dist-info dir names and crashes on hyphenated package
    # names (see _HYPHENATED_TORCH_PACKAGES); since pip's summary runs in the same
    # process as the install, there's no way to insert a rename between them.
    print(f"[install workaround] Fetching {wheel_url}")
    with tempfile.NamedTemporaryFile(suffix=".whl", delete=False) as tmp:
        wheel_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(wheel_url, wheel_path)
        with zipfile.ZipFile(wheel_path) as zf:
            zf.extractall(site_packages)
        print(f"[install workaround] Extracted {wheel_url.rsplit('/', 1)[-1]}")
    finally:
        wheel_path.unlink(missing_ok=True)


def _ensure_cuda_wheels(python_path: Path, site_packages: Path) -> None:
    if _is_aarch64_linux():
        # aarch64 path: extract the bundled wheels from wheels/ directly.
        for package in _HYPHENATED_TORCH_PACKAGES:
            wheel_path = _resolve_local_wheel(package)
            print(f"[install workaround] {package} from {wheel_path}")
            with zipfile.ZipFile(wheel_path) as zf:
                zf.extractall(site_packages)
            _rename_broken_torch_dist_info(site_packages)
        return

    cuda_version = get_recommended_cuda_version()
    if not cuda_version:
        raise RuntimeError("CUDA wheel workaround requested, but no CUDA version was detected")

    torch_version = CUDA_TORCH_MAP.get(".".join(cuda_version.split(".")[:2]), "2.8")
    py_version = subprocess.check_output(
        [
            str(python_path),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
    ).strip()

    for package in _HYPHENATED_TORCH_PACKAGES:
        wheel_url = get_wheel_url(package, torch_version, cuda_version, py_version)
        if not wheel_url:
            raise RuntimeError(f"No wheel URL resolved for {package}")
        _install_wheel_direct(wheel_url, site_packages)
        _rename_broken_torch_dist_info(site_packages)


def _finalize_isolated_env(build_dir: Path, env_path: Path) -> None:
    pixi_default = build_dir / ".pixi" / "envs" / "default"
    final_env = build_dir / "env"
    if not pixi_default.exists():
        raise RuntimeError(f"Missing pixi env: {pixi_default}")

    if final_env.exists():
        shutil.rmtree(final_env, ignore_errors=True)
    shutil.move(str(pixi_default), str(final_env))

    if env_path.exists():
        try:
            env_path.unlink()
        except OSError:
            env_path.rmdir()

    env_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(env_path), str(final_env)], check=True)
    else:
        env_path.symlink_to(final_env)

    (build_dir / ".done").touch()
    print(f"[install workaround] Finalized {env_path} -> {final_env}")


def _resolve_build_dir(env_path: Path) -> Path:
    # Match comfy_env.install._install_via_pixi: Windows uses sha256(env_path)[:8]
    # under C:/ce, other platforms use "<env_path>_build" alongside env_path.
    if sys.platform == "win32":
        short_hash = hashlib.sha256(str(env_path).encode()).hexdigest()[:8]
        return Path("C:/ce") / short_hash
    return env_path.parent / f"{env_path.name}_build"


def _run_workaround() -> None:
    env_path = get_local_env_path(NODE_DIR, CONFIG_PATH)
    build_dir = _resolve_build_dir(env_path)
    python_path = _python_in_pixi_env(build_dir)
    if not python_path.exists():
        raise RuntimeError(f"Expected staged pixi Python not found: {python_path}")

    site_packages = python_path.parent / "Lib" / "site-packages" if sys.platform == "win32" else next(
        (python_path.parent.parent / "lib").glob("python*/site-packages")
    )
    _rename_broken_torch_dist_info(site_packages)
    _ensure_cuda_wheels(python_path, site_packages)
    _finalize_isolated_env(build_dir, env_path)


def _patched_install_via_pixi(cfg, node_dir, log, dry_run, is_root=True):
    # Drop-in replacement for comfy_env.install._install_via_pixi. The only
    # behavioural change is `torchvision_map` (and its fallback) now consults
    # _TORCH_TORCHVISION_MAP so cu126/cu129/cu130 + torch 2.6–2.11 resolve to a
    # torchvision version that actually exists on the cu* PyTorch index.
    # Everything else is byte-for-byte the upstream code path.
    from comfy_env.config import CONFIG_FILE_NAME
    from comfy_env.environment.cache import get_root_env_path, get_local_env_path as _get_local_env_path
    from comfy_env.install import _find_main_node_dir, _find_uv, _has_isolated_subdirs
    from comfy_env.packages.pixi import ensure_pixi
    from comfy_env.packages.toml_generator import write_pixi_toml
    from comfy_env.packages.cuda_wheels import get_wheel_url as _get_wheel_url

    deps = cfg.pixi_passthrough.get("dependencies", {})
    pypi_deps = cfg.pixi_passthrough.get("pypi-dependencies", {})
    if not cfg.cuda_packages and not deps and not pypi_deps:
        if not is_root or not _has_isolated_subdirs(node_dir):
            log("No packages to install")
        return

    log("\nInstalling via pixi:")
    if cfg.cuda_packages: log(f"  CUDA: {', '.join(cfg.cuda_packages)}")
    if deps: log(f"  Conda: {len(deps)}")
    if pypi_deps: log(f"  PyPI: {len(pypi_deps)}")
    if dry_run: return

    if is_root:
        env_path = get_root_env_path(node_dir)
    else:
        config_path = node_dir / CONFIG_FILE_NAME
        main_node_dir = _find_main_node_dir(node_dir)
        env_path = _get_local_env_path(main_node_dir, config_path)

    if sys.platform == "win32":
        short_hash = hashlib.sha256(str(env_path).encode()).hexdigest()[:8]
        short_base = Path("C:/ce")
        short_base.mkdir(parents=True, exist_ok=True)
        build_dir = short_base / short_hash
    else:
        build_dir = env_path.parent / f"{env_path.name}_build"
    log(f"[comfy-env] build_dir={build_dir}")
    log(f"[comfy-env] env_path={env_path}")

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    pixi_path = ensure_pixi(log=log)

    cuda_version = torch_version = None
    if cfg.has_cuda and sys.platform != "darwin":
        cuda_version = get_recommended_cuda_version()
        if cuda_version:
            torch_version = CUDA_TORCH_MAP.get(".".join(cuda_version.split(".")[:2]), "2.8")

    write_pixi_toml(cfg, build_dir, log)
    log("Running pixi install...")
    pixi_env = dict(os.environ)
    pixi_env["UV_PYTHON_INSTALL_DIR"] = str(build_dir / "_no_python")
    pixi_env["UV_PYTHON_PREFERENCE"] = "only-system"
    result = subprocess.run([str(pixi_path), "install"], cwd=build_dir, capture_output=True, text=True, env=pixi_env)
    if result.returncode != 0:
        raise RuntimeError(f"pixi install failed:\nstderr: {result.stderr}\nstdout: {result.stdout}")

    if cfg.cuda_packages and sys.platform != "darwin":
        pixi_default = build_dir / ".pixi" / "envs" / "default"
        python_path = pixi_default / ("python.exe" if sys.platform == "win32" else "bin/python")
        if not python_path.exists():
            raise RuntimeError(f"No Python in pixi env: {python_path}")

        result = subprocess.run([str(python_path), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                                capture_output=True, text=True)
        py_version = result.stdout.strip() if result.returncode == 0 else f"{sys.version_info.major}.{sys.version_info.minor}"

        uv_path = _find_uv()

        pytorch_packages = {"torch", "torchvision", "torchaudio"}
        # PATCHED: extended torchvision compatibility table (was: {"2.8": "0.23", "2.4": "0.19"})
        torchvision_map = dict(_TORCH_TORCHVISION_MAP)

        if cuda_version:
            pytorch_index = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')[:3]}"
            pin_torch_version = torch_version
            log(f"Installing CUDA packages from {pytorch_index}")
        else:
            pytorch_index = "https://download.pytorch.org/whl/cpu"
            pin_torch_version = "2.8"
            log(f"Installing CPU packages from {pytorch_index}")

        for package in cfg.cuda_packages:
            if package in pytorch_packages:
                if package == "torch":
                    pin_version = pin_torch_version
                elif package == "torchvision":
                    # PATCHED: fallback to upstream's "0.23" only if the torch version is unknown
                    pin_version = torchvision_map.get(pin_torch_version, "0.23")
                else:
                    pin_version = pin_torch_version
                pkg_spec = f"{package}=={pin_version}.*"
                pip_cmd = [uv_path, "pip", "install", "--python", str(python_path),
                           "--extra-index-url", pytorch_index, "--index-strategy", "unsafe-best-match", pkg_spec]
                log(f"  {' '.join(pip_cmd)}")
                result = subprocess.run(pip_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to install {package}:\nstderr: {result.stderr}\nstdout: {result.stdout}")
            elif cuda_version:
                if _is_aarch64_linux():
                    # cuda-wheels has no aarch64 builds; use wheels/ shipped with this
                    # repo (built on a real DGX Spark, see wheels/MANIFEST.txt). Bypass
                    # uv entirely because it crashes parsing hyphenated dist-info dir
                    # names like torch-sparse-0.6.18+cu130torch2.9.dist-info.
                    wheel_path = _resolve_local_wheel(package)
                    sp = next((python_path.parent.parent / "lib").glob("python*/site-packages"))
                    log(f"  {package} from {wheel_path} (bundled aarch64 cu130 wheel)")
                    with zipfile.ZipFile(wheel_path) as zf:
                        zf.extractall(sp)
                    _rename_broken_torch_dist_info(sp)
                else:
                    wheel_url = _get_wheel_url(package, torch_version, cuda_version, py_version)
                    if not wheel_url:
                        raise RuntimeError(f"No wheel for {package}")
                    log(f"  {package} from {wheel_url}")
                    cmd = [uv_path, "pip", "install", "--python", str(python_path), "--no-deps", wheel_url]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(f"Failed to install {package}:\nstderr: {result.stderr}\nstdout: {result.stdout}")
            else:
                log(f"  {package} (skipped - GPU only)")

    pixi_default = build_dir / ".pixi" / "envs" / "default"
    if pixi_default.exists():
        if sys.platform == "win32":
            # Python 3.11 has no Path.is_junction(); use unlink → rmdir → rmtree
            # cascade. unlink works on junctions (treated as link), rmdir on empty
            # dirs (incl. junction targets), rmtree as last resort for real dirs.
            env_path.parent.mkdir(parents=True, exist_ok=True)
            if env_path.exists() or env_path.is_symlink():
                try:
                    env_path.unlink()
                except OSError:
                    try:
                        env_path.rmdir()
                    except OSError:
                        shutil.rmtree(env_path)
            final_short = build_dir / "env"
            if final_short.exists():
                shutil.rmtree(final_short)
            shutil.move(str(pixi_default), str(final_short))
            subprocess.run(["cmd", "/c", "mklink", "/J", str(env_path), str(final_short)],
                           capture_output=True)
            log(f"Env: {env_path} -> {final_short}")
        else:
            if env_path.exists():
                shutil.rmtree(env_path)
            shutil.move(str(pixi_default), str(env_path))
            shutil.rmtree(build_dir, ignore_errors=True)
            log(f"Env: {env_path}")
        shutil.rmtree(node_dir / ".pixi", ignore_errors=True)


def _apply_comfy_env_patches() -> None:
    import comfy_env.install as ce_install
    ce_install._install_via_pixi = _patched_install_via_pixi


def main() -> None:
    _check_platform_supported()
    cuda_override = os.environ.get("COMFY_ENV_CUDA_VERSION", "").strip()
    if cuda_override:
        print(f"[install] COMFY_ENV_CUDA_VERSION override active: {cuda_override}")

    _apply_comfy_env_patches()

    try:
        comfy_install()
    except Exception as exc:
        if not _is_known_uv_metadata_failure(exc):
            raise
        print("[install workaround] Falling back for known comfy-env/uv metadata bug")
        _run_workaround()


if __name__ == "__main__":
    main()
