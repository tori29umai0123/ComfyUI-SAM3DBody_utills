from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from comfy_env import install as comfy_install
from comfy_env.detection import get_recommended_cuda_version
from comfy_env.environment.cache import get_local_env_path
from comfy_env.packages.cuda_wheels import CUDA_TORCH_MAP, get_wheel_url


NODE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = NODE_DIR / "nodes" / "comfy-env.toml"


def _is_known_uv_metadata_failure(exc: Exception) -> bool:
    text = str(exc)
    return (
        ".dist-info" in text
        and "torch-" in text
        and "expected version to start with a number" in text
    )


def _python_in_pixi_env(build_dir: Path) -> Path:
    return build_dir / ".pixi" / "envs" / "default" / ("python.exe" if sys.platform == "win32" else "bin/python")


def _rename_broken_torch_dist_info(site_packages: Path) -> None:
    for broken in site_packages.glob("torch-*.dist-info"):
        fixed = broken.with_name(broken.name.replace("torch-", "torch_", 1))
        if fixed.exists():
            continue
        broken.rename(fixed)
        print(f"[install workaround] Renamed {broken.name} -> {fixed.name}")


def _run(cmd: list[str]) -> None:
    print("[install workaround]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ensure_cuda_wheels(python_path: Path) -> None:
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

    for package in ("torch-scatter", "torch-sparse", "torch-cluster"):
        wheel_url = get_wheel_url(package, torch_version, cuda_version, py_version)
        if not wheel_url:
            raise RuntimeError(f"No wheel URL resolved for {package}")
        _run([str(python_path), "-m", "pip", "install", wheel_url])


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


def _run_workaround() -> None:
    env_path = get_local_env_path(NODE_DIR, CONFIG_PATH)
    build_dir = Path("C:/ce") / env_path.name if sys.platform == "win32" else Path.home() / ".ce" / env_path.name
    python_path = _python_in_pixi_env(build_dir)
    if not python_path.exists():
        raise RuntimeError(f"Expected staged pixi Python not found: {python_path}")

    site_packages = python_path.parent / "Lib" / "site-packages" if sys.platform == "win32" else next(
        (python_path.parent.parent / "lib").glob("python*/site-packages")
    )
    _rename_broken_torch_dist_info(site_packages)
    _ensure_cuda_wheels(python_path)
    _rename_broken_torch_dist_info(site_packages)
    _finalize_isolated_env(build_dir, env_path)


def main() -> None:
    try:
        comfy_install()
    except Exception as exc:
        if not _is_known_uv_metadata_failure(exc):
            raise
        print("[install workaround] Falling back for known comfy-env/uv metadata bug")
        _run_workaround()


if __name__ == "__main__":
    main()
