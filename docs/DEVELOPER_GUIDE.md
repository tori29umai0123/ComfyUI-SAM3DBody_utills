# 開発者ガイド: ブレンドシェイプ追加 + tools/ リファレンス

**Language:** 🇯🇵 日本語 (current) ・ [🇬🇧 English](DEVELOPER_GUIDE.en.md)

> このドキュメントは、**Blender を使って `tools/bone_backup/all_parts_bs.fbx` を編集し、新しいブレンドシェイプ (シェイプキー) を追加・更新したい上級ユーザー向け**です。同梱の 18 ブレンドシェイプをそのまま使うだけなら触る必要はありません — メインの [README.md](../README.md) に戻ってください。

### ⚠ ブレンドシェイプ編集には Blender が必須（※上級者向け）

> **この項は「自分でモデルをいじって新しいブレンドシェイプ・パラメータを増やしたい／既存シェイプを調整したい Blender ユーザー向け」の項目です。** 同梱済みの 18 ブレンドシェイプをそのまま使って ComfyUI でレンダリングするだけなら Blender は不要で、この節は読み飛ばしてかまいません。
>
> **Blender が必要になる理由:** FBX のシェイプキーを GUI で編集し、`tools/extract_face_blendshapes.py` を Blender headless で呼び出して `presets/face_blendshapes.npz` を再生成するため。

本ノードのブレンドシェイプは `tools/bone_backup/all_parts_bs.fbx` の**シェイプキー**として格納されています。自分で新しいシェイプキーを追加したり既存のものを編集したりする場合は、**追加・編集・値変更のすべてに [Blender](https://www.blender.org/) が必要**になります。

- **動作確認環境**: Blender 4.1（`C:/Program Files/Blender Foundation/Blender 4.1/blender.exe`）
- 他のバージョンでも動作する可能性はあるが、`tools/extract_face_blendshapes.py` 内のパスは 4.1 を前提に記載
- 別バージョンを使う場合は、コマンド例の `blender.exe` パスを置き換えてください

ブレンドシェイプ編集のワークフロー：

1. Blender で `tools/bone_backup/all_parts_bs.fbx` を開く
2. `mhr_reference` オブジェクトのシェイプキーを追加・編集
3. FBX として上書き保存（**エクスポート設定は下記に従う**）
4. 下記の [FBX 作成・更新後の実行コマンド](#fbx-作成更新後の実行コマンド) の 2 本を実行

繰り返しますが、既存の `presets/face_blendshapes.npz` をそのまま使ってレンダリングするだけなら Blender は触らなくて構いません。

#### FBX エクスポート設定（再出力時はこの設定で）

![Blender FBX export settings](blender_fbx_export_settings.png)

| 項目 | 値 |
|---|---|
| パスモード | 自動 |
| バッチモード | オフ |
| **対象（Limit to）** | **全て OFF**（選択したオブジェクト / 可視オブジェクト / アクティブコレクション はチェックしない） |
| **オブジェクトタイプ** | **アーマチュア** + **メッシュ** のみ選択（エンプティ / カメラ / ランプ / その他は選択しない） |
| カスタムプロパティ | OFF |
| **スケール** | **1.00** |
| **スケールを適用** | **すべて FBX** |
| **前方** | **-Z が前** |
| **上** | **Y が上** |
| 単位を適用 | ON |
| 空間の変換を使用 | ON |
| トランスフォームを適用 | ON |
| アニメーションをベイク | ON |

**重要**：**前方 = -Z / 上 = Y / スケール 1.0** は内部の座標変換行列（`_FBX_TO_MHR_ROT`）と整合するために必須です。他の軸設定で保存すると `extract_face_blendshapes.py` での位置マッチングが狂い、ブレンドシェイプが誤った方向に効きます。

## 開発者ガイド: 新しいブレンドシェイプの追加

UI のブレンドシェイプは `presets/face_blendshapes.npz` の `meta_shapes` から動的に生成されるため、**Python コードの変更なしで追加できる**。ワークフローは以下。

### 手順

1. **Blender で `tools/bone_backup/all_parts_bs.fbx` を開く**
2. 変形させたいオブジェクト（`head` / `neck` / `chest` / `torso` / `hips` / 左右 `upper_arm` / `forearm` / `hand` など）を選択
3. オブジェクトプロパティ → **Shape Keys** から新しいキーを追加
   - まず `Basis`（ある場合はスキップ）
   - `＋` ボタンで新キー作成 → 名前を付ける（例: `belly_fat`）
   - キーを編集モードに入れて変形。保存時に value=1.0 時の形状が記録される
4. **複数オブジェクト横断させる場合**：同じ名前のシェイプキーを関係する全オブジェクトに追加し、それぞれで自分の領域を変形する（例: `neck_thicken` は `head` + `neck` + `chest` にそれぞれ存在）
5. FBX をエクスポート上書き保存（`File` → `Export` → `FBX` 、もしくは同名で保存）
6. **npz 再生成**（Blender headless で FBX を読み込み直し、全シェイプキーを抽出）：

   ```
   "C:/Program Files/Blender Foundation/Blender 4.1/blender.exe" ^
       --background --python tools/extract_face_blendshapes.py
   ```
7. **region JSON 再同期**（FBX 頂点を MHR インデックス空間へ NN マッピング）：

   ```
   .venv/Scripts/python.exe custom_nodes/ComfyUI-SAM3DBody_utills/tools/rebuild_vertex_jsons.py
   ```
8. **ComfyUI を再起動**（もしくは該当ノードを再読み込み）。新しいシェイプ名がスライダーとして自動的に UI に現れる

### 命名規則

FBX のシェイプキー名は `settings_json` のキー名と 1:1 で対応する（UI 上だけ `bs_` プレフィックスが付く）。UI 表示順 (`_UI_BLENDSHAPE_ORDER` @ `process.py`) は頭 → 首 → 胸 → 肩 → 腰 → 手足。新しいシェイプ名もこの部位順に揃えて追加すると並びが崩れません。`_UI_BLENDSHAPE_ORDER` に入っていないシェイプはアルファベット順で末尾に追加されます。

delta の posed フレームへの回転は **MHR LBS スキニング重み** から頂点単位に計算されるので、シェイプ名の規則による anchor 関節の指定は不要です。

### 既存シェイプの値を編集した場合

Blender で既存のシェイプキーを編集 → FBX 上書き → 手順 6〜8 を実行するだけ。値は **完全に上書き**される（npz / JSON はソースから毎回再生成されるため差分管理不要）。

### 新シェイプを UI に「出したくない」場合

ソース FBX の shape key 名を一時的に `_hidden_foo` のように変えれば抽出には含まれるが UI にも並ぶ（現状は全シェイプを無差別に公開）。UI からのみ隠したい場合は `_discover_blendshape_names()` にフィルタを追加する。

### 既存の npz / JSON を全削除して再構築したい場合

```
del presets\face_blendshapes.npz
del presets\*_vertices.json
# 手順 6 と 7 を実行
```

実行後、`presets/` に npz + 17 オブジェクト分の `<obj>_vertices.json` が揃う。

## FBX 作成・更新後の実行コマンド

Blender で `tools/bone_backup/all_parts_bs.fbx` を編集・保存したあと、**ComfyUI ルート (`C:/ComfyUI`)** で以下の 2 本を順に実行するだけで全同期されます。

### 1. FBX からブレンドシェイプ + 全オブジェクト位置を抽出

```
"C:/Program Files/Blender Foundation/Blender 4.1/blender.exe" ^
    --background --python custom_nodes/ComfyUI-SAM3DBody_utills/tools/extract_face_blendshapes.py
```

→ `presets/face_blendshapes.npz` を生成・上書き

### 2. 各オブジェクトの頂点 JSON を再生成

```
.venv/Scripts/python.exe custom_nodes/ComfyUI-SAM3DBody_utills/tools/rebuild_vertex_jsons.py
```

→ `presets/<obj>_vertices.json` を全オブジェクト分生成（FBX に存在しないオブジェクト名の JSON は自動削除）

### 3. ComfyUI を再読み込み

ブラウザで ComfyUI を更新すれば、新しいシェイプキーのスライダーが UI に自動出現し、新しいパーツ構成での変形が有効化されます。

## tools/ スクリプト一覧

Blender と Python venv を使い分けるスクリプトが揃っています。通常運用で触るのは **extract_face_blendshapes.py** と **rebuild_vertex_jsons.py** の 2 本だけです。

### 日常運用（FBX を編集したら毎回）

| スクリプト | 実行環境 | 役割 |
|---|---|---|
| `tools/extract_face_blendshapes.py` | **Blender** (`--background --python`) | `tools/bone_backup/all_parts_bs.fbx` を読み、全メッシュ・全シェイプキーを走査して `presets/face_blendshapes.npz` を生成。各オブジェクトの base 位置と shape delta を `base__<obj>` / `delta__<obj>__<shape>` キーで格納。全オブジェクトの world 座標も `all_base__<obj>` に保存（rebuild 用） |
| `tools/rebuild_vertex_jsons.py` | **ComfyUI venv** (`.venv/Scripts/python.exe`) | npz の `all_base__<obj>` と MHR rest-pose 頂点を NN マッチングし、`presets/<obj>_vertices.json` を全オブジェクト分再生成。FBX のパーツ分けが変わった場合は自動追従、stale JSON は自動削除 |

Blender で FBX を編集 → 保存 → この 2 本を走らせる → ComfyUI 再読み込み、の流れが基本フロー。

### セットアップ / 再構築時

| スクリプト | 実行環境 | 役割 |
|---|---|---|
| `tools/export_reference_obj.py` | ComfyUI venv | MHR rest-pose メッシュを**単一の非パーツ OBJ**として `tools/bone_backup/mhr_reference.obj` に出力。FBX が破損した場合のバックアップ / Blender でゼロからパーツ再分割したい時の出発点として使用 |

## Preset packs (ブレンドシェイプ定義の切替 / 配布)

このプラグインは、ブレンドシェイプ定義・頂点マップ・キャラクタープリセット JSON を **preset pack** という単位でまとめて管理します。ユーザーがオリジナルのブレンドシェイプ集を作って公開・配布できるようにするための仕組みです。

### pack の切替

他のユーザーが配布した pack (例: `my_custom_pack`) を使うには:

1. `presets/my_custom_pack/` として pack フォルダを配置 (上記構造と同じレイアウト)
2. リポジトリ直下の `config.ini` を開き、`[active] pack = default` を `pack = my_custom_pack` に書き換え
3. ComfyUI を再読み込み (F5)

指定した pack が見つからない場合は自動的に `default` へフォールバックします。

### 独自 pack の作り方

1. `presets/default/` を `presets/my_pack/` にコピー
2. `config.ini` の `[active] pack` を `my_pack` に変更
3. `tools/bone_backup/all_parts_bs.fbx` を Blender で開き、独自のブレンドシェイプを追加・編集
4. `tools/extract_face_blendshapes.py` を実行 — active pack (`my_pack`) の下に `face_blendshapes.npz` が更新され、`chara_settings_presets/*.json` と `process.py` の UI 順序も自動同期
5. 必要なら `tools/rebuild_vertex_jsons.py` で `mhr_reference_vertices.json` も再生成
6. `presets/my_pack/` フォルダ全体を zip して配布

pack 内のファイルはすべて自己完結しているので、受け取ったユーザーは `presets/` に drop して `config.ini` の `[active] pack` を書き換えるだけで使えます。

