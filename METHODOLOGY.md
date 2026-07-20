# 先選對物件，再談 Grounding：JasCueVideoLab 方法論

這份文件說明一個看似簡單、實際上很容易做錯的影片 AI 流程：如何讓 Gemini 從影片理解內容、找到適合截圖的時間，再對原始影格框出「使用者真正想要的那一個物件」。

這是獨立研究，不是 JasCue 正式功能。所有時間與 bbox 都是待人工審核的 proposal，不是剪輯點、追蹤資料或正式 SpatialTrack。

## 給一般人的版本

如果一個畫面同時有三支手機、產品背板和招牌，只問 AI「幫我框重要物件」，AI 可能每次選到不同東西。它也許都框得很準，但框的不是你想要的那一個。

因此流程改成四步：

1. **AI 先提候選**：列出「左邊白色手機」「中央紫色手機」「右邊白色手機」等可辨識實例。
2. **人選目標**：使用者選擇真正關心的物件；如果一開始已指定，就跳過候選步驟。
3. **AI 找代表時間**：Gemini 只用 `MM:SS` 建議幾個物件清楚可見的時刻。
4. **本機抽原始幀再框選**：FFmpeg 找到真正的 frame PTS，Gemini 對那張原始影格輸出 bbox，最後產生 debug overlay 供人檢查。

簡化成一句話：

```text
沒指定目標 → Gemini 提候選 → 使用者選一個
已指定目標 ────────────────────┘
                 ↓
Gemini 建議 MM:SS → 本機驗證片長 → FFmpeg 抽原始幀／保存 PTS
                 ↓
      Gemini 單幀 bbox → debug overlay → 人工確認
```

這個順序把兩個不同問題拆開：

- **Selection**：到底要追哪個實例？這是使用者意圖。
- **Geometry**：該實例在這張圖的哪裡？這才是 Grounding。

「bbox 很準」不代表「target 選對」。把 selection 留給使用者，是目前最重要的可靠性改進。

## 實際觀察

在 22 秒的 SAMPLE-CONTINUITY 展示影片中，未指定 target 的泛化實驗三次雖都產生合法結果，但模型有時選招牌、有時選紫色手機。改用候選階段後，Gemini 一次提出 5 個可選實例：

- 左側白色手機
- 中央紫色手機
- 中央白色手機
- 銀色手提包
- 粉紅花瓶

Structured Output 通過本機 Pydantic schema；呼叫耗時 5.65 秒，依 Standard paid-tier 公開牌價估算為 US$0.012141。這次沿用先前上傳的 File API 物件，沒有重複上傳影片。

選定中央紫色手機後，Gemini 建議 `00:02`、`00:10`、`00:18`；在 2.002 秒的原始 3840×2160 frame 上，bbox proposal 為 `[412, 684, 467, 871]`（左上角原點、0–1000 normalized x-first），人工視覺抽查選中了正確的紫色實體手機，而不是大型紫色背板或旁邊的白色手機。

這只證明該測例可行，不等於全域準確率，也不是獨立 human ground truth。

## 技術流程

### 1. 媒體身份與時間基準

原始影片先經 ffprobe 取得 duration、coded/display dimensions、rotation、frame rate、time base、stream 與 container metadata，並計算 SHA-256。`asset_id` 與 `duration_ms` 由本機決定，模型必須原樣回傳。

Gemini 的 `MM:SS` 是 coarse semantic anchor。本機先檢查格式與片長，再把它換成 FFmpeg seek request；FFmpeg 真正抽到的 `frame_pts` 與 `frame_time_ms` 會另外保存。兩者不可混稱為 frame-accurate cut point。

### 2. File API 快取

Gemini File API 的檔案會保存 48 小時，期間可重複用同一個 file name／URI 呼叫模型。JasCueVideoLab 保存初始與最終 File API response：

- 有已保存紀錄時，先用 `files.get` 確認檔案仍為 `ACTIVE`。
- `ACTIVE` 就重用，不重新上傳。
- 只有 API 明確回報 `404`／`NOT_FOUND`，才視為已逾期或被刪除並重新上傳。
- 暫時性網路錯誤、權限錯誤或其他不確定失敗會直接保存並停止，不會以重傳掩蓋問題。
- 重傳前把舊 upload metadata 存到 `upload/history/`。
- `--force-reupload` 是明確覆寫快取的 escape hatch。

官方說明：[Files API](https://ai.google.dev/gemini-api/docs/files)、[File input methods](https://ai.google.dev/gemini-api/docs/file-input-methods)。

### 3. Target Candidate Map

沒有 target 時，不再讓 Content Map 或時間模型順便替使用者挑物件。獨立的 `TargetCandidateMap` 至少保存：

- `candidate_id`：穩定、可重複引用的 ID。
- `entity_kind`：phone、phone_screen、face、hand、product 等物件層級。
- `target_description`：可直接交給單幀 Grounding，並明確排除相似物件。
- `distinguishing_features`：顏色、位置、持有人、朝向或操作狀態。
- `representative_timestamp_mmss`：只用於候選預覽與人工判斷。
- `selection_reason`、`confidence` 與 `uncertainties`。

候選階段禁止 bbox、crop、mask 與 tracking data。它只回答「可以選什麼」，不回答「框在哪裡」。

### 4. 鎖定 target 後才找時間

選定候選後，`candidate_id` 與 `target_description` 會成為不可變輸入。每個 Direct Moment 都必須逐字回傳同一 target；模型若改選背板或其他手機，本機 contract 直接判定失敗。

若使用者一開始已提供 target ID 與精確描述，可以直接跳過候選階段。描述應包含：

- 物件層級，例如「手機螢幕」而不是「手機」。
- 實例特徵，例如中央、紫色、背面朝鏡頭。
- 排除條件，例如不可框大型 Reno 16 背板，也不可框左右白色手機。

### 5. 單幀 Grounding

FFmpeg 從 orientation-corrected 原始 source 抽幀後，以官方 Gemini image bbox convention 接收 `[y_min, x_min, y_max, x_max]`。API boundary schema 明確命名為 `box_2d_yxyx`，再由本機做純軸序轉換，輸出專案 canonical `[x_min, y_min, x_max, y_max]`。

如果 target 不可見，模型必須回傳 `visible=false`、`candidates=[]`，不得利用前後時刻猜位置。Debug overlay 必須畫在原始影格上供人工檢查。

### 6. Tracking 仍是另一層問題

正確的單幀 seed 可以初始化外部 tracker，但 tracker 回報 success 不代表幾何正確。本實驗曾出現 CSRT 221/221 accepted，bbox 卻逐漸縮到招牌上半部的反例。因此追蹤至少需要週期性 re-grounding、幾何／語意 gate 與真人 ground truth；單幀 Gemini bbox 不能直接當 production tracking data。

## 怎麼執行

先準備 artifact；同一 artifact 再執行時會優先重用 File API 物件：

```bash
uv run jascue-video-lab upload VIDEO.mp4 --output ARTIFACT
```

沒有 target 時先產生候選：

```bash
uv run jascue-video-lab suggest-targets ARTIFACT \
  --output-dir ARTIFACT/target-candidates
```

使用者選定 `candidate_id` 後，讓 Gemini 找代表時間並對抽出的原始影格 Grounding：

```bash
uv run jascue-video-lab direct-moment-repeat ARTIFACT \
  --candidate-map ARTIFACT/target-candidates/run-01/target_candidates.json \
  --candidate-id phone_purple_center \
  --runs 3 --ground-runs 1 \
  --output-dir ARTIFACT/purple-phone
```

也可直接提供 target：

```bash
uv run jascue-video-lab direct-moment-repeat ARTIFACT \
  --target-id center-purple-phone \
  --target-description '中央偏左、背面朝向鏡頭的紫色實體手機；排除大型背板與兩支白色手機。' \
  --runs 3 --ground-runs 1 \
  --output-dir ARTIFACT/purple-phone
```

若直接執行 `direct-moment-repeat` 卻沒有任何 target，CLI 會只產生候選並停止，不會偷偷挑一個物件進行 bbox。

## 可以分享的結論

> 影片 AI Grounding 最容易被忽略的問題，不是「框得準不準」，而是「它框的是不是使用者要的那一個」。我的實驗把流程拆成 Target Candidates → User Selection → MM:SS Moment → Exact Frame PTS → Single-frame Grounding。沒有指定目標時，AI 只提候選，不替人做最後選擇；指定後才找時間與 bbox。同一影片在 Gemini File API 的 48 小時保存期內會重用，只有確認過期才重傳。這讓錯誤更容易被看見，也讓每一步都能獨立驗證。

## 尚未證明的事

- Gemini timestamp 不是 frame-accurate cut point。
- Gemini bbox 不是 pixel mask 或 production-ready tracker。
- AI-assisted reviewer reference 不是獨立真人 ground truth。
- 一支影片的成功案例不能推論到所有拍攝條件、遮擋、快速 UI 或相似物件。
- 模型信心分數不能取代人工檢查。

可重現性依賴保存 raw API response、Structured Output schema validation、原始 frame hash、exact PTS、debug overlay、模型與 SDK provenance，以及同一輸入的多次比較；不能只保存整理後的漂亮結果。

## 本機 Blind Review App

Repository 內提供可直接拖放影片的本機 Web App。它不是展示報告，而是把上述方法論變成有狀態的審核流程：候選不預選、target 必須由使用者鎖定、Grounding 結果只顯示中性 Candidate 字母，且 reveal endpoint 在人工判定 JSON 寫入前一律拒絕存取模型 label、confidence 與理由。

人工判定至少保存 reviewer type、target、semantic request、exact frame PTS、frame hash、verdict、備註、選填的人工修正 bbox，以及 `model_details_revealed_before_annotation=false`。這能證明操作順序，但 reviewer 仍應如實填寫身分；系統不能單靠欄位宣稱某份標註具有獨立 ground-truth 品質。

```bash
set -a; source .env; set +a
uv run jascue-video-lab serve-review
```

開啟 `http://127.0.0.1:8765`。預設 local-only，影片、raw response、人工標註與匯出資料都保存在被 Git 排除的本機 artifacts。

### Direct-video bbox A/B

App 另提供隔離的 B 模式，直接把完整影片、target 與 `MM:SS` 交給 Gemini 要求 bbox。這不是官方 image object detection 範例所保證的 exact-frame 行為：官方影片文件說明 File API 影片預設以 1 FPS 保存／處理，但 API 不回傳被模型採用的 frame PTS 或 hash。因此 B 模式 schema 禁止宣稱 exact frame，固定保存 `reference_frame_status=unknown_gemini_video_sample`。

B 的 normalized bbox 可以投影到相同 `MM:SS` 所抽到的 FFmpeg 原始幀供人觀察，但投影圖必須標示 sample unknown。只有 A 與 B 都完成獨立盲審後，系統才計算 bbox IoU 與 center distance；這些數字量測的是兩個 proposal 的幾何差，不證明它們來自同一影格。

首次 SAMPLE-CONTINUITY live A/B 使用相同的中央紫色手機 target 與 `00:02`：原始 2.002 秒單幀方法為 `[412, 684, 467, 871]`，direct-video 方法為 `[413, 664, 466, 842]`，IoU 0.738123、normalized center distance 24.5。Direct-video 確實選中正確紫色手機，牌價估算 US$0.007224；但框較短且實際影片取樣幀未知，所以這項結果只能說「值得繼續 A/B」，不能取代原始單幀 Grounding。視覺檢查由 Codex 執行，尚未成為獨立 human annotation。
