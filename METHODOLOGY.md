# 先鎖定證據，再談 Geometry：JasCueVideoLab 方法論

這份文件說明一個看似簡單、實際上很容易做錯的影片 AI 流程：如何讓 Gemini 從影片理解內容、找到適合截圖的時間，再對原始影格框出「使用者真正想要的那一個物件」。

這是獨立研究，不是 JasCue 正式功能。所有時間與 bbox 都是待人工審核的 proposal，不是剪輯點、追蹤資料或正式 SpatialTrack。

## 給一般人的版本

如果畫面同時有多個相似實例、背景描繪、反射或物件局部，只問 AI「幫我框重要物件」，AI 可能每次選到不同東西。它也許都框得很準，但框的不是你想要的那一個。這個問題和物件是人、產品、動物、工具、螢幕或文件無關。

因此流程改成四步：

1. **AI 先提候選**：列出畫面中可區分的實例，不產生座標。
2. **人選目標並鎖定條件**：使用者選擇真正關心的實例、必要特徵、排除特徵與可選的事件條件；如果一開始已指定，就跳過候選步驟。
3. **AI 找代表時間**：Gemini 只用 `MM:SS` 建議幾個物件清楚可見的時刻。
4. **本機抽原始幀再框選**：FFmpeg 找到真正的 frame PTS，Gemini 對那張原始影格只輸出 bbox，最後產生 debug overlay 供人檢查。需要連續幾何時，才把人工核准的 bbox 交給 SAM。

簡化成一句話：

```text
沒指定目標 → Gemini 提候選 → 使用者選一個
已指定目標 ────────────────────┘
                 ↓
Gemini 建議 MM:SS → 本機驗證片長 → FFmpeg 抽原始幀／保存 PTS
                 ↓
      Gemini 單幀 bbox → debug overlay → 人工確認
                                      ↓
                    選配：SAM 在同一 shot 內傳播 mask
```

這個順序把兩個不同問題拆開：

- **Query Lock**：到底要找哪個實例、驗證哪個可觀察條件？這是使用者意圖。
- **Geometry**：該實例在這張圖的哪裡？這才是 Grounding。

「bbox 很準」不代表「target 選對」。把 selection 留給使用者，是目前最重要的可靠性改進。

## 實際觀察

在 22 秒的匿名化產品展示影片中，未指定 target 的泛化實驗三次雖都產生合法結果，但模型有時選招牌、有時選紫色手機。改用候選階段後，Gemini 一次提出 5 個可選實例：

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

### 3. Target Candidate Map 與 EvidenceQueryLock

沒有 target 時，不再讓 Content Map 或時間模型順便替使用者挑物件。獨立的 `TargetCandidateMap` 至少保存：

- `candidate_id`：穩定、可重複引用的 ID。
- `entity_kind`：person、face、hand、product、device、screen、document、UI element 或 other 等 schema 層級。
- `target_description`：可直接交給單幀 Grounding，並明確排除相似物件。
- `distinguishing_features`：顏色、位置、持有人、朝向或操作狀態。
- `representative_timestamp_mmss`：只用於候選預覽與人工判斷。
- `selection_reason`、`confidence` 與 `uncertainties`。

候選階段禁止 bbox、crop、mask 與 tracking data。它只回答「可以選什麼」，不回答「框在哪裡」。

候選由使用者選定後會形成 domain-neutral `EvidenceQueryLock`，至少保存穩定 target ID、正向／排除特徵、可選的 observable predicate、必要證據、負面限制、版本與 hash。後續 Grounding 和 tracking artifact 都保存同一份 lock hash；變更條件會建立新 cache variant，不會沿用舊框。

既有 dense frame selection 若是在 QueryLock 建立前產生，即使剛好使用相同 target ID 也不會重用，因為它沒有 lock hash，無法證明採用了新的排除條件或 predicate。現階段會退回事件的 coarse keyframe／本機中點並讓 exact-frame gate 決定是否有足夠證據；未來的 lock-aware dense refinement 必須把完整 lock hash 寫入 artifact 才能重用。

### 4. 鎖定 target 後才找時間

選定候選後，`candidate_id` 與 `target_description` 會成為不可變輸入。每個 Direct Moment 都必須逐字回傳同一 target；模型若改選背板或其他手機，本機 contract 直接判定失敗。

若使用者一開始已提供 target ID 與精確描述，可以直接跳過候選階段。描述應包含：

- 目標層級，例如局部、整體或持有人，不可互相擴張。
- 可直接看見的實例特徵，例如位置、顏色、朝向、持有關係或狀態。
- 排除條件，例如另一個同類實例、背景圖像、反射或支撐物。

### 5. 單幀 Grounding

FFmpeg 從 orientation-corrected 原始 source 抽幀後，以官方 Gemini image bbox convention 接收 `[y_min, x_min, y_max, x_max]`。API boundary schema 明確命名為 `box_2d_yxyx`，再由本機做純軸序轉換，輸出專案 canonical `[x_min, y_min, x_max, y_max]`。

如果 target 不可見，模型必須回傳 `visible=false`、`candidates=[]`，不得利用前後時刻猜位置。`match_status` 另外區分 `matched`、`ambiguous`、`not_visible`、`target_mismatch` 與 `insufficient_evidence`；可選的 `predicate_status` 只回答指定動作／狀態是否有證據，不能取代物件身分或可見性。多候選不得依最高 confidence 自動選框，必須由人指定 candidate。Debug overlay 必須畫在原始影格上供人工檢查。

### 6. Gemini bbox → SAM；不使用 Gemini polygon 當主路徑

正確的單幀 bbox seed 可以初始化 SAM 2.1，由 SAM 將矩形精煉為 mask 並向前／向後傳播。主路徑明確拒絕 Gemini polygon seed：在目前最具辨識難度的 A/B 反例中，bbox seed 保住了指定實例而 polygon seed 跟錯區域；polygon artifacts 只保留為唯讀歷史實驗，不會進入 Full v1 或剪輯 renderer。這是風險導向的架構決策，不是宣稱 bbox 在所有物件上都有較高 pixel accuracy。

SAM predictor 在初始化前先以 FFmpeg 找出 seed 所屬 shot，實際分析區間為 `使用者／事件允許範圍 ∩ seed shot`。每個 sample 保存原始 decoded `source_pts`、time base 衍生的時間與 mask-derived bbox；固定 FPS debug MP4 只是預覽，不是剪輯時間軸。tracker 有 mask 仍不等於語意身分已確認，因此 drift、lost 或遮擋後重現仍需重新 Grounding 或人工確認。

### 7. 入選後才做 Trim Intent，且不能自動核准

Clip Card 的 `MM:SS` event 只負責召回可能可用的區間。真正入選後，預設讓 Gemini 直接觀看完整 proxy，只在 `coarse event ∩ FFmpeg shot` 內提出 coarse `MM:SS` 代表性 select。模型必須保留可理解的 setup／action／result，但不應因同事件其他階段也可用就全部保留；可用但重複的階段可以不入選，且不得被誤稱為失敗或 reset。

局部 Trim Intent 與全片 duration budget 是兩個不同問題。各段獨立保留完整動作後，總長可能超過 open-edit planner 原先分配的秒數；因此正式流程必須在所有入選段落取得實際 PTS 長度後，再做一次全片 keep／drop／reorder 協調。第一版不得為了湊總長而在未知語意位置硬切片段中間。

9:16 的 crop 決策必須可回放與診斷。除 target、Grounding、SAM track 與 fallback 外，renderer 需保存每個 crop keyframe 的片段相對時間、required-region union、合法 crop interval、containment、可見比例及實際 `crop_x_pixels`。裁切路徑先平滑，再逐 sample 投影回可行區間，不能讓平滑延遲把主體推出畫面。多個必留人物／物件／文字／UI 應分成獨立 required regions，各自取得 bbox 並共用一個 SAM session，不能期待一個複合自然語言 target 產生可靠聯合 mask。

同一套 geometry 不應假設來源比例固定。Renderer 以 orientation-corrected display dimensions 建立 aspect-preserving cover transform，並在 x／y 兩軸分別求解合法 crop interval；因此 4:3、直式、超寬來源不會被拉伸。FFmpeg 的 sample aspect ratio 也屬於來源幾何：非方形像素須先正規化成 square-pixel display space；在 tracker 尚未與該座標系綁定前，動態 reframe 必須 fail closed 到 SAR-corrected 靜態版本並留下 review risk。track seed dimensions、analysis aspect 或多 track lineage 不一致時也不得重新解釋 normalized bbox。

`primary_center` 表示未列為 required 的次要 context 可以犧牲，不表示 required target 可以被裁掉。required union 比 9:16 視窗寬、tracking coverage 不完整或任一 sample 無法 containment 時，預設換候選或 fail closed。只有人工明示 `controlled_clip` 時才可依 `preserve_start`、`preserve_end` 或 `balanced` 做受控溢位，並保存最小可見比例與 review requirement。文字規則同樣泛化：把必須讀完的語意核心列為 required `text_region`，而不是為特定品牌或語言寫判斷分支。

在明示禁用背景補邊的 preview 中，若 propagation coverage 失敗但 exact-frame seed 仍有效，可將 required seed union 作為整段靜態 anchor，而不是盲目退回來源中央。此降級固定保存 `seed_anchor_static_hold` 與 `motion_outside_seed_unverified`，不能聲稱追蹤完成；若主體會移動或 required union 本來就過寬，仍應改選 take、調整主次、拆鏡或交由人工決定 layout。

Gemini 不需要每次重看整支成片。成本合理的順序是先跑零 API 成本的本機 geometry、coverage、shot 與 media gate；只有 text/UI、多 required regions、controlled clip、drift、fallback 或語意身分疑慮才觸發複核。black/freeze gate 仍是後續規劃，現階段不得在報告中冒充已執行。方法升級或對外發布前，可讓 Gemini 以一次 720p、`thinking=minimal` 的 9:16 成片請求檢查語意、重要文字、重複與銜接，但它不回精確時間、不取代逐 sample geometry，也不取代真人最終核准。模型回報 `pass` 時，本機仍必須以 required-region coverage、containment、controlled clip、fallback 與 source-edge gate 推導最終狀態；幾何 gate 有風險就固定為 `review`。

本機再以 FFmpeg 將 coarse in／exclusive-out 解析到原始來源：入點保存第一張保留 decoded frame 的 PTS／hash；一般出點保存第一張不保留 decoded frame，片尾沒有下一張影格時則明確保存 EOS time boundary，不偽造 frame hash。2／4／8 FPS DF contact sheet 只在快速 UI、短暫動作、低信心或人工質疑邊界時局部啟用，讓 Gemini 從既有 ID refine；不以全事件 dense 抽格取代影片理解。相鄰 handles 另外保存供人檢查，不能因模型建議而丟棄原片。

靜止片尾不自動視為廢尾。模型只能以可見證據提出 `natural_pause`、`intentional_hold`、`title_safe_hold`、`clean_plate`、`reset_or_false_end` 或 `uncertain`；「疑似刻意」仍不是導演意圖的 ground truth。每份 proposal 都固定 `requires_human_review=true`，並輸出可播放 preview。只有 `review-trim` 寫入明確真人核准紀錄後，`feature-cut --trim-decision` 才會套用；未核准、被拒絕、跨 shot、source hash 不符或多筆重疊都 fail closed。沒有 matching reviewed decision 的段落仍保留原本的 keyframe-centered rough trim，且 manifest 明確標成 fallback。

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
  --candidate-id selected-subject \
  --runs 3 --ground-runs 1 \
  --output-dir ARTIFACT/selected-subject
```

也可直接提供 target：

```bash
uv run jascue-video-lab direct-moment-repeat ARTIFACT \
  --target-id selected-subject \
  --target-description '中央偏左、具有指定標記的前景實體；排除背景描繪、反射與其他相似實例。' \
  --runs 3 --ground-runs 1 \
  --output-dir ARTIFACT/selected-subject
```

若直接執行 `direct-moment-repeat` 卻沒有任何 target，CLI 會只產生候選並停止，不會偷偷挑一個物件進行 bbox。

Full v1 的 selected event 也可以直接使用版本化 QueryLock；若 lock 有多個 targets，必須另外明選一個 ID：

```bash
uv run jascue-video-lab full-ground-event ARTIFACT EVENT_ID \
  --query-lock query-lock.json \
  --query-target-id selected-subject \
  --sam-checkpoint artifacts/models/sam2.1_hiera_tiny.pt
```

若 exact-frame Grounding 回傳多個合理 bbox，命令會 fail closed；人工看過 debug 圖後才能以 `--grounding-candidate-number` 指定候選。此編號從 1 開始，與 debug 圖上的 `1.`、`2.` 一致；artifact 同時保存 1-based number 與 0-based array index。QueryLock、Grounding request 與 bbox seed 都會各自保存 fingerprint。

Trim Intent 採相同的 evidence-first 邊界：模型先回 coarse `MM:SS`，本機回查 PTS；必要時才升級成局部 frame-ID refinement，最後仍由真人決定是否採用。它處理的是入選片段的動作完整度與可疑 hold，不取代全庫 take grouping 或「多次重拍中哪一個最好」的比較任務。

## 可以分享的結論

> 影片 AI Grounding 最容易被忽略的問題，不是「框得準不準」，而是「它框的是不是使用者要的那一個」。我的實驗把流程拆成 Target Candidates → Human Query Lock → MM:SS Moment → Exact Frame PTS → Gemini bbox → 選配的 shot-local SAM。沒有指定目標時，AI 只提候選，不替人做最後選擇；多個 bbox 也不靠 confidence 偷選。同一影片在 Gemini File API 的有效期內會重用，只有確認過期才重傳。這讓錯誤更容易被看見，也讓每一步都能獨立驗證。

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

首次匿名化展示測例的 live A/B 使用相同前景實體 target 與 `00:02`：原始 2.002 秒單幀方法為 `[412, 684, 467, 871]`，direct-video 方法為 `[413, 664, 466, 842]`，IoU 0.738123、normalized center distance 24.5。Direct-video 選中相同實例，但框較短且實際影片取樣幀未知，所以這項結果只能說「值得繼續 A/B」，不能取代原始單幀 Grounding。視覺檢查由 Codex 執行，尚未成為獨立 human annotation。
