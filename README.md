# JasCueVideoLab

這是一個**完全獨立、實驗性**的 Gemini 3.5 Flash 影片理解與單幀 Grounding 驗證專案。它不是 JasCue 正式產品，不引用也不修改任何 JasCue 程式碼；實驗未通過前，不應將這裡的程式合併回 JasCue。

最新方法採用「未指定 target 就先提出候選，使用者以 QueryLock 鎖定實例與可選事件條件後才找時間與 bbox」，完整說明見 [METHODOLOGY.md](METHODOLOGY.md)。毛片 coarse-to-fine 全量流程見 [FULL-VERSION-PLAN.md](FULL-VERSION-PLAN.md)，目前已完成 per-clip Full v1 垂直切片與批次入口。Gemini polygon 與 bbox seed 的舊 A/B 僅保留為唯讀歷史資料；目前支援路徑只使用 Gemini／人工 bbox → SAM。

目前的最小垂直切片是：

```text
本機影片 → ffprobe / SHA-256 → Gemini File API
        → Gemini Interactions API Structured Content Map
        → 點選 HTML 事件 → FFmpeg 抽 orientation-corrected 原始影格
        → Gemini Interactions API Structured bbox
        → Pillow debug overlay
```

## 毛片挑帶與雙比例粗剪實驗

一般人版本：把一整個拍攝資料夾交給程式後，本機先做有固定編號的低解析度「看帶影片」；Gemini 只挑它看中的編號與說明用途，不負責猜精確剪輯時間。程式再從編號查回原片位置，用 FFmpeg 的真實時間與切鏡邊界取出乾淨片段，分別組成 16:9 與 9:16 人工審核版。

```text
原始毛片（不上傳整批 4K）
  → 本機 ffprobe／SHA-256／每 2 秒代表幀
  → 烙印 immutable frame ID 的低解析 analysis reel
  → Gemini Structured Output 只選 frame ID、用途與直式構圖意圖
  → 本機 frame ID 映射回原片時間
  → FFmpeg scdet 的 decoded-frame PTS 限制片段不跨硬切鏡
  → 輸出 16:9／9:16 silent rough cuts 與 HTML review page
```

這個設計刻意不把 Gemini timestamp 當 cut point。模型回傳的 frame ID 必須存在於 catalog，Pydantic contract 才會接受；實際 `source_in_ms`／`source_out_ms` 由本機資料生成並 clamp 在單一 shot。舊版 rushes rough cut 的 9:16 仍只有 `left`／`center`／`right` 三種固定構圖意圖，不是逐幀 tracking；需要動態構圖時，應改走下方 exact-frame bbox → shot-local SAM 2.1 mask propagation 路徑。

## Full v1：完整逐片 Clip Card，按需才做 geometry

Full v1 不會把整支毛片切成數百張圖片送入模型。每支影片先建立 720p analysis proxy，讓 Gemini 看完整影片並以 Structured Output 寫一張 `MM:SS` Clip Card；音訊採選配，預設 `auto` 是來源有音軌才保留，沒有音軌就只依視覺分析。FFmpeg shot detection 只保存切點資料與每個 shot 一張 960px 中間 JPEG 供稽核。只有使用者或剪輯 brief 選中事件、且確實需要 9:16 reframe／callout／去背時，才從原始影片抽一張 exact frame 取得 bbox，並選配 SAM 2.1。SAM 現在會在初始化前限制到 `允許區間 ∩ seed shot`，但仍需完成下方其他 production-readiness gate 才能作正式剪輯輸入。

```text
毛片資料夾
  → 每支 720p proxy（音訊 auto／off／required）
  → Gemini 完整觀看 → MM:SS Clip Card
  → 本機驗證事件、Entity、target kind 與片長
  → Clip Cards 可重用於不同剪輯 brief

只有選中的事件需要空間座標時：
  → 人工鎖定 target／排除條件／可選 predicate
  → FFmpeg 抽原始 exact frame／PTS／hash
  → Gemini image bbox（多候選需人工指定）
  → SAM 2.1 bbox-only、shot-local mask propagation

只有快速 UI／短暫狀態不確定時：
  → 事件內 1–5 秒局部 4／8 FPS frame-ID contact sheet
  → Gemini 只選既有 ID；時間仍由本機映射

只有入選片段需要精修頭尾時：
  → Clip Card coarse event ∩ FFmpeg shot
  → 局部 2／4／8 FPS immutable DF IDs
  → Gemini 標記 setup／action／result／hold／reset 與建議 in／exclusive-out ID
  → 本機將 ID 映射為 decoded-frame PTS、半開區間與安全 handles
  → 產生 proposal preview，真人核准後才可套入 feature cut
```

```bash
# 一支完整毛片；預設不做 dense、不做 bbox/SAM
uv run jascue-video-lab full-clip VIDEO.mp4 \
  --output-dir artifacts/full-v1-clip

# 一個毛片資料夾逐片建立可續跑 Clip Cards
uv run jascue-video-lab full-library /path/to/rushes \
  --output-dir artifacts/full-v1-library

# 已有 feature plan 時，只替實際入選的 source clips 建立 Clip Cards
uv run jascue-video-lab full-selected \
  artifacts/my-rushes-run/catalog.json \
  artifacts/my-feature-cut/gemini-plan/feature_edit_plan.json \
  --prepared-library artifacts/full-v1-library-prepared \
  --output-dir artifacts/my-selected-clip-cards

# 只有被選中的事件才抽原始影格並選配 SAM
uv run jascue-video-lab full-ground-event \
  artifacts/full-v1-library/clips/ASSET_PREFIX EVENT_ID \
  --query-lock examples/evidence-query-lock.json \
  --query-target-id subject.primary \
  --sam-checkpoint artifacts/models/sam2.1_hiera_tiny.pt

# 明確要求某個短暫事件進入 4／8 FPS 局部 fallback
uv run jascue-video-lab full-clip VIDEO.mp4 \
  --dense-event EVENT_ID --dense-fps 8 --dense-window-ms 4000 \
  --output-dir artifacts/full-v1-clip

# 入選事件的 trim intent；只產生待審 proposal，不會自動核准
uv run jascue-video-lab trim-event \
  artifacts/full-v1-library/clips/ASSET_PREFIX EVENT_ID \
  --sampling-fps 4 \
  --editorial-intent '保留完整動作與結果；標記可疑 hold、reset 與品質風險。' \
  --output-dir artifacts/trim-review/EVENT_ID

# 真人看過 index.html／trim-preview.mp4 後，明確核准或拒絕
uv run jascue-video-lab review-trim \
  artifacts/trim-review/EVENT_ID/trim-decision.json \
  --decision approved --reviewer REVIEWER_ID \
  --notes '已確認動作完整，片尾停留可保留。' \
  --output artifacts/trim-review/EVENT_ID/trim-decision.reviewed.json
```

Trim Intent 不把「畫面變靜」直接等同廢尾或刻意留白。模型只能依畫面提出 `natural_pause`、`intentional_hold`、`title_safe_hold`、`clean_plate`、`reset_or_false_end` 或 `uncertain`，並保存可見證據與不確定性；它不能宣稱知道導演意圖。預設流程讓 Gemini 直接觀看完整 proxy，在指定 Clip Card event／shot 內回傳 coarse `MM:SS` 代表性 select；FFmpeg 只抽入點與 exclusive-out 邊界，將其解析到原始影片的 decoded PTS。Gemini proposal 永遠是 `requires_human_review=true`，因此 schema 通過也不會直接改動正式成片。

4／8 FPS dense DF contact sheet 現在是局部升級手段，不是預設 Trim Intent：只有快速手勢、短暫 UI 或真人對 coarse 邊界有疑義時，才在小視窗內讓模型從既有 exact frame ID 選擇。不得把整支毛片拆成大量圖片來取代影片理解。若 Gemini 只回傳 hold 的單側端點，系統不會推測另一端，而會捨棄不完整 hold interval 並把 contract normalization 寫入 uncertainties；若 exclusive out 位於片尾且沒有下一張 decoded frame，則保存明確的 end-of-stream time boundary，而不偽造 frame hash。

`--audio-mode auto` 是預設值：有音軌就保留，無音軌也正常完成；`off` 明確移除音訊；`required` 只適合音訊證據不可缺少的實驗，來源沒有音軌時會保存錯誤並停止該片。artifact 會記錄 `source_has_audio` 與 `proxy_has_audio`，Clip Card 不得為 silent source 捏造 audio evidence。

Clip Card response reuse 會驗證 source hash、proxy hash、模型、schema、prompt fingerprint 與實際保存的 raw request；prompt 改變一定重跑。File API cache 以 exact proxy SHA-256 跨 library 共用，並在每次使用前查詢遠端 `ACTIVE` 狀態；不同編碼／解析度的 proxy、原始 4K 與整批 analysis reel 不會互相冒用。成本報告分成本次新增請求 `execution-pricing.json` 與含歷史的 artifact lifetime `pricing.json`。公開 library index 不含使用者名稱、絕對路徑或原始檔名；這些資訊只保存在 gitignored private manifest。

這不代表所有階段 cache 都已達 production 級。Exact-frame Grounding 與 bbox→SAM 已使用包含 source/frame、target、prompt/schema/model、shot bounds、checkpoint 與處理參數的 variant fingerprint；較早的 proxy、shot／dense catalog 仍有部分沿用「檔案存在」式重用。完成全鏈路 fingerprint 前不要在同一 output directory 偷換來源或參數，也不要把 cache hit 當成內容身分已驗證。

若執行環境禁止批次外傳，可先完全離線準備；此模式不建立 Gemini client：

```bash
uv run jascue-video-lab full-library /path/to/rushes \
  --prepare-only --output-dir artifacts/full-v1-library
```

之後在允許連線的環境移除 `--prepare-only` 重跑同一 output directory，會重用 proxy、shot manifest 與 audit frames，只執行尚未完成的 File API／Clip Card 階段。

若批次上傳被政策阻擋，但已經有一份 feature plan，可以先用 `full-selected --prepare-only` 在本機解析實際入選的 clip IDs。此模式只驗證既有 prepared proxies，完全不建立 Gemini client；之後在使用者自己的允許環境，以相同指令移除 `--prepare-only`，依序處理入選素材，而不是重跑整個資料夾。

### Production-readiness gate

目前最可信的輸出是「可搜尋的 Clip Card」與「exact-frame bbox proposal」。下列狀態不代表已形成 production 自動剪輯器：

1. **已完成核心 contract**：SAM predictor 的實際輸入只含 `允許區間 ∩ seed shot`，不跨切鏡傳播。
2. **已完成核心 contract**：多候選不取最高 model confidence；自動 seed 只接受唯一 `matched` candidate，其餘必須人工指定。
3. **部分完成**：exact-frame Grounding 與 bbox seed/SAM 有完整 fingerprint，仍要補齊較早的 proxy、shot 與 dense cache。
4. **部分完成**：每個新 SAM sample 可回映原始 decoded source PTS，但 renderer 的 in/out、seed、track、crop 核准狀態與週期性 identity revalidation 尚未完成。
5. **已完成效率／一致性 contract**：同一 shot 內的多個 bbox target 可共用一次 decoded-frame catalog、predictor 與 SAM inference state；每個 target 仍保存獨立 seed、mask、狀態與 provenance。共享與獨立執行可用逐格 mask agreement 自動比較，但 agreement 不是 ground truth。

另外，silent source 不得生成 audio evidence、失敗但已有 usage 的 API response 仍必須計價、公開匯出需採 allowlist sanitizer。完整測試還要加入 non-zero PTS、VFR、B-frame、rotation/edit-list、快速 UI 命中，以及相似物件跨鏡 identity-switch 等 fixture。

```bash
# 一次完成 catalog、Gemini selects、雙比例粗剪、review HTML 與成本／計時
uv run jascue-video-lab rushes-run /path/to/CLIP \
  --sample-interval-ms 2000 \
  --scdet-threshold 4 \
  --output-dir artifacts/my-rushes-run

# 不呼叫 Gemini，只建立可重用的 catalog／analysis reel
uv run jascue-video-lab catalog-rushes /path/to/CLIP \
  --sample-interval-ms 2000 \
  --output-dir artifacts/my-rushes-run

# 單支影片保存 FFmpeg scdet 的精確 decoded-frame PTS
uv run jascue-video-lab detect-shots VIDEO.mp4 --threshold 4 --output shots.json
```

一批經授權的多片毛片已完成端到端 live test，涵蓋本機 catalog、analysis reel、Gemini frame-ID 選片、雙比例 review cut、成本與時間記錄。測試證明 frame ID 可以穩定回映原片，卻不代表模型選片已達人工剪輯品質；實際費用仍取決於素材長度、輸入 token、重跑次數、方案與當時牌價。

兩秒抽樣只適合舊版第一輪粗看帶，不能當成泛用的唯一視覺取樣。Full v1 已改為完整 proxy Clip Card；0.2–0.5 秒 UI、快速手勢與短暫對焦狀態則可在指定事件與單一 shot 內建立 4／8 FPS immutable dense frame IDs。dense fallback 預設關閉，不會把整支影片或整個資料夾全量抽成圖片。

### Brief-ordered feature cut 與安全 Reframe

固定 `left`／`center`／`right` crop 已被實驗性 9:16 輸出證明不可靠：人物或指定物件移動後仍可能被裁掉。`feature-cut` 改以使用者提供的章節 brief 控制敘事順序，Gemini 分別選橫式／直式 take 與明確 reframe target，再以 exact-frame image Grounding + SAM 2.1 mask propagation 約束 16:9 punch-in 與 9:16 crop。

```text
使用者功能 brief（文案事實來源）
  → Gemini 只找每章的可見影片證據與 frame IDs
  → FFmpeg shot PTS 決定 source handles
  → 指定主體 exact-frame Gemini bbox
  → SAM 2.1 在單一 shot 內傳播 mask
  → 16:9：剪輯 zoom intent ∩ mask 安全倍率 ∩ 4K→1080 解析度上限
  → 9:16：平滑 tracked crop；strict 與 primary-center 分開記錄
  → 可選字卡 + 原始現場音 + H.264/AAC review cuts
```

SAM 只提供幾何，不自行決定剪輯美學。16:9 的 `none`／`subtle`／`detail` 由 feature plan 表示 editorial intent，實際倍率不得超過 mask 安全值。9:16 的 `strict` 要求完整保留指定物件；人工 `primary_center` 則允許犧牲次要人物或 context，集中呈現指定主視覺。這個 sacrifice 必須寫入 brief／manifest，不能由 tracker 默默決定。使用者 brief 的規格文字與模型觀察到的畫面證據分開保存，沒有 ASR 或 transcript。

```bash
uv run jascue-video-lab feature-cut \
  artifacts/my-rushes-run/catalog.json \
  BRIEF.json \
  --sam-checkpoint artifacts/models/sam2.1_hiera_tiny.pt \
  --sam-analysis-fps 2 \
  --output-dir artifacts/my-feature-cut
```

若某章已有真人核准的 Trim Intent，可重複傳入 `--trim-decision PATH`。Renderer 只接受 `approval_status=approved` 且帶有人類 review record 的 decision，並再次驗證 source SHA-256 與目前 FFmpeg shot；代表性 select 可以位於同一 source shot 中但不包含較早的 coarse RF anchor。proposed、rejected、跨鏡或同 source shot 多筆造成歧義的 decision 會被拒絕。沒有匹配 decision 的章節仍使用原本「keyframe 中心 ± brief duration、限制在 shot」的粗剪方式，manifest 會分別標示 `human_approved_frame_id_pts` 或 `keyframe_centered_requested_duration`，不會把 fallback 冒充成精修結果。

若目的是先產生影片讓真人整體觀看，可明確加入 `--allow-proposed-trim-preview`。這只接受仍為 `proposed` 的可用 decision，輸出 manifest 會標記 `contains_unreviewed_trim_proposals=true`，每段也標成 `unreviewed_proposed_frame_id_pts`；它不能建立人工 review record、不能接受 rejected decision，也不能冒充正式核准 cut。

```bash
uv run jascue-video-lab feature-cut \
  artifacts/my-rushes-run/catalog.json BRIEF.json \
  --sam-checkpoint artifacts/models/sam2.1_hiera_tiny.pt \
  --trim-decision artifacts/trim-review/event-a/trim-decision.reviewed.json \
  --trim-decision artifacts/trim-review/event-b/trim-decision.reviewed.json \
  --output-dir artifacts/my-feature-cut-reviewed-trims
```

經授權的多功能產品素材已完成 16:9 與 9:16 live review-cut 實驗：橫式版本只在通過 geometry gate 時套用有限 reframe，直式版本則以使用者明確指定的 `strict` 或 `primary_center` 規則處理動態 crop，不使用模糊背景掩蓋構圖失敗。Grounding schema 通過、bbox/contact sheet 經視覺檢查，仍只代表流程可稽核，不代表每個構圖或選片已獲獨立真人核准。

Feature renderer 同樣接受無音軌來源：有原音時保留並淡入淡出，無音軌時為 review segment 明確合成 deterministic stereo silence，讓所有 segment 維持一致的 A/V concat contract；manifest 會標示 `audio_origin=source` 或 `synthetic_silence`，不把靜音說成來源音訊證據。

若 geometry 與片段已經渲染，只想比較另一種敘事順序，不需要再呼叫 Gemini。`scripts/resequence_segments.py` 讀取明確的 trim/sequence JSON，重新編排現有編號 A/V segments，並輸出包含每段來源、trim 與新時間軸的 manifest。這只適合可稽核的 picture-edit A/B；它不會把既有片段描述冒充成新 Full Clip Card，也不能取代原片層級的 take selection。

完整的 Clip Card-driven A/B 則分成兩次 Gemini 任務：第一輪逐片產生 Clip Cards；第二輪只讀已驗證 Clip Cards 與使用者 brief，輸出 Structured narrative plan。`scripts/plan_selected_clip_cards.py` 實作第二輪，`scripts/render_clip_card_narrative.py` 只從通過 evidence gate 的 source/event/MM:SS 建立 16:9 review cut。第二輪仍可能產生規格換算錯誤，Clip Card 也可能把局部可見的數字或型號字元誤判成另一個相似值。因此任何 OCR／身分衝突只能觸發 `needs_human_review`，必須回查 orientation-corrected 原始影格後才能採用或排除；schema validation 不能取代 claim validation，也不能把模型 OCR 當成 ground truth。

`scripts/plan_clip_card_feature_cut.py` 將這個方法延伸到完整 feature cut：模型可閱讀整個已驗證 Clip Card library，但只能選 catalog 中既有的 asset／event／RF frame ID；本機會再次驗證影格確實屬於該素材且位在事件區間，才投影成 `feature-cut` 可使用的 plan。選片階段不產生 bbox 或剪點，只有真正入選、需要動態構圖的區間才執行 exact-frame Grounding 與 SAM。實務上的幾何降級順序應保留多個候選：先允許 brief 明示的少量主體裁切，再嘗試同事件的另一 seed／更明確 target，其次換用候選素材，最後才採本機 fallback。`vertical_fallback_strategy=center_crop` 可明確禁止模糊背景；所有 fallback 原因仍寫入 render manifest，不能冒充成功追蹤。

主要影片／圖片辨識請求另使用 Interactions API `system_instruction` 建立 evidence-only 邊界：本次媒體與明確 metadata 是唯一證據，禁止以模型記憶、常見名稱、相似外觀或「最可能答案」補完品牌、型號、數字與 UI 文字。Full Clip Card prompt 也要求任一關鍵字元不清楚時改用泛稱並保存 uncertainty。控制 A/B 曾觀察到舊 prompt 以先驗補完一個相似但錯誤的型號；改用 domain-neutral 規則後該欄位在重跑中恢復正確，模型卻又把另一處模糊小字補成畫面不存在的規格。這證明 prompt guardrail 不是 ground truth：單一正確 claim 不代表整張 Clip Card 都正確，衝突與重要文字仍需 exact-frame 驗證及人工核准。

`scripts/verify_clip_card_text.py` 實作不覆蓋原始 Clip Card 的文字驗證：從原片抽多張 exact frames、保存 PTS／hash、裁出文字證據，以 `resolution=high` 分別做 blind transcription，再以明列 `other`／`unreadable` 的候選式請求交叉檢查。方法不一致時輸出 `needs_human_review`；只有人工核准後才能另外產生 reviewed Clip Card。

## 重要界線

- `start_ms`、`end_ms` 與 `recommended_keyframe_ms` 是 **coarse semantic time**，只用於搜尋與人工瀏覽，不是 frame-accurate cut point。
- 對 Gemini 原生影片理解索取少量截圖候選時，API contract 使用官方文件慣例 `MM:SS`，不要求模型計算毫秒。程式只把合法且未超界的 `MM:SS` 換算成 FFmpeg seek 值；換算結果仍不是精確 frame time。
- `frame_pts` 與 `frame_time_ms` 是 FFmpeg 實際抽到之原始影格的媒體時間；每張 `frame.png` 都另外保存 SHA-256。
- Gemini bbox 是單張影格的人工審核 proposal，不是 pixel mask，也不是 production-ready tracking data。
- `main` baseline 沒有 ASR、transcript、字幕、temporal tracker、SAM/EdgeTAM/Apple Vision、逐幀追蹤、自動裁切、NLE timeline、FCP/Motion/FxPlug 或成片輸出。
- `experiment/dynamic-tracking` branch 另有一條明確隔離的 optional CSRT bbox propagation 實驗。它不屬於 baseline，也不得把輸出稱為 Gemini 原生 tracking 或正式 SpatialTrack。
- `experiment/sam21-video-segmentation` branch 把 Gemini／人工 bbox 當語意 seed，交由 SAM 2.1 產生並傳播 mask；原始 seed、SAM prompt box、mask 與 mask-derived bbox 分開保存。
- `experiment/gemini-segmentation-seed` 曾測試 Gemini 原生 polygon 作為 SAM mask seed。A/B 後已從目前主路徑退休：執行入口會拒絕 polygon，歷史 artifact 只用於說明為何選擇 bbox seed。
- Live 成本與時間資料會分開記錄 analysis proxy、Gemini raw usage 牌價估算、API latency 與 tracker geometric drift。成本只依官方 Standard list price 估算；free tier 與沒有 usage response 的失敗請求不得假裝成已知帳單金額。
- Interactions API 的影片視覺處理預設約 1 FPS；官方目前未在 Interactions API 開放 `video_metadata` 自訂 FPS。因此 0.2–0.5 秒 UI 狀態可能漏掉。本實驗以完整影片 Content Map 對照「抽出的原始單幀 Grounding」量測這個限制，不把未觀察到的狀態靜默補上。

官方依據：

- [Interactions API overview](https://ai.google.dev/gemini-api/docs/interactions-overview)
- [Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Video understanding / File API](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Image understanding / Gemini 原生 bbox 座標順序](https://ai.google.dev/gemini-api/docs/image-understanding)
- [google-genai Python SDK](https://googleapis.github.io/python-genai/)

## 環境

需求：Python 3.12、`uv`、FFmpeg/ffprobe，以及 Gemini API key。

```bash
cd ~/Experiments/JasCueVideoLab
UV_CACHE_DIR=.uv-cache uv sync --python 3.12
export GEMINI_API_KEY='...'
```

若執行環境不會繼承 terminal export，可在專案根目錄建立已被 `.gitignore` 排除的 `.env.local`，內容只放 `GEMINI_API_KEY=...`，執行前先 `source .env.local`。不要把 key 貼進 issue、artifact 或 commit。

只使用官方 `google-genai` SDK；模型 ID 固定為穩定版 `gemini-3.5-flash`。程式不依賴已淘汰的 `google-generativeai`，也沒有舊 Gemini model ID。

## 本機 Blind Review Web App

不想透過 Codex 代為判讀時，可直接啟動 human-first 審核介面：

```bash
cd ~/Experiments/JasCueVideoLab
set -a; source .env; set +a
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab serve-review
```

瀏覽器開啟 `http://127.0.0.1:8765`，直接拖入影片。預設只綁定本機 loopback，沒有登入系統；除非完全理解風險，請勿使用 `--allow-network` 對區網開放。

App 的固定順序是：

1. 影片串流寫入本機，ffprobe 與 SHA-256 驗證後建立 session。
2. 可選擇建立 1080p／30fps analysis proxy；Gemini 語意分析使用 proxy，bbox 仍從原始影片抽幀。
3. 沒有 target 時只顯示候選卡，且不預選、不顯示候選 confidence。
4. 使用者選擇候選或自行輸入精確 target，才允許產生 target-locked `MM:SS` 時刻。
5. 選一個時刻後，FFmpeg 保存原始 frame PTS，再執行單幀 Grounding。
6. Blind review 只顯示 `Candidate A/B` 框；提交「正確／錯物件／太大／太小／不可見／無法判斷」前，reveal API 會拒絕提供模型 label、confidence 與理由。
7. 可在畫面拖曳人工修正框；人工判定寫入後才能揭露完整 Gemini proposal。
8. `匯出完整 JSON` 包含 media identity、human annotations、已揭露 proposals 與尚未審核清單。

每個時刻提供兩個隔離模式：A 是預設且可驗證的「FFmpeg exact frame → image Grounding」；B 是實驗性的「完整影片 → 指定 `MM:SS` → bbox」。Google 官方明確文件化的是 image object detection bbox，而 File API 影片預設以 1 FPS 保存／處理；官方沒有提供 B 模式實際採用 frame 的 PTS 或 hash。因此 B 的 contract 永遠標記 `unknown_gemini_video_sample`，投影到 FFmpeg frame 的圖只供 A/B 診斷，不能成為 production geometry。兩種方法都經獨立盲審後，export 才會計算第一候選 bbox IoU 與 center distance。

一個經授權的真實短片測例曾讓 A、B 兩種模式選中相同指定實例，但兩個 bbox 仍有明顯幾何差異。公開文字不揭露原始檔名與私人路徑；媒體本身仍可能含人物、品牌與活動場景，不能稱為已去識別化。這是模型輔助視覺檢查，不是獨立 human ground truth；B 模式的 reference frame 仍不可知，因此不能用來建立正式 tracking seed。

持久資料位於被 Git 排除的 `artifacts/blind-review-app/<session-id>/`；跨 session 的 Gemini File API cache 依 analysis source SHA-256 位於 `artifacts/blind-review-file-cache/`。同一 upload identity 在官方 48 小時保存期內會重用。App 不會把 API key 傳到瀏覽器，也不以 browser storage 當實驗資料來源。

## 產生四種真實影片 fixture

Fixture 是由 Pillow 畫面經 FFmpeg 編碼而成的真實 MP4，不是 API mock。A 是 30 秒無旁白手機 UI 操作；B 是人物與手機同時移動的 16:9 畫面；C 含 0.3 秒快速按鈕狀態；D 含硬切鏡與兩支相似手機。

```bash
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab make-fixtures
```

每支影片都會生成 `.media.json`，包含 duration、coded/display dimensions、rotation、frame rate、time base、stream/format metadata 與 SHA-256。參考標註在 `fixtures/annotations/`；每份標註必須聲明 author type、方法與是否經獨立真人確認。`real_continuity_public.json` 目前由 Codex 視覺檢查後手動輸入，**不是 independent human ground truth**。

## 實跑垂直切片（三次）

```bash
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab run \
  fixtures/generated/A_silent_phone_ui.mp4 \
  --runs 3 \
  --ground-per-event 1 \
  --annotations fixtures/annotations/A_silent_phone_ui.json
```

若要依建議測試 B 對同一關鍵幀分別框手機、手機螢幕、手、臉與另一支手機，可把 `--ground-per-event` 提高；實際數量仍取決於 Content Map 中該事件建立的 Entity。

其他命令：

```bash
# 只做媒體探測
uv run jascue-video-lab probe VIDEO.mp4 --output media.json

# 抽取 >= 2.8 秒的第一張原始影格，旁邊會寫 frame.png.json
uv run jascue-video-lab extract VIDEO.mp4 2800 frame.png

# 從既有結果重建 timeline
uv run jascue-video-lab timeline ARTIFACT/run-01 ARTIFACT/source.mp4

# 比較任意多次執行
uv run jascue-video-lab compare ARTIFACT/run-01 ARTIFACT/run-02 ARTIFACT/run-03 \
  --output ARTIFACT/comparison.json --annotations fixtures/annotations/A_silent_phone_ui.json

# 對同一張已抽出的原始影格重跑 Grounding
uv run jascue-video-lab ground-repeat ARTIFACT FRAME.png.json \
  --event-id EVENT --event-description DESCRIPTION --entity-id ENTITY \
  --target-description TARGET --runs 5 --output-dir OUTPUT

# 不讓 Gemini 產生時間數字：FFmpeg 建立 PTS 網格，Gemini 只選 frame ID
uv run jascue-video-lab storyboard-temporal ARTIFACT \
  --interval-ms 4000 --output-dir ARTIFACT/storyboard-pts-grid-4s-live

# 讓 Gemini 推薦少量官方 MM:SS 截圖時刻，本機驗證、抽幀並 Grounding
# 若未提供 target，這個命令只會提出候選並停止，不會自行挑物件 Grounding
uv run jascue-video-lab direct-moment-repeat ARTIFACT \
  --runs 3 --ground-runs 1 --output-dir ARTIFACT/direct-mmss-3runs-live

# 明確的候選階段；沒有 bbox，也不做 tracking
uv run jascue-video-lab suggest-targets ARTIFACT \
  --output-dir ARTIFACT/target-candidates

# 使用者選定候選後，鎖定 target 才找時間與 Grounding
uv run jascue-video-lab direct-moment-repeat ARTIFACT \
  --candidate-map ARTIFACT/target-candidates/run-01/target_candidates.json \
  --candidate-id selected-subject \
  --runs 3 --ground-runs 1 --output-dir ARTIFACT/selected-subject

# 僅限 experiment/dynamic-tracking branch；用 Gemini GroundingProposal 當 seed
uv sync --extra tracking
uv run jascue-video-lab track-csrt VIDEO.mp4 \
  --grounding-json ARTIFACT/events/EVENT/groundings/ENTITY/grounding.json \
  --target-description '審核者指定的前景實體；排除背景圖像與相似實例' \
  --output-dir ARTIFACT/tracking-selected-subject

# 僅限 experiment/sam21-video-segmentation branch；官方 checkpoint 不進 Git
SAM2_BUILD_CUDA=0 UV_CACHE_DIR=.uv-cache uv sync --extra segmentation
mkdir -p artifacts/models
curl -L -o artifacts/models/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Gemini bbox → SAM seed mask → 向前／向後 mask propagation
uv run jascue-video-lab track-sam21 VIDEO.mp4 \
  --checkpoint artifacts/models/sam2.1_hiera_tiny.pt \
  --grounding-json ARTIFACT/grounding.json \
  --target-description '審核者指定的前景實體；排除背景圖像與相似實例' \
  --analysis-fps 2 --output-dir ARTIFACT/sam21-selected-subject

# 同一 shot 的多個 bbox seed 共用一個 SAM predictor／inference state
# targets.json 必須含兩個以上、target_id 唯一的 bbox-only targets
uv run jascue-video-lab track-shared-sam21 VIDEO.mp4 \
  --checkpoint artifacts/models/sam2.1_hiera_tiny.pt \
  --targets-json ARTIFACT/targets.json \
  --analysis-fps 15 --device cpu \
  --output-dir ARTIFACT/sam21-shared
```

`targets.json` 必須把每個 bbox 鎖到 upstream exact-frame Grounding 所聲明的 decoded source PTS，而不是只給可被重新四捨五入的毫秒：

```json
{
  "asset_id": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
  "targets": [
    {
      "target_id": "subject-a",
      "target_description": "審核者選定的第一個前景實體",
      "seed_source": "exact-frame Gemini bbox",
      "seed_time_ms": 5739,
      "seed_frame_pts": 344344,
      "seed_frame_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "seed_source_width": 3840,
      "seed_source_height": 2160,
      "seed_box_2d": [410, 370, 490, 570]
    },
    {
      "target_id": "subject-b",
      "target_description": "審核者選定的第二個前景實體",
      "seed_source": "exact-frame Gemini bbox",
      "seed_time_ms": 5739,
      "seed_frame_pts": 344344,
      "seed_frame_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "seed_source_width": 3840,
      "seed_source_height": 2160,
      "seed_box_2d": [520, 330, 575, 505]
    }
  ]
}
```

範例中的全零 hash 必須替換為實際值。原始 Grounding PNG 與 SAM 的 analysis JPEG 具有不同編碼，所以 hash 不同；dimensions 也可能因縮放而不同。追蹤器會把要求的 source PTS 強制加入固定間距抽樣序列，並以該 PTS 精確選 seed，不使用 nearest frame。此命令能驗證 PTS 與重解碼影格對齊；`seed_frame_sha256`、原始 dimensions 與 `seed_source` 則是呼叫端提供並保存的 upstream provenance，仍須由上游 Grounding bundle 驗證，不能由追蹤命令單獨證明 Gemini 看過該 PNG。

```bash
# 將共享 session 的 per-target tracks 合成一支正常播放速度的人工審核影片
uv run jascue-video-lab render-multi-sam21 \
  ARTIFACT/sam21-shared/targets/subject-a/segmentation-track.json \
  ARTIFACT/sam21-shared/targets/subject-b/segmentation-track.json \
  --label 'Selected subject A' \
  --label 'Selected subject B' \
  --analysis-frames-dir ARTIFACT/sam21-shared/analysis-frames \
  --display-fps 30 \
  --output-dir ARTIFACT/multi-track-review

# 比較兩條完全對齊的 SAM tracks；只量 agreement，不宣稱其中一條是真值
uv run jascue-video-lab compare-sam21-tracks \
  ARTIFACT/reference/segmentation-track.json \
  ARTIFACT/candidate/segmentation-track.json \
  --output ARTIFACT/sam-track-agreement.json

# 與 CSRT 比較只稱為 agreement；兩者都不是 human ground truth
uv run jascue-video-lab compare-trackers \
  ARTIFACT/sam21-selected-subject/segmentation-track.json \
  ARTIFACT/csrt/tracking.json --output ARTIFACT/tracker-agreement.json
```

`track-shared-sam21` 要求所有 seed 都落在同一個 FFmpeg shot，且不接受 Gemini polygon。它只共用重複的影片解碼、predictor 與 inference state；每個物件仍有自己的 object ID、bbox seed、mask 與 drift 狀態。每個 propagation output 會立即縮減成 binary mask 與小型統計並落盤，不會把 `影格數 × 物件數` 份 full-resolution float logits 全留在 RAM；缺格、重複格、越界格或缺物件則直接失敗，不會偽裝成 `lost`。`render-multi-sam21` 只會合併來自同一 asset、同一區間與完全相同 decoded-source PTS 取樣序列的軌跡；共享 session 必須顯式提供並驗證 immutable frame manifest，任一對齊資料不同就拒絕輸出。`--display-fps 30` 只控制審核影片播放時間，不表示 tracker 已在 30 FPS 上推論。產出為含來源音軌的 H.264/yuv420p MP4 與可追溯 manifest，只供人工審核，不是準確率證明或 production SpatialTrack。

## macOS SAM 2.1 runtime 狀態

目前可採用的 reference path 是 Meta 官方 SAM 2.1 PyTorch video predictor。它具備影片 memory、bbox prompt、多物件與雙向傳播；在 macOS 上可跑 CPU 或 MPS。實測顯示 MPS 能產生幾乎相同的 mask，但在目前的 Apple Silicon／PyTorch 組合上不一定更快，因此 `device=auto` 的結果不能取代目標機實測，正式預設仍以可重現 benchmark 決定。

Apple 發布在 Hugging Face 的 `coreml-sam2.1-tiny` 與 `coreml-sam2.1-large` 都是**單張圖片 segmentation** package。它們沒有可直接取代 SAM video predictor 的 temporal memory pipeline；Large 仍然只是較大的 image-only 模型，不能因名稱含 SAM 2.1 就當成影片 tracker。

EdgeTAM 的官方 PyTorch video predictor 同樣具備 temporal memory、bbox prompt、多物件與雙向 propagation。本機單一 fixture 的 MPS propagation 約為官方 SAM 2.1 Tiny MPS 的 2.87 倍速，但其中一個目標曾連續五格輸出空 mask，且目前依賴組合需要隔離的 tensor 相容修正；因此只列為需要人工 golden set 驗證的 experimental candidate，不取代 reference path。

EfficientTAM-Ti 也以相同的兩個 bbox、一個共享 state 與雙向 propagation 跑通；速度介於 EdgeTAM 與官方 SAM 2.1 Tiny CPU 之間，而且不需修改 upstream source。不過 MPS 無法使用其 CUDA-only 小孔洞後處理，所以同樣必須以人工 mask fixture 驗證邊緣品質，不能只看 throughput 或 peer IoU。

MLX 原生的完整 SAM 2.1 video predictor 也是值得繼續比較的 macOS 方向：它可保留 bbox、多物件與 memory propagation，且初步測速較 PyTorch 快。不過目前找到的社群實作仍很新，repository 的程式碼授權標示也尚未完整；因此只列為隔離的研究候選，不加入預設依賴。另一個僅支援 point、單向 propagation 或不能可靠共用多物件 state 的實作，不會用不等價條件加入 benchmark。詳細證據、MPS／CPU 實測、benchmark 限制與採用 gate 見 [MACOS-SAM21-EVALUATION.md](MACOS-SAM21-EVALUATION.md)。

## 產出

每次 `run` 會建立唯一 artifact 目錄：

```text
artifacts/<asset-sha-prefix>/<UTC timestamp>/
├── media.json
├── source.mp4 -> 原始影片
├── upload/
│   ├── file_upload_initial.json
│   └── file_upload_final.json
├── run-01/
│   ├── run.json
│   ├── content_map.request.json
│   ├── content_map.attempt-01.*              # 每次失敗／修正皆獨立保存
│   ├── content_map.attempt-02.*
│   ├── content_map.raw_interaction.json
│   ├── content_map.raw_output.json
│   ├── content_map.schema_validation.json
│   ├── content_map.json
│   ├── errors.json                         # 有錯才出現，不靜默吞掉
│   ├── index.html                          # 每個事件可點選播放
│   └── events/<event-id>/
│       ├── frame.png
│       ├── frame.json                      # requested time 與真實 PTS 分開
│       └── groundings/<entity-id>/
│           ├── grounding.request.json      # 不含 base64 圖片，只記 hash
│           ├── grounding.raw_interaction.json
│           ├── grounding.raw_output.json
│           ├── grounding.native.json         # Gemini 官方 y-first 座標
│           ├── grounding.coordinate_transform.json
│           ├── grounding.schema_validation.json
│           ├── grounding.json
│           └── debug.png
├── run-02/...
├── run-03/...
├── comparison.json
└── result.json
```

Gemini File API 物件依官方文件保存 48 小時。命令會先以已保存的 file name 查詢：仍為 `ACTIVE` 就重用；只有明確收到 `404/NOT_FOUND` 才重新上傳，其他不確定錯誤會保存並停止。`upload/file_cache.json` 記錄是否 reuse，舊 metadata 在重傳前移到 `upload/history/`；只有明確傳入 `upload --force-reupload` 才無條件重傳。參考：[Files API](https://ai.google.dev/gemini-api/docs/files)、[File input methods](https://ai.google.dev/gemini-api/docs/file-input-methods)。

API Structured Output 仍會由本機 Pydantic 再驗證。原始 Interaction response 與原始 `output_text` 都先保存；若 JSON 或語意 contract 失敗，錯誤、類型與 traceback 會寫入 `errors.json`，不會以假資料補值。請注意 request 使用 `store=false`，以本機 artifacts 作為實驗紀錄。

Gemini 官方 object detection 格式為 `[ymin, xmin, ymax, xmax]`，而本專案 canonical contract 依需求固定為 `[xmin, ymin, xmax, ymax]`。API boundary 因此使用明確命名的 `box_2d_yxyx`，通過 native schema 後再以純軸序重排轉成 `box_2d`；兩份 JSON 與 transform record 都保存。不得用框的長寬比例猜座標順序。

`comparison.json` 包含：Event 數量差、label 相似度、start/end 差、keyframe 差、第一候選 bbox center distance（0–1000 空間）、IoU、每份 schema validation 結果及 reviewer-reference 對照。歷史 JSON key `human_annotation_comparison` 目前為相容性保留，不代表已由真人標註。無候選或不可見 proposal 會保留為不可比較，不會捏造 bbox。

## 測試

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

Contract tests 驗證 schema、entity reference、半開事件區間、禁止 `frame_accurate`、不可見目標空候選與 bbox 範圍；geometry tests 驗證 normalized-to-pixel、center distance、IoU、mask-derived bbox、碎片 gate 與跨 shot latch；media test 會實際呼叫 FFmpeg 產生短片並確認推薦毫秒與真實 PTS 分離。這些測試不會 mock 或宣稱 Gemini/SAM live call 成功；live 成功只能由 artifacts 中的 raw response、validated JSON、mask 與 debug overlay 證明。

## Live 實驗結論

已使用多支經授權的真實素材完成重複 live run。公開文件中的素材描述已去識別化，但媒體本身仍可能含可辨識人臉、品牌與活動場景。原始素材、可辨識檔名、活動資訊與逐次報告只保存在 Git 排除的本機 artifacts；公開 README 只記錄可泛化的方法學結論：

- Content Map、Grounding Proposal 與 HTML timeline 的垂直資料流程可以完成，但 schema 合法不代表事件時間、物件身分或剪輯判斷正確。
- 初版曾把 Gemini 官方 y-first bbox 當成專案 x-first bbox，造成正確物件被畫成錯誤形狀。現行 API boundary 固定使用明確命名的 `box_2d_yxyx`，再由本機 deterministic conversion 轉成 canonical x-first；不以長寬比例做 heuristic auto-swap。
- 在目標層級、實例特徵及排除條件鎖定後，單幀 Grounding 的重跑可以相當穩定；這只能證明它適合作為待審核 bbox seed，不能取代獨立真人 ground truth 或成為 production tracking data。
- 完整 Content Map 曾在正確媒體 duration 下產生超界時間。縮小 schema 或以 contract-error feedback 修成合法數值，仍不能保證推薦影格具有正確語意。
- 少量顯著時刻可先採 Gemini `MM:SS`，經本機片長驗證後再由 FFmpeg 抽幀並保存真實 PTS；若時間非法、目標不可見或需要全片 coverage，改用帶 immutable frame ID 的本機 storyboard，讓模型只選既有 ID。
- 固定間距 storyboard 只提供 coarse coverage；快速 UI、短暫手勢與瞬時對焦狀態必須在候選事件及單一 shot 內使用更密的局部 frame-ID 網格，不能把固定抽樣結果當 frame-accurate cut point。
- 多個相似實例同框時，即使 bbox 重跑一致，也只能稱為模型穩定度。必須先由人工或 QueryLock 鎖定目標，且在不知道模型框的情況下建立 reference，才能計算有意義的準確率。
- OCR 實驗曾出現兩類錯誤：把模糊字元以既有知識補成相似名稱，以及在另一處小字捏造畫面沒有的規格。evidence-only system instruction 能降低但不能消除此問題；重要名稱、數字與 UI 狀態仍需 exact-frame 驗證與人工核准。

上述 IoU、重跑一致性與模型輔助抽查只能作為實驗診斷。完成獨立真人 blind review 前，不得宣稱人工 ground-truth 驗收通過。可攜式 HTML 的 build/schema 與瀏覽器載入檢查也只證明報告工具可用，不代表其中的模型判斷正確。

## 未來與 JasCue 的資料邊界

在人工審核與多次穩定度門檻通過後，可考慮轉成 JasCue **fixture** 的只有：

- 去識別化的測試影片及其 SHA-256/media metadata。
- 人工確認後的 coarse Content Map 事件、Entity 描述與不確定性案例。
- 人工確認的單幀 bbox 測試案例、schema contract 與 geometry 測試向量。
- 多次執行的比較報告，用於建立未來 regression threshold。

下列資料不得直接成為正式 SpatialTrack：

- Gemini 的 semantic timestamps 或推薦 keyframe。
- 未經人工確認的 bbox、不可見物件推測或相似物件選擇。
- `debug.png`、模型 confidence 或 label similarity。
- 單幀 bbox 串接、內插或任何假裝為逐幀 tracking 的衍生資料。
- 本實驗的 HTML timeline；它是審核工具，不是 NLE timeline。

任何移入 JasCue 的 fixture 都應經明確人工審核與獨立變更流程；本 repository 不提供也不執行合併回 JasCue 的命令。
