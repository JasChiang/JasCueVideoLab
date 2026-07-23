# JasCueVideoLab

這是一個**完全獨立、實驗性**的 Gemini 3.6 Flash 影片理解與單幀 Grounding 驗證專案。它不是 JasCue 正式產品，不引用也不修改任何 JasCue 程式碼；實驗未通過前，不應將這裡的程式合併回 JasCue。

## 一般人也看得懂的工作流程

假設手上有一整批還沒整理的拍攝毛片，這套實驗流程會先幫忙「看帶、整理、提出剪輯建議」，而不是一開始就直接把影片自動剪完：

1. **整理素材**：程式先讀取每支影片的長度、尺寸與切鏡等基本資訊，並建立較輕量的分析版本，不必反覆處理原始 4K 檔案。
2. **AI 看帶**：Gemini 逐支理解影片，整理成可重用的 Clip Card，記錄拍到了什麼、有哪些人物或物件、動作是否完整，以及可能適合放在哪一段。
3. **提出選片建議**：有剪輯 brief 時，AI 依指定主題、功能與片長挑選素材；沒有 brief 時，則先根據素材內容提出一版故事方向與候選片段。
4. **真人確認目標**：如果畫面裡有多個相似人物或物件，系統先提出候選，讓使用者確認真正要保留或追蹤的是哪一個，不讓 AI 在後續步驟自行換成相似目標。
5. **需要時才追蹤與重構**：一般接片不需要物件座標。只有要把橫式影片改成 9:16、跟隨人物或產品、避讓圖卡時，才從原片抽出清楚影格取得 bbox，再由 SAM 追蹤同一個鏡頭內的目標。
6. **需要時再規劃音樂卡點**：本機先分析音樂的節拍、重音、能量與段落變化；真人確認 BPM、第一個強拍與拍號後，程式才提出畫面事件和音樂 cue 的對齊建議。
7. **輸出人工審核版**：程式產生 16:9／9:16 review cut、構圖紀錄、卡點建議與失敗原因。真人看過選片、頭尾、裁切及節奏結果並核准後，才適合進一步完成正式剪輯。

```text
一批毛片
  → AI 看帶並建立 Clip Cards
  → 有 brief 就照需求挑片；沒有 brief 就先提出故事候選
  → 真人確認選片與重要目標
  → 只有需要直式重構或圖卡避讓時才做 bbox／SAM tracking
  → 有音樂時建立 MusicMap，真人鎖定音樂網格後再提出 CuePlan
  → 輸出可播放的人工審核版
  → 真人修改或核准
```

Clip Cards 建立後可以重複使用。同一批素材之後要剪成不同主題、長度或比例時，可以先查既有資料，只重新分析真正入選且需要精確畫面座標的片段。AI 的選片、時間、bbox、mask 與 confidence 都只是待審建議，不會因為 schema 合法就自動成為正式剪輯資料。

### 用到哪些技術

| 技術 | 在這個流程裡負責什麼 | 不負責什麼 |
| --- | --- | --- |
| Python 3.12＋`uv` | 執行整套實驗程式、管理套件與可重現的環境 | 不分析影片內容 |
| FFmpeg／ffprobe | 讀取片長、尺寸、旋轉與影格時間；製作 proxy、偵測切鏡、抽原始影格及輸出 review cut | 不理解人物、物件或故事 |
| Temporal Risk Window scanner | 以本機低解析影格差異找出可能被約 1 FPS 粗取樣漏掉的短暫視覺變化，提出需要加密檢查的時間窗 | 不宣稱時間窗內一定有語意事件，也不產生剪點 |
| Gemini File API | 上傳並暫存可重用的影片或圖片，避免同一檔案在有效期內重複上傳 | 不執行內容判斷 |
| Gemini 3.6 Flash＋Interactions API | 看完整 proxy、建立 Clip Cards、提出選片與敘事候選；在指定的單張影格中找出目標 bbox | 影片時間只適合語意搜尋，不提供 frame-accurate 剪點；單張 bbox 也不是逐幀追蹤 |
| Pydantic Structured Output | 限制模型輸出欄位與型別，拒絕超界時間、非法 bbox 或不存在的 frame ID | Schema 合法不代表模型的內容判斷一定正確 |
| SHA-256＋不可變 frame ID | 確認素材、proxy、影格與模型結果的來源，並把 AI 選中的畫面映射回原片 | 不判斷畫面好不好 |
| Evidence Proposal／QueryLock | 先讓真人確認目標身分、動作條件與構圖需求，再把這份決定鎖定供後續步驟引用 | 不自動創造新目標，也不取代人工核准 |
| Gemini image Grounding | 在 FFmpeg 抽出的原始單張影格上，找出指定人物或物件的 0–1000 normalized bbox | 不跨影格追蹤，也不能把不可見目標的位置猜出來 |
| SAM 2.1（選配） | 以人工或 Gemini bbox 作為 seed，在同一個 shot 內產生 mask 並向前、向後追蹤 | 不理解剪輯 brief，也不應跨切鏡自行延續物件身分 |
| Identity checkpoint | 在固定預算內挑出追蹤起點／終點、遮擋後重現或幾何異常的 exact frames，再驗證是否仍為鎖定實例 | 不修改 SAM geometry，也不能用未執行的檢查冒充通過 |
| 本機 crop solver | 根據整段 tracking、required regions 與畫面邊界計算 9:16 安全裁切路徑 | 不自行決定哪個人物或物件最重要 |
| 本機 MusicMap analyzer | 將音訊解碼成 PCM，提出 beat、accent、energy、section 與 ending-hit 候選 | 不理解歌詞、音樂情緒或剪輯 brief；BPM、第一個 downbeat 與 meter 未經真人核准不可執行 |
| Gemini semantic music pairing（選配） | 聽取音樂的強弱、張力、留白與收尾感，再把既有 visual event ID 配對既有 music cue ID | 不重新偵測拍點、不輸出精確時間，也不能創造本機 MusicMap 沒有的 cue |
| VisualSyncMap＋CuePlan | 把畫面的 cut、reveal、action apex、ending pose 等事件，在明確 timing window 內對到已核准的音樂 cue；可把 Gemini 配對當排序加分 | 不會為了卡拍暗中截斷 setup／action／result，也不會直接改寫選片、trim、identity 或 geometry |
| Pillow | 把 bbox 或 mask 畫回原始影格，產生方便人工檢查的 debug 圖 | 不參與辨識或追蹤 |
| 本機 HTML／JavaScript review page | 播放事件、候選片段、debug 圖與裁切結果，供真人核准或退回 | 不會因頁面能正常開啟就宣告模型結果正確 |
| pytest | 驗證 schema contract、時間邊界、座標轉換、cache 與 geometry 規則 | 不取代對真實影片的人工觀看 |

簡單來說，Gemini 負責「理解內容與選對目標」，FFmpeg 負責「精確回到原始媒體時間」，SAM 負責「在同一鏡頭裡延續空間位置」，本機規則負責「檢查與裁切」，最後仍由真人決定結果能不能採用。

### 技術上的對應

最新方法採用「先鎖定證據，再驗證 geometry」：未指定 target 時先提出候選，使用者先審核 `EvidenceQueryProposalV2`，再明確核准成不可變的 `EvidenceQueryLockV2`。V2 把持續物件身分（Identity）、只在特定時刻成立的動作／狀態（Predicate）與構圖義務（Framing）分成三份 contract 與 hash；時間 refinement、單幀 bbox、SAM seed 與 layout 因此可各自重用正確層級的證據。自動直式構圖則在一份 planner response 內保留 Top-K 素材候選，只有實際嘗試的候選才由具名自動政策建立 QueryLock 並進入 exact-frame geometry preflight。完整說明見 [METHODOLOGY.md](METHODOLOGY.md)，毛片 coarse-to-fine 全量流程見 [FULL-VERSION-PLAN.md](FULL-VERSION-PLAN.md)。Gemini polygon 與 bbox seed 的舊 A/B 僅保留為唯讀歷史資料；目前支援路徑只使用 Gemini／人工 bbox → SAM。

## 這個專案要驗證什麼

JasCueVideoLab 是可重跑、可計價、可人工稽核的研究工具，目的是分開量測以下能力，而不是用一次模型回答宣稱「AI 已經會自動剪片」：

- Gemini 是否能完整觀看逐支 proxy，產生可重用的 Clip Cards 與 coarse semantic events。
- 有 brief 或無 brief 時，Gemini 是否能從整批 Clip Cards 選出合適 take、敘事順序、時間段，以及針對 16:9／9:16 應保留或可犧牲的內容。
- 從原片 exact frame 取得的 Gemini bbox，是否能正確指定語意實例並成為 SAM shot-local tracking seed。
- 本機 crop solver 是否能利用整段 track 做雙比例構圖，保存每次候選切換、失敗、fallback、成本與處理時間，供真人決定是否採用。
- 同一素材重跑時，選片、事件、時間與 geometry 是否穩定；模型錯誤、429、不可見與不確定狀態是否能 fail closed，而非靜默補值。

Gemini 負責語意選擇，FFmpeg 負責可回映的媒體時間，Gemini image Grounding 負責單張 seed，SAM 負責同一 shot 內的時序 geometry，本機規則負責驗證與裁切，真人負責最終內容和畫面品質。任何模型 confidence、schema 通過、bbox 或 SAM mask 都不是 production ground truth，也不得直接當成 JasCue 正式 SpatialTrack。

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
  → 建立 Proposal（Identity／Predicate／Framing 分層）
  → 人工核准成 QueryLock；自動流程只能引用具名 policy
  → predicate 存在時，在單一 shot 的 4／8 FPS DF catalog 做一次 frame-ID refinement
  → 本機把 DF ID 查回 PTS，再由 FFmpeg 抽原始 exact frame／hash
  → Gemini image bbox（多候選需人工指定）
  → SAM 2.1 bbox-only、shot-local mask propagation

只有快速 UI／短暫狀態不確定時：
  → 本機 Temporal Risk Window scanner 可先獨立於 Clip Card 找出視覺變化窗口
  → 已知事件內再建立 1–5 秒局部 4／8 FPS frame-ID contact sheet
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

# 有 predicate 時先做一次局部 frame-ID refinement；輸出只引用既有 DF ID
uv run jascue-video-lab refine-query-predicate \
  artifacts/full-v1-library/clips/ASSET_PREFIX EVENT_ID \
  --query-lock examples/evidence-query-lock-v2.json \
  --query-target-id subject.primary \
  --sampling-fps 8 --window-ms 4000 \
  --output-dir artifacts/query-refinement/EVENT_ID

# 只有被選中的事件才抽原始影格並選配 SAM；DECISION_JSON 由上一步 result.json 指向
uv run jascue-video-lab full-ground-event \
  artifacts/full-v1-library/clips/ASSET_PREFIX EVENT_ID \
  --query-lock examples/evidence-query-lock-v2.json \
  --query-target-id subject.primary \
  --predicate-decision DECISION_JSON \
  --sam-checkpoint artifacts/models/sam2.1_hiera_tiny.pt

# 明確要求某個短暫事件進入 4／8 FPS 局部 fallback
uv run jascue-video-lab full-clip VIDEO.mp4 \
  --dense-event EVENT_ID --dense-fps 8 --dense-window-ms 4000 \
  --output-dir artifacts/full-v1-clip

# 不依賴既有 Clip Card event，先在本機找可能值得加密檢查的視覺變化窗口
uv run jascue-video-lab scan-temporal-risk VIDEO.mp4 \
  --sampling-fps 4 \
  --output artifacts/temporal-risk.json

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

4／8 FPS dense DF contact sheet 現在是局部升級手段，不是預設 Trim Intent：只有快速手勢、短暫 UI、本機 risk window 或真人對 coarse 邊界有疑義時，才在小視窗內讓模型從既有 exact frame ID 選擇。`scan-temporal-risk` 只輸出 recall-only 視覺變化窗口，會排除已知硬切鏡且不把畫面差異冒充語意事件；目前仍需由後續流程或真人把該窗口配對到事件，尚未自動改寫 Clip Card。不得把整支毛片拆成大量圖片來取代影片理解。若 Gemini 只回傳 hold 的單側端點，系統不會推測另一端，而會捨棄不完整 hold interval 並把 contract normalization 寫入 uncertainties；若 exclusive out 位於片尾且沒有下一張 decoded frame，則保存明確的 end-of-stream time boundary，而不偽造 frame hash。

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
3. **部分完成**：QueryLock v2 已把 temporal（identity＋predicate＋catalog）、Grounding（identity＋exact frame）、SAM（identity＋seed／interval）與 framing lineage 分開；較早的 proxy、shot 與部分 dense cache 仍要補齊全鏈路 fingerprint。
4. **部分完成且已 fail closed**：每個新 SAM sample 可回映原始 decoded source PTS，並會以零 API 成本規劃 bounded identity checkpoints；exact-frame Gemini verifier 與有界 executor contract 已存在，錯誤、不可見與歧義會保存成明確狀態。Renderer 尚未自動解析 checkpoint frames 並執行 verifier，因此 tracked crop 目前記為 `required_pending`，不能再把未執行的 `None` 當成通過。遮擋後自動 re-identification 與完整 renderer 核准仍未完成。
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

#### Full Auto v2 executor＋selection planner v3

單一模型首選可能在語意上合理，卻在目標比例下無法安全構圖。Full Auto v2 executor 因此要求每個 `supported`／`partial` chapter 保存 2–4 個不同 evidence frame 的候選，而不是只留下 rank 1。selection planner v3 仍由 Gemini 依 brief 決定 source asset、Clip Card event、immutable RF frame、可見證據、品質風險、橫／直策略、簡短 framing intent，以及有序且互斥的 `required_entity_ids`、`preferred_entity_ids`、`sacrificable_entity_ids`。因此模型沒有失去「要剪哪裡、要保留哪個部位」的判斷。

v3 不再要求 Gemini 重抄 rank-1 asset/event/frame、target description 或 verbose resolved regions。本機會把模型選出的 entity IDs 對回一份 hash-bound `selected-clip-card-evidence.json`，確定性補出 target descriptions、相容欄位與 executable region contracts；projection 可由原始模型輸出和這份證據快照完整重現。相較 v2，送入模型的 Clip Card payload 約縮小 30%，response schema 字元數約縮小 44%，也移除了先前造成付費整批重試的 mirror-field 不一致來源。

目前自動 candidate routing 已接到 9:16 路徑：renderer 依候選順序，先核對 asset／event／frame lineage 與單一 shot 邊界；只有真的要跑 geometry 的候選，才由 `policy:full-auto-topk-lazy-geometry-querylock-v2:v1` 建立具真實 `auto_policy` provenance 的 QueryLock。接著從原始來源抽 exact frame，以 identity-only Gemini bbox 建立 SAM seed，完成 shot-local tracking 與實際 crop path，最後才執行本機 preflight。16:9 的 Top-K 也會保存在 schema 與 provenance 中，但目前仍採投影後的選定候選，尚未執行同等的 runtime geometry switching。

構圖需求使用領域中立的 region contract：

- `hard_core`：語意上必須完整保留的區域；來自 `required`。
- `soft_extent`：有助構圖但可以有限取捨的脈絡；來自 `preferred`，並有明確的最小可見比例。
- `overlay_keepout`：後續圖卡或版面不應遮住的區域；來自 `avoid_overlay`。
- `atomic=true`：局部裁切會破壞意義的單一區域，不論原角色為何都視為 hard core，必須 100% 保留。

同一候選可把多個 region 分別 Grounding，並在同一個 SAM session 內建立獨立 track。Crop solver 以 hard-core tracks 求每個 sample 的合法窗口，soft-extent tracks 只影響構圖中心與可見比例稽核；它們不會擠掉 hard core。Preflight 另外檢查來源 lineage、shot 範圍、Grounding／tracking gate、首尾與中段 coverage、hard-core containment、soft-extent floor、overlay keepout、crop speed、acceleration、jerk，以及 source／track／geometry SHA-256。

自動路徑使用版本化 `auto_bounded_clip_v1`：候選必須先用 `preserve_all` 解出 hard core；只有 soft extent 仍高於明列 floor 時，才可標記為 bounded clip。它不授權裁掉 hard core 或 atomic region。相對地，`controlled_clip` 仍必須來自 content-addressed 的真人 policy sidecar；一旦存在此 binding，renderer 會停用自動換候選，完全依真人核准的候選與 edge priority 執行。

失敗會保存 typed failure code 與 recovery action，例如 shot crossing、coverage 不足、hard core 被裁、soft extent 低於門檻、keepout 違規或 crop motion 過快。現行 executor 會實際嘗試下一個 Top-K 候選，並把預先規劃的 safe-fit 候選延後到 tracked candidates 之後；其他 recovery action 目前是可稽核建議，尚未自動執行。所有候選都失敗時，輸出只是一份 `policy_blocked_preview_fit` 全內容補邊預覽並固定要求人工 review，不會偷偷改用未驗證的中心裁切。

這個設計也控制成本：Top-K 是同一次 planner 回應中的 2–4 個備選，不是把每支影片重送 K 次。Clip Card 可跨 brief 重用；planner 預設只允許一次 text-only Structured Output request，`--repair-attempts` 預設為 `0`。bbox／SAM geometry 採 lazy evaluation，只有已選 chapter 的候選才依序執行，遇到第一個通過 preflight 的候選就停止。Predicate refinement 也是明確指令才執行的一次局部 image call，不會在 Grounding 內暗中 repair；同一 identity＋exact frame 可在 framing 改變後重用 bbox。每次 model request、raw response、usage、prompt/schema/model fingerprint，以及每個候選的嘗試與 geometry fingerprint 都會保存；重跑時每次 response 另存 immutable attempt，canonical 檔不再覆蓋歷史成本。計價會把 `total_cached_tokens` 依 cached-input 牌價和一般 input 分開計算；若某份 response 沒有 usage，會列為未計價 request 並把總額標成不完整下限，不會當成免費。失敗候選仍可能增加 Grounding 費用與 SAM 時間，因此成本報告須以實際 raw usage 與本機 tracker timing 為準，不能只用候選數乘固定牌價。

429／quota failure 不屬於候選內容問題。為了避免隱藏成本，SDK 明確設成每個 Gemini operation 只嘗試一次；若上游回傳真正的 HTTP 429、`RESOURCE_EXHAUSTED` 或 spending-cap error，geometry executor 會立即寫出 `geometry-model-circuit-breaker.json` 並中止整次 render，不再換候選、不再繼續輸出看似完成但沒有 Gemini geometry 證據的 fallback 成片。一般的 target 不可見、tracking coverage 或構圖不可行才會繼續嘗試下一個候選。

Full Auto v2 目前仍有清楚限制：已有風險導向、固定預算的 identity checkpoint 規劃器、exact-frame Gemini verifier 與 executor artifact，但尚未把 frame extraction／verifier execution 自動接入每個 candidate preflight；因此 tracked crop 會保持 `required_pending` 並要求人工處理，而不會自動通過。遮擋後自動 re-identification 與自動圖卡避讓也尚未完成；`overlay_keepout` 在有字卡但沒有 layout solver 時會 fail closed。獨立的成片 QA 可以提出語意 review，但不會替 preflight 補造證據或覆蓋 geometry gate。Safe-fit 只是方便人工觀看的預覽，不是核准構圖；模型 rank、confidence、SAM mask 與 schema validation 也都不是 human ground truth。

每個 tracked 9:16 segment 現在保存 renderer 真正使用的 crop keyframes、required-region union、逐時刻合法 crop interval、containment、可見寬度比例、首尾／中段 tracking coverage、crop speed 與 acceleration。裁切器不再先平滑 target 中心後直接裁切，而是先由每一個 required bbox 算出合法範圍，再把平滑路徑投影回該範圍；這可避免平滑延遲把快速移動主體推出畫面。`primary_center` 只會放寬 target 外圍的 8% safety margin，不暗中授權裁掉 target。

Reframe geometry 使用 FFmpeg 自動旋轉後的 display dimensions 做 aspect-preserving cover，不再假設來源一定是 16:9。4:3、直式、超寬與相同比例素材都共用二維 x／y crop solver；manifest 保存來源／縮放／輸出座標空間、兩軸合法區間與實際 crop keyframes。非方形像素來源會先依 FFmpeg frame SAR 還原顯示比例；在 tracking 尚未建立同一顯示座標系前，只能 fail closed 到已正規化 SAR 的靜態 reframe 並標記人工複核。track seed 尺寸、analysis aspect 或多 track lineage 與來源不一致時同樣不能把 normalized 座標硬套進 renderer。

一個構圖可用 `vertical_regions` 分開表達多個 required、preferred 或 avoid-overlay 區域；kind 只使用泛用的 `subject`、`text_region`、`ui_region`、`graphic`、`other`。多個 required 區域會各自取得 exact-frame Gemini bbox，再共用一個 SAM 2.1 video session，逐 sample 合併 union，避免把兩個人物或「人物＋螢幕」寫成一個模糊 target。文字也沒有品牌特例：brief 應指定「必須完整可讀的語意核心」為 required text region，裝飾或整塊容器則可列 preferred。

預設 `vertical_overflow_policy=preserve_all`；required union 太寬、任一 required track 遺失、首尾 coverage 不足或逐 sample containment 失敗時都 fail closed。Gemini plan 的 schema 也只能輸出 `preserve_all`；模型若認為有限裁切值得考慮，只能留下 `vertical_overflow_proposal`，proposal 不具執行權限。

人工若明確接受有限裁切，才可透過 `scripts/apply_reframe_policy.py` 選 `controlled_clip`，並以 `preserve_start`、`preserve_end` 或 `balanced` 指定優先側。腳本不會重跑選片，而會把原始 catalog、preserve-all brief、feature plan、plan binding、選定 frame IDs 與人工 policy 寫入 content-addressed sidecar；修改其中任一輸入，renderer 都會 fail closed。產生的 revised bundle 必須搭配 `--reuse-feature-plan` 使用，manifest 仍保存可見比例與 review requirement，不能把受控裁切冒充完整保留。

```bash
uv run python scripts/apply_reframe_policy.py \
  SOURCE_BRIEF.json HUMAN_POLICY.json REVIEWED_OUTPUT_DIR \
  --catalog CATALOG.json \
  --feature-plan SOURCE_OUTPUT/gemini-plan/feature_edit_plan.json \
  --reviewer "human-reviewer" \
  --review-note "reviewed required-region tradeoffs"

uv run jascue-video-lab feature-cut \
  CATALOG.json REVIEWED_OUTPUT_DIR/brief.json \
  --sam-checkpoint SAM_CHECKPOINT.pt \
  --output-dir REVIEWED_OUTPUT_DIR \
  --reuse-feature-plan
```

`scripts/build_vertical_crop_audit.py` 可把候選、fallback、風險碼、Grounding debug 與實際 crop 軌跡合成審核頁。

若使用者明示不要背景補邊，而 SAM propagation 又不完整，renderer 可把已驗證的 Gemini seed union 當成靜態 anchor，依同一套 edge policy 產生 `seed_anchor_crop`；它比盲目裁來源正中央更接近指定主體，但不代表已驗證 seed 前後的移動，因此固定標記 `motion_outside_seed_unverified` 並要求人工複核。若 required union 本來就比 9:16 寬，仍應優先換 take、調整 required／preferred、拆鏡或選擇 contain／split／PiP；靜態 anchor 不能把幾何上放不下的內容變成完整可見。

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

SAM 只提供幾何，不自行決定剪輯美學。16:9 的 `none`／`subtle`／`detail` 由 feature plan 表示 editorial intent，實際倍率不得超過 mask 安全值。9:16 的 `strict` 要求完整保留 required regions；`primary_center` 只表示可犧牲未列為 required 的次要 context。真正允許裁掉 required union 時，必須另以 `controlled_clip` 明示並接受語意複核。使用者 brief 的規格文字與模型觀察到的畫面證據分開保存，沒有 ASR 或 transcript。

```bash
uv run jascue-video-lab feature-cut \
  artifacts/my-rushes-run/catalog.json \
  BRIEF.json \
  --sam-checkpoint artifacts/models/sam2.1_hiera_tiny.pt \
  --sam-analysis-fps 2 \
  --output-dir artifacts/my-feature-cut
```

預設 `--aspect both` 會輸出兩種比例。只需要 Shorts 時可傳
`--aspect 9x16`，只需要橫式時則傳 `--aspect 16x9`；未要求的比例不會執行
Grounding、SAM geometry、segment render 或 concat，manifest 會明確標成
`not_requested`，避免把不存在的輸出或未發生的模型成本記成成功。

若同一個 output directory 已保存 feature plan，renderer 不會再因檔案存在就自動假設它仍符合目前 prompt／brief。要做只比較裁切器的 controlled A/B，必須明示 `--reuse-feature-plan`；程式會保存舊 plan、目前 catalog／brief／prompt 的 hash，並只重算 geometry 與成片。想重新選片時則使用新的 output directory，不加此旗標。

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

`scripts/plan_clip_card_feature_cut.py` 將這個方法延伸到完整 feature cut：模型可閱讀整個已驗證 Clip Card library，但只能選 catalog 中既有的 asset／event／entity／RF frame ID；本機會再次驗證影格確實屬於該素材、位在事件區間，且每個 brief-specific entity priority 都能回溯到 event，再以 hash-bound Clip Card evidence 投影出 `feature-cut` 可使用的 target 與 region contract。選片階段不產生 bbox 或剪點，只有真正入選、需要動態構圖的區間才執行 exact-frame Grounding 與 SAM。新版保留每章 2–4 個候選，9:16 renderer 會先試可驗證的 tracked candidates，再考慮 planner 明列的 safe-fit；所有候選均失敗時只輸出待審 preview，不會把中心裁切冒充成成功追蹤。

Clip Card plan 轉成 renderer plan 時會另寫不可變的 external-projection sidecar，保存來源 catalog、brief、模型 request／raw response、projection contract 與輸出 plan 的 hash。candidate override 也必須接續並驗證這條 provenance；任一上游內容改變就 fail closed。早於此 contract 的舊 artifact 不可手動複製 plan 冒充可重用結果，必須從仍保存的原始 artifact 重新投影。

`scripts/plan_clip_card_open_edit.py` 是沒有內容 brief 的對照實驗：只給 60–90 秒與雙比例等操作限制，讓 Gemini 從完整 Clip Card library 自行推論主題、時間軸位置與每格 2–4 個候選。新版 evidence payload 也保留 Entity kind、required／optional／avoid-overlay 關係，讓模型可產生泛用 `vertical_regions`，而不是把多個獨立主體合寫成一個 bbox target。局部 Trim Intent 可能為保留完整動作而使成片超過模型原先配置的秒數，因此 `scripts/reconcile_open_edit_budget.py` 另讀實際 segment durations，只以 keep／drop／reorder 完整片段把全片拉回 duration contract；它不會在動作中間靜默截短。

Planner 的 JSON Schema 無法完整表達所有跨欄位 invariant；模型若同時填入互斥但可保守消解的欄位，本機只允許不增加執行權限的 canonicalization，例如明示 `original` 時清除 zoom／focus，或將 required／atomic region 收緊成完整可見。原始付費 response 與原始 request 永遠保持不變，逐 JSON path 的 before／after／rule、canonical output 與兩邊 hash 另存；無法安全消解的矛盾仍 fail closed。`--reuse-raw-output` 只會重投影完全配對的 request／interaction／raw-output，fresh paid run 也拒絕覆寫既有 artifact namespace。

當 9:16 audit 證明 hard-core union 不可容納或 tracking coverage 不足時，自動路徑會先換同一 chapter 的下一個 evidence-bound candidate，而不是立刻套背景補邊。`scripts/apply_open_edit_candidate_overrides.py` 仍可接受人工審查過的 `feature_id + aspect + candidate_id + reason` patch，保留原始 OpenEditPlan 與兩邊 hash，再重新投影 brief／feature plan／trim plan；候選不存在或同一 aspect 重複覆寫會 fail closed。人工 override 與自動 candidate routing 是不同權限層，不能互相冒充。

成片 Gemini QA 採成本分級，不是每次 render 都重看全部。所有片段先跑零 API 成本的本機 geometry／coverage／media gate；只有文字／UI、多 required targets、controlled clip、tracking risk 或 fallback 段落需要語意複核。`scripts/verify_feature_cut.py` 可把完成版 9:16 壓成 720×1280 proxy，以一次 `gemini-3.6-flash`、`thinking_level=low` Structured Output call 檢查主體身分、重要文字、語意是否符合與重複／突兀問題；它不回時間戳、不驗證 frame-accurate geometry，也不會自行改剪。相同 render／manifest／prompt／schema／model 會重用結果，最終仍由真人核准。若 schema 驗證失敗或重試，每一次 request、raw response、錯誤、timing 與 pricing 都保存於不可覆寫的 attempt 目錄，總成本會聚合所有實際有 usage 的請求。

Gemini 的成片 `pass` 不可覆蓋本機幾何證據。QA validator 會把 required-region coverage、完整 containment、controlled clip、fallback 與 source-edge 診斷一起納入本機最終狀態；只要其中一項需要複核，即使模型認為主體「看得見」，validated status 仍固定為 `review`。feature-cut 另把本次新增或改變的 raw interactions 記在 `pricing.incremental.json`，與包含歷史快取的 `pricing.json` 分開，避免把舊請求重複算成本輪花費。

主要影片／圖片辨識請求另使用 Interactions API `system_instruction` 建立 evidence-only 邊界：本次媒體與明確 metadata 是唯一證據，禁止以模型記憶、常見名稱、相似外觀或「最可能答案」補完品牌、型號、數字與 UI 文字。Full Clip Card prompt 也要求任一關鍵字元不清楚時改用泛稱並保存 uncertainty。控制 A/B 曾觀察到舊 prompt 以先驗補完一個相似但錯誤的型號；改用 domain-neutral 規則後該欄位在重跑中恢復正確，模型卻又把另一處模糊小字補成畫面不存在的規格。這證明 prompt guardrail 不是 ground truth：單一正確 claim 不代表整張 Clip Card 都正確，衝突與重要文字仍需 exact-frame 驗證及人工核准。

`scripts/verify_clip_card_text.py` 實作不覆蓋原始 Clip Card 的文字驗證：從原片抽多張 exact frames、保存 PTS／hash、裁出文字證據，以 `resolution=high` 分別做 blind transcription，再以明列 `other`／`unreadable` 的候選式請求交叉檢查。方法不一致時輸出 `needs_human_review`；只有人工核准後才能另外產生 reviewed Clip Card。

## 音樂卡點 MVP

音樂卡點採獨立 evidence chain，不讓音樂分析器或 renderer 直接改寫已核准的選片、Identity、Trim 或 geometry：

```text
本機音樂檔
  → FFmpeg 解碼單聲道 PCM
  → MusicMap Proposal：beat／accent／energy／section／ending-hit 候選
  → 真人核准 BPM、第一個 downbeat、meter
  → immutable MusicMap Lock

既有 feature-cut render manifest
  → VisualSyncMap：目前 cut／chapter start／ending pose
  → 可另外加入經證據確認的 reveal／action apex／UI change
  → 選配：Gemini 聽音樂並把既有 visual ID 配對既有 cue ID
  → 全局、順序保持的 CuePlan scheduler
  → CuePlan Proposal＋HTML 人工審核
  → 真人核准成 CuePlan Lock
```

零成本 baseline 全部在本機執行，不呼叫 Gemini。分析器只提出聲學候選，不把 `section_001` 冒充為 verse、chorus 或 drop；human review 之前，beat grid 不具執行權限。`narrative`、`balanced`、`montage` 三種 preset 只改變 section／downbeat／accent／一般 beat 的排序權重，不改變素材語意。

若要減少規則式卡點的機械感，可選擇再執行一次 `gemini-3.6-flash` 音樂語意配對。Gemini 會同時取得音樂、已核准的 MusicMap cue IDs，以及 Clip Card／render manifest 衍生的視覺事件語意；它只能回答「哪個 visual event 適合哪些既有 cue IDs」，不能自己發明秒數。這是每支音樂一次、可由 File API 重用的選配請求，不會對每個鏡頭重送音樂。最終 sample-accurate 位置、合法 timing window、全局順序與 hard gate 仍由本機決定。

```bash
# 1. 本機分析音樂；輸出 proposal，不會自動核准
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab analyze-music MUSIC.wav \
  --output-dir artifacts/music-demo

# 2. 真人確認；可覆寫 BPM、第一個 downbeat 與拍號
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab review-music-map \
  artifacts/music-demo/music-map.proposal.json \
  --reviewer "human-editor" \
  --decision approved \
  --bpm 120 \
  --first-downbeat-ms 240 \
  --meter 4 \
  --output-dir artifacts/music-demo/reviewed

# 3. 從既有成片 manifest 建立視覺事件。預設 flex=0，因此只做唯讀稽核。
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab build-visual-sync-map \
  FEATURE_OUTPUT/render-manifest.json \
  --aspect 9:16 \
  --output artifacts/music-demo/visual-sync-map.json

# 若操作者明確允許 boundary 前後各移動 250 ms，才可另建有 window 的 proposal：
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab build-visual-sync-map \
  FEATURE_OUTPUT/render-manifest.json \
  --aspect 9:16 \
  --default-flex-ms 250 \
  --output artifacts/music-demo/visual-sync-map.flex-250.json

# 4. 產生全局 CuePlan 與可播放的 HTML review；尚未修改影片
# 選配：先讓 Gemini 做一次音樂—畫面語意配對
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab plan-semantic-music \
  MUSIC.wav \
  artifacts/music-demo/reviewed/music-map.lock.json \
  artifacts/music-demo/visual-sync-map.flex-250.json \
  --output-dir artifacts/music-demo/semantic-pairing

UV_CACHE_DIR=.uv-cache uv run jascue-video-lab plan-music-cues \
  artifacts/music-demo/reviewed/music-map.lock.json \
  artifacts/music-demo/visual-sync-map.flex-250.json \
  --preset balanced \
  --semantic-pairing artifacts/music-demo/semantic-pairing/semantic-music-pairing.proposal.json \
  --music MUSIC.wav \
  --video FEATURE_OUTPUT/renders/feature-cut-9x16-clean.mp4 \
  --output-dir artifacts/music-demo/cue-plan

# 5. 真人核准 hash-bound CuePlan
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab review-cue-plan \
  artifacts/music-demo/cue-plan/cue-plan.proposal.json \
  --reviewer "human-editor" \
  --decision approved \
  --output-dir artifacts/music-demo/cue-plan/reviewed
```

目前 MVP **只完成卡點分析、排程、稽核與 lock，尚未自動重剪或混音**。這是刻意的 fail-closed 邊界：render manifest 只有既有 segment duration，不能證明把 cut 移動 250 ms 仍保留完整 setup／action／result、合法 source handle、同一 shot、可用構圖與片尾 hold。下一階段必須讓經核准的 Trim Intent 提供 action-safe timing window，通過 geometry preflight 後才可將 CuePlan 套入新的 RenderPlan；不會直接用 `setpts` 變速或裁掉動作來製造「有卡拍」的假象。

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

只使用官方 `google-genai` SDK；預設模型是穩定版 `gemini-3.6-flash`，需要可重現歷史 A/B 時才以 `JASCUE_GEMINI_MODEL` 明確覆寫。3.6 請求不送出已淘汰的 `temperature`、`top_p` 或 `top_k`；純 geometry 目前維持 `low`，exact-frame semantic identity checkpoint 使用 `medium`，較複雜的少數規劃實驗才使用 `high`。不同 task profile 必須進入 request 與 cache fingerprint，且不能假設較高 thinking 一定改善 bbox。模型 ID 會進入 request、cache identity、provenance 與逐模型計價；切換模型不會誤用舊 response cache，但仍可重用相同 File API 上傳。程式不依賴已淘汰的 `google-generativeai`，也沒有舊 Gemini 1.5／2.0 model ID。

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
4. 使用者選擇候選或自行輸入精確 target 後，先檢查完整 Identity／Predicate／Framing Proposal 與 hash，再明確核准 QueryLock，才允許產生 target-locked `MM:SS` coarse candidates。
5. Identity-only QueryLock 可選一個時刻，由 FFmpeg 保存原始 frame PTS，再執行單幀 Grounding。若 QueryLock 含 predicate，Web 只顯示 coarse candidates 並鎖住 Grounding；必須先匯出、執行 `refine-query-predicate` 取得正式 DF frame-ID evidence，不能把 `MM:SS` 冒充為 predicate 已驗證。
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
