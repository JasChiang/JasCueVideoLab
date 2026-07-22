# JasCueVideoLab 全量版設計建議

這份設計把現有實驗升級為可對完整毛片庫執行、可人工審核、可重現的挑帶與構圖 pipeline。它仍是獨立實驗，不是 JasCue 正式產品，也不將模型時間或 bbox 當成 production SpatialTrack。

## 結論先講

不應把每 2 秒一張直接改成全庫 4–8 FPS。這會同時擴大本機預處理、上傳、token 與審核負擔，也不會自動解決切鏡、相似物件與語意選錯。

推薦使用自適應的 coarse-to-fine 架構：

```text
全庫媒體登錄／shot detection
  → 每支完整低解析 proxy 逐片建立 Gemini Clip Card
  → 每個 shot 的一張縮小稽核幀（不作 Grounding）
  → Gemini 根據 brief 從 Clip Cards 與代表幀找候選 shot
  → 人工鎖定 EvidenceQueryLock（target、排除條件、可選 predicate）
  → 入圍影片由 Gemini 直接觀看並提出 coarse MM:SS representative select
  → FFmpeg 對回原片 boundary PTS／hash
  → 快速或有疑義的局部邊界才加密為 4–8 FPS immutable frame IDs
  → Gemini image Grounding 只提出 bbox；多候選需人工選擇
  → SAM 2.1 以 bbox seed，在允許區間 ∩ seed shot 內傳播 mask
  → 人工審核選帶、in/out、bbox 與 9:16 構圖
  → 輸出 review cut 與完整 evidence manifest
```

2 秒幀只是「找哪一段值得繼續看」的便宜索引。快速 UI、0.2–0.5 秒短暫狀態、高速手勢與對焦轉換都必須進入密集第二層。

## 第一階段：建議現在做的 Full v1

### 1. Immutable media registry

- 每支影片保存 SHA-256、ffprobe metadata、rotation、duration、frame rate 與 time base。
- 公開報告只使用 `asset_id`，本機原始路徑放在不進 Git 的 private manifest。
- 所有後續產物使用 source hash、proxy hash、prompt hash、schema hash、model ID 與 frame hash 組成 cache key；cache hit 還要反查實際保存的 raw request，不能只信 cache key 檔。

### 2. Per-clip full-video understanding

全量版應該讓 Gemini 逐支看完整毛片，但上傳的是本機產生、可重建的低解析 analysis proxy，不是把整批 4K 原檔直接上傳。每支影片產生一份 Structured `ClipCard`：

- 影片全體摘要、可觀察動作與動作是否完整。
- 人物、動物、物件、產品、裝置、螢幕、UI、文件與文字區域等 entities。
- 可能的剪輯用途：establishing、hero、detail、demo、reaction、transition 或 ending。
- 畫面品質：對焦、運鏡、動態模糊、遮擋、曝光、動作開頭／結尾是否可用。
- 直式構圖可行性與必留實例，但不輸出最終 crop 座標。
- 不確定、快速 UI 可能漏看、必須進入密集層的理由。
- `first_1_5s_impact`、`narrative_priority` 與 `claim_source`，用來區分「代表畫面」和「適合吸引觀眾的開場」，並避免把模型觀察誤當產品規格來源。
- `repetition_cluster`／`take_group` 候選，只負責召回相似拍攝，不直接替使用者淘汰 take。

`ClipCard` 與 coarse Event Map 不要求 Gemini 產生毫秒。模型時間固定使用 `MM:SS` 字串作為 coarse 語意 anchor；本機必須驗證格式、事件順序、半開區間與 ffprobe 片長，再衍生毫秒。若模型時間非法，保存錯誤並停止，不得靜默 clamp。入選片段的 coarse 邊界由 FFmpeg 解析到原始來源 PTS；只有快速或有疑義的局部狀態才要求 dense frame-ID refinement。

Full v1 應分開保存兩層 schema：

```text
Gemini semantic schema
  event start/end/keyframe = MM:SS、shot/frame references、語意與不確定性

Local derived schema
  validated milliseconds、source PTS、frame hash、boundary source
```

完整事件邊界仍存在，但 Gemini 的秒級區間只是搜尋範圍，不是 frame-accurate cut point。

逐片完整理解能補足靜態索引無法回答的時序問題，例如「人物是否完整拿起產品並轉向鏡頭」。但 Gemini File API 的影片視覺處理約為 1 FPS，所以這一層仍不能單獨證明 0.2–0.5 秒的 UI 狀態存在。

每支 Clip Card 以 source SHA-256、proxy SHA-256、model、prompt 與 schema version 快取。在資產或 prompt 未變的情況下，重跑剪輯 brief 不需要再花一次完整影片分析費用。

### 3. Shot-first visual catalog

- 使用 FFmpeg `scdet` 取得 decoded-frame PTS，必要時併用 `blackdetect`。
- 每個 shot 預設只保存一張縮小的中間 JPEG 供人工稽核；不保存三張 4K PNG，也不把這些稽核圖當成 Grounding evidence。
- 真正選中事件後，才回原始影片抽 1–3 張 exact frame，保存 PTS 與 hash 並做 Grounding。
- 切鏡邊界是 tracker 的強制中止點；新 shot 必須建新 seed。

### 3.5 已完成單事件垂直切片：trimming；待續：長毛片重拍與相似 take

這一層應在 Clip Card／Content Map 完成後另外執行，不得在媒體登錄時自動刪除素材：

- 將 10 分鐘等長毛片拆成可審核的 take／shot 區段，保存 coarse 建議 in/out 與 exact source PTS 的分工。
- 分開標記 `recommended_select`、`technical_reject`、`incomplete_action`、`possible_retake`、`intentional_hold`、`title_safe_hold`、`needs_human_review`。
- 靜止尾段不等於廢尾；模型必須考慮它可能是刻意留白、字卡空間、旁白 hold 或乾淨 plate。
- 同景多次拍攝先建立 `take_group`／`variant_group`，比較動作完整度、對焦、遮擋、運鏡、表演與留白，不直接判定檔案重複。
- 位元完全相同可用 SHA-256 判斷；視覺近重複可用 perceptual hash／embedding 做候選召回；最終「哪個 take 較適合 brief」再交給 AI 與人工審核。
- 所有 reject 都是可逆標記，不移除原檔；輸出 selects reel 前必須能查看相鄰 handles。

目前已實作入選事件的 Trim Intent 垂直切片：Gemini 直接觀看完整 proxy，在 `Clip Card event ∩ FFmpeg shot` 內提出 coarse `MM:SS` 代表性 select；本機解析原始 boundary PTS、半開區間與 handles，並輸出 preview。2／4／8 FPS DF IDs 保留作局部疑義的升級路徑。Proposal 永遠需要真人核准；feature renderer 只接受帶有 human review record 的 approved decision，或以明確 flag 輸出仍標示未核准的 review cut。尚未完成的是 10 分鐘等長毛片的自動 take segmentation、跨檔 take/variant grouping、近重複召回與全庫比較。

### 4. Brief-driven evidence retrieval

使用者先提供片長、目標比例、章節與想表達的功能。Gemini Structured Output 針對每個需求回傳：

- `supported`、`partial`、`not_found`，不得靜默補齊。
- 候選 shot ID 與 coarse frame ID。
- 直接可觀察的證據、風險、相似物件與建議主體。
- 16:9 與 9:16 必留、可犧牲與應避免覆蓋的 entity。

文案事實來源必須是使用者 brief；模型只能回報影像證據，不能自行發明產品規格。

### 4.5 Full Auto v2 executor＋selection planner v3

每個 `supported`／`partial` chapter 現在保留 2–4 個 evidence frame 不重複的候選，而不是只保存模型首選。候選包含 source asset、event、RF frame、可見證據、品質風險，以及 16:9／9:16 各自可執行的 strategy。Gemini 仍針對 brief 決定有序且互斥的 required／preferred／sacrificable entity IDs 與 framing intent；v3 不再讓模型重複輸出 target descriptions、rank-1 mirrors 或 verbose regions，而由 hash-bound Clip Card evidence 本機補出。Planner 不產生 bbox、mask、crop 座標或精確 cut point。

9:16 renderer 會以 immutable plan 順序逐候選執行 exact-frame bbox、shot-local SAM 與完整 crop-path preflight。Region contract 採 `hard_core`、`soft_extent`、`overlay_keepout` 與 `atomic`；自動政策不得裁掉 hard core／atomic，只能在 soft extent 高於明列 floor 時接受 `auto_bounded_clip_v1`。所有候選失敗時只輸出 `policy_blocked_preview_fit` 並要求人工 review，不會使用未驗證 center crop。`controlled_clip` 仍只接受 hash-bound 的真人 policy sidecar。

目前會實際執行的 recovery 是嘗試下一候選與延後 safe-fit；其他 typed recovery action 只保存為診斷建議。16:9 Top-K 已保存在 schema／provenance，runtime geometry switching 尚未接入。週期性語意 identity checkpoint、遮擋後 re-identification 與 overlay layout solver 也仍未完成，不能因 geometry preflight 通過就宣稱成片已自動語意驗收。

即使第二個模型只讀 Clip Cards，也必須另設 claim validator：逐條比對輸出旁白中的型號、畫素、倍率與功能名稱是否能從 brief deterministic 對回。模型在 uncertainties 中指出素材型號衝突，不代表衝突本身一定正確，也不代表它不會同時寫出肯定旁白；Structured Output 通過同樣不代表 OCR 或數值換算正確。任何疑似錯型號、可見浮水印或不一致標牌都應 fail closed，進入 `needs_human_review`，再由 orientation-corrected 原始影格確認。不得只憑 Clip Card OCR 自動淘汰素材，也不得由 narrative planner 自行決定採用。

### 5. Adaptive dense refinement

對每個候選區間建立第二層影格 ID：

- 一般操作：候選中心前後 3–5 秒，4 FPS。
- 快速 UI／短暫動作：8 FPS，或依 optical difference 觸發更密局部取樣。
- 長時間靜態產品展示：2 FPS 即可，但保留 shot 兩端。
- 產出多張 4×4 或 5×5 contact sheet，每格烙印 immutable dense frame ID。
- Gemini 只能回傳已存在的 ID；毫秒與 PTS 全部由本機 catalog 映射。

密集層應在以下任一情況自動觸發：`partial`、`not_found`、多個相似實例、快速 UI、低信心、推薦幀 Grounding 不可見，或人工點選「重找」。

### 6. Exact-frame Grounding and tracking

- 對 dense frame ID 對應的原始影格抽幀，保存 `frame_pts`、`frame_time_ms`、dimensions 與 SHA-256。
- Gemini image Grounding 只是 `semantic_seed_box`；不可見必須是 `visible=false` 與空 candidates。
- QueryLock 建立前的 dense selection 不得因 target ID 相同而重用；只有保存並完全符合 lock hash 的 dense artifact 才可提供 seed frame。
- `match_status`（target 身分）與 `predicate_status`（可選事件條件）分開保存；多候選不以 confidence 自動決勝。
- Gemini polygon seed 不進主路徑。SAM 2.1 只接收人工核准的 bbox，將其精煉成 mask，並只在 `允許區間 ∩ seed shot` 向前／向後傳播。
- 分開保存 `semantic_seed_box`、`sam_prompt_box`、`refined_mask` 與 `derived_tracking_box`。
- 每個 sample 保存 decoded source PTS、來源 time base 與 PTS 衍生時間；constant-rate debug MP4 只供播放，不是 edit timeline。
- 追蹤狀態使用 `tracked`、`reacquired`、`occluded`、`low_confidence`、`drift_suspected`、`lost`，不使用單一 success flag。
- 切鏡、完全遮擋、mask 面積／中心異常或身份疑似改變時強制重新 Grounding。

SAM 3 現階段不是 Full v1 的前置條件。Gemini 已負責複雜語意選物，SAM 2.1 已能承擔主要幾何傳播。只有在需要文字概念直接多物件追蹤、遮擋後重新識別，且有合適 NVIDIA GPU 時，才建議另開 SAM 3 A/B。

### 7. Human review app

全量版不應直接自動輸出正式成片。審核界面至少需要：

- 左側原片播放器，右側 shot／coarse／dense 候選庫。
- 每個 brief item 的「接受、拒絕、重找、備選」狀態。
- 可編輯 in/out、選擇 bbox 候選、手動修正框。
- 同時預覽 16:9 與 9:16，显示 strict 保留或 primary-center 犧牲的理由。
- 隱藏或開啟字卡；系統不默認燒錄文案。
- 審核完才輸出 review MP4 與 manifest。

## 資料、成本與隱私

- 原始 API response、schema validation、錯誤、不可見與不確定都必須保存，不靜默補值。
- Gemini Files 可在有效期內以 SHA-256 快取重用；重用前必須查詢遠端狀態，不能只信本機 URI。
- 獨立重跑預設 `store=false`，避免 previous interaction 把上次答案污染穩定度實驗。
- 執行前預估上限；執行後依 raw usage 分開記錄 video、image、text input，output，模型 latency 與本機 CPU 時間。
- 公開匯出不得包含使用者名稱、絕對路徑或攝影機原始檔名。

## Full v1 驗收標準

1. 完整影片庫不上傳原始 4K，只上傳可重建的 analysis media。
2. 每個最終 select 都能回溯 brief item → shot ID → dense frame ID → source PTS → frame hash。
3. 快速 UI fixture 的 0.2–0.5 秒狀態可觸發密集層，且不用 Gemini 毫秒作為對應依據。
4. 所有 tracker 不跨 shot；drift／lost 不得被 accepted flag 隱藏。
5. 9:16 每段都有 target、實際 crop path 與人工審核結果，不能以模糊背景掩蓋失敗構圖。
6. 同一輸入可獨立重跑三次，自動比較候選 shot/frame agreement、label 相似度、bbox IoU／center distance、schema 與人工標註。
7. 用戶可在本機 Web App 完成選帶、框修正、in/out 與雙比例預覽，不需修改 JSON。
8. 每次執行都有估算成本、實際 usage、未知計費項目與分階段計時。

## 建議開發順序

1. **Full v1a**：逐片 Clip Cards、shot-first catalog、dense contact sheets、brief evidence contract、cache/privacy manifest。
2. **Full v1b**：本機 review app 整合 coarse/dense 挑帶、Grounding 修正與雙比例預覽。
3. **Full v1c**：SAM 週期語意 revalidation、遮擋／drift recovery 與三次重跑報告。
4. **後續選配**：EdgeTAM/Core ML、NLE export、SAM 3 A/B。這些不應阻擋 Full v1 的驗證。

## 目前實作狀態（2026-07-22）

Repository 現已有逐片完整 proxy → Structured Clip Card、模型只回 `MM:SS`、本機衍生毫秒、FFmpeg shot PTS、每 shot 一張縮小稽核 JPEG、選定事件後 exact-frame Grounding、SAM 2.1 propagation，以及 4／8 FPS 局部 dense frame-ID fallback。批次 `full-library` 預設只建立 Clip Cards，不自動對全部素材跑 bbox、SAM 或密集抽格；`full-selected` 可從既有 feature plan 反查實際入選的 source clips，避免為全庫重複付費。公開 hash 索引與含路徑／檔名的 private manifest 分開保存。成本也已拆成本次新增請求與 artifact lifetime 歷史累計。

2026-07-21 已將 geometry 主路徑收斂成 domain-neutral QueryLock → exact-frame Gemini bbox → reviewed candidate → SAM 2.1。SAM 在 predictor 初始化前即只抽取 `允許區間 ∩ seed shot` 的影格，並保存每張影格的 decoded source PTS；不再先跨鏡傳播後才標記風險。exact-frame Grounding 與 SAM seed 也改用包含 target、frame、prompt/schema、checkpoint、shot bounds 與處理參數的 variant fingerprint。舊 Gemini polygon A/B 只保留歷史報告，執行入口會拒絕使用 polygon seed。

已新增直接看影片的入選事件 Trim Intent：模型回 coarse `MM:SS`，本機解析 source PTS，並保留 raw failure、usage、成本與 prompt/schema fingerprint；不完整 hold 不補猜，EOS 不偽造 decoded frame。成功 proposal 可產生 preview，人工核准後才會以 PTS bounds 取代 feature cut 的固定 duration 粗剪；另有顯式未核准 review-render 模式。

Full Auto v2 executor 與 selection planner v3 已完成 Top-K schema、brief-specific entity priorities、entity／event／frame lineage validation、hash-bound local evidence projection、9:16 runtime candidate switching、domain-neutral region roles、exact-frame geometry preflight、版本化 `auto_bounded_clip_v1` audit、typed failure／recovery record，以及候選耗盡後的 fail-closed preview。Clip Card 可跨 brief 重用，planner 預設不做自動完整 repair 重送；geometry artifact 以來源、frame、target、track 與處理參數 fingerprint 分 variant 保存，raw usage 與本機 timing 也分開記錄。429／quota／spending-cap 會在第一個已耗盡 SDK retry 的錯誤後中止 render，不再用候選切換製造無效工作；舊 v1／v2 projection contracts 則維持原語意供既有 artifact 重現。

尚未完成的是長毛片自動 take segmentation／重拍分組、可變鏡頭數的全片編排、16:9 runtime candidate switching、coarse/dense 統一 review UI、所有較早 pipeline 階段的完整 cache fingerprint、SAM 週期語意重驗、遮擋後 re-identification、overlay layout solver 與三次穩定度報告。本文件同時包含已實作與後續設計；任何未經人工審核的建議都不得當成 production cut 或 SpatialTrack。

## 官方參考

- [Gemini Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)
- [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini media resolution](https://ai.google.dev/gemini-api/docs/media-resolution)
- [Gemini Structured Outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini Files API](https://ai.google.dev/gemini-api/docs/files)
- [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching/)
- [Gemini zero data retention](https://ai.google.dev/gemini-api/docs/zdr)
- [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html)
- [SAM 2 official repository](https://github.com/facebookresearch/sam2)
