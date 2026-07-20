# JasCueVideoLab

這是一個**完全獨立、實驗性**的 Gemini 3.5 Flash 影片理解與單幀 Grounding 驗證專案。它不是 JasCue 正式產品，不引用也不修改任何 JasCue 程式碼；實驗未通過前，不應將這裡的程式合併回 JasCue。

最新的 target-first 方法採用「未指定 target 就先提出候選，使用者選定後才找時間與 bbox」，完整的通俗說明、技術分析、實測數據與可分享摘要見 [METHODOLOGY.md](METHODOLOGY.md)。

目前的最小垂直切片是：

```text
本機影片 → ffprobe / SHA-256 → Gemini File API
        → Gemini Interactions API Structured Content Map
        → 點選 HTML 事件 → FFmpeg 抽 orientation-corrected 原始影格
        → Gemini Interactions API Structured bbox
        → Pillow debug overlay
```

## 重要界線

- `start_ms`、`end_ms` 與 `recommended_keyframe_ms` 是 **coarse semantic time**，只用於搜尋與人工瀏覽，不是 frame-accurate cut point。
- 對 Gemini 原生影片理解索取少量截圖候選時，API contract 使用官方文件慣例 `MM:SS`，不要求模型計算毫秒。程式只把合法且未超界的 `MM:SS` 換算成 FFmpeg seek 值；換算結果仍不是精確 frame time。
- `frame_pts` 與 `frame_time_ms` 是 FFmpeg 實際抽到之原始影格的媒體時間；每張 `frame.png` 都另外保存 SHA-256。
- Gemini bbox 是單張影格的人工審核 proposal，不是 pixel mask，也不是 production-ready tracking data。
- `main` baseline 沒有 ASR、transcript、字幕、temporal tracker、SAM/EdgeTAM/Apple Vision、逐幀追蹤、自動裁切、NLE timeline、FCP/Motion/FxPlug 或成片輸出。
- `experiment/dynamic-tracking` branch 另有一條明確隔離的 optional CSRT bbox propagation 實驗。它不屬於 baseline，也不得把輸出稱為 Gemini 原生 tracking 或正式 SpatialTrack。
- `REPORT-SAMPLE-CONTINUITY-COST-TIMING.md` 記錄 analysis proxy、Gemini raw usage 牌價估算、API latency 與 tracker geometric drift。成本只依官方 Standard list price 估算；free tier 與沒有 usage response 的失敗請求不得假裝成已知帳單金額。
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

SAMPLE-CONTINUITY 的首次 live B 模式在 `00:02` 選中正確中央紫色手機，bbox `[413, 664, 466, 842]`；既有 A 模式為 `[412, 684, 467, 871]`，IoU 0.738123、center distance 24.5。這是 Codex 視覺檢查而非獨立 human ground truth，且 B 的 reference frame 仍不可知。

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
  --candidate-id phone_purple_center \
  --runs 3 --ground-runs 1 --output-dir ARTIFACT/purple-phone

# 僅限 experiment/dynamic-tracking branch；用 Gemini GroundingProposal 當 seed
uv sync --extra tracking
uv run jascue-video-lab track-csrt VIDEO.mp4 \
  --grounding-json ARTIFACT/events/EVENT/groundings/ENTITY/grounding.json \
  --target-description '中央紫色 OPPO Reno16 手機' \
  --output-dir ARTIFACT/tracking-purple-phone
```

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

Contract tests 驗證 schema、entity reference、半開事件區間、禁止 `frame_accurate`、不可見目標空候選與 bbox 範圍；geometry tests 驗證 normalized-to-pixel、center distance 與 IoU；media test 會實際呼叫 FFmpeg 產生短片並確認推薦毫秒與真實 PTS 分離。這些測試不會 mock 或宣稱 Gemini live call 成功；live 成功只能由 artifacts 中的 `raw_interaction`、validated JSON 與 debug PNG 證明。

## 實際公開影片結果（2026-07-20）

已對公開影片 `【接續互通：與 iPhone 完美配合】(clean).mp4` 使用 `gemini-3.5-flash` 完成三次獨立 live run，產物在 `artifacts/real-continuity-public-3runs/`：

- 三次 Content Map 與 16 個 Grounding Proposal 全部通過本機 schema validation；`result.json` 為 `ok=true`、`failure_count=0`。
- 事件數為 5、5、6。第三次將 Introduction 獨立切出；對齊後 AirDrop start 最大差 6,000 ms，Clipboard keyframe 最大差 2,000 ms。
- 初版把 Gemini 官方 y-first box 當成專案 x-first box，造成部分正確物件被畫成橫向大框。這是 adapter bug，不是可以歸咎模型的 Grounding failure；歷史 A/B 頁只保留作錯誤分析，不再當準確率結論。
- 修正為 native y-first schema 後，同一個 19.019 秒 iPhone 螢幕以泛化描述重跑 5 次，canonical bbox 對 Codex reviewer reference 的 IoU 為 0.964–0.969，兩兩 IoU 為 0.987–1.000。
- 垂直切片與資料 contract 已跑通，但 reference 仍未經獨立真人確認，且單幀框不能成為 production／SpatialTrack tracking data。
- `report.html` 是自包含技術報告；`comparison.json` 保存 pairwise 與 reviewer-reference 對照細節；各 `run-*/index.html` 是事件 timeline。

針對 19.019 秒的相同原始影格完成座標順序診斷：

- 明確描述「右側黃色 iPhone 的螢幕、不可框整支手機或左側 MacBook」重跑 5 次，5/5 通過暫定 reviewer-reference IoU 0.8 門檻，IoU 為 0.964–0.968；兩兩 bbox IoU 為 0.993–1.000。
- 舊泛化描述 5 次中有 3 次輸出看似「橫向大框」，但將 raw box 按官方 y-first 讀取後，實際 IoU 為 0.959–0.967；另 2 次則違反指示、直接使用 x-first。真正問題是模型在自訂非原生座標順序下會混用 convention。
- 現行解法不是做 heuristic auto-swap，而是在 Structured Output field 名稱與 prompt 上都固定官方 `box_2d_yxyx`，再由本機 deterministic conversion。

第二支公開影片 `public-continuity-demo.mp4` 暴露了時間軸限制：

- ffprobe 為 116,883 ms，Gemini File API metadata 也為 117 秒；但完整影片 Content Map 反覆產生 119,000、136,000、145,000、148,000 ms 等超界時間。
- 加入 exact duration 與更短的 temporal-only schema 仍是 0/3 通過，證明問題不只來自原 prompt 複雜度。
- 一次 contract-error feedback retry 能讓 JSON 數值合法，但推薦幀 Grounding 顯示 10 個事件中有 3 個 primary/required entity 不可見；合法 schema 不等於語意正確。
- 實用 fallback 改用 4 秒 FFmpeg PTS storyboard：30 張縮圖各有 immutable frame ID，Gemini 只選 ID、不輸出時間。本機映射出 6 個事件，全部落在 0–116,883 ms；Codex 逐張檢查 6 張代表幀，片頭、四個手勢與片尾皆符合標籤。
- 三支相似 iPhone 同畫面的 70.003 秒原始影格，指定「中間手機螢幕」重跑 5 次，5/5 完全相同 bbox `[431,188,576,727]`、pairwise IoU 1.0；Codex 視覺抽查確認選中正確實例。這仍不是獨立真人 ground truth。

後續在同一支 61.862 秒接續互通影片做 A/B：直接 `MM:SS` 三次皆通過 schema 與片長檢查；第一輪 4 個時刻為 00:04、00:19、00:29、00:46，抽到的實際 PTS 為 4,004、19,019、29,029、46,013 ms，4/4 Grounding overlay 經 Codex 視覺抽查符合目標。PTS storyboard 則有 5/6 Grounding 成功，另 1 個 frame-ID 的事件描述誤稱黃色 iPhone 可見；單幀 gate 正確回傳 `visible=false`，沒有猜 bbox。

更關鍵的隔離測試是：原本會輸出 148 秒超界時間的 116.883 秒 iPhone 手勢影片，改成少量 `MM:SS` 截圖候選後三次全部合法且未超界，主要候選穩定落在 00:23、00:38–00:39、00:56–00:58、01:11–01:12、01:31–01:32。這表示問題主要是完整 Content Map 的複雜時間算術與任務負擔，而不是影片 metadata。

目前建議流程因此是：少量顯著截圖優先採 Gemini `MM:SS` → 本機片長驗證 → FFmpeg 抽幀並保存真實 PTS → 單幀 Grounding visible gate；若時間非法、目標不可見，或需要全片 coverage，再 fallback 到 PTS storyboard → Gemini frame-ID selection。4 秒取樣只提供 coarse boundary，快速 UI 另以更密的局部 PTS 網格驗證，不把任何結果當 frame-accurate cut point。

可直接開啟的第二支影片產物：

- `artifacts/iphone-gestures-native-adapter-retry-01/storyboard-pts-grid-4s-live/index.html`：可點事件跳到影片並查看代表幀。
- `.../temporal-first-3runs-live/summary.json`：精簡 prompt 0/3 的原始失敗摘要。
- `.../repeat-center-screen-70003-native-yxyx-live/summary.json`：三支相似手機的 5 次 Grounding 穩定度。
- `artifacts/iphone-gestures-content-repair-live-01/run-01/`：contract repair 成功但 semantic keyframe 抽查失敗的完整證據。

上述 IoU 僅能作為 AI reviewer-assisted 實驗指標。原始驗收條件中的「人工標註對照結果」仍待使用者或另一位獨立真人在不知道 Gemini bbox 的情況下建立／確認 reference boxes，完成前不得宣稱人工 ground-truth 驗收通過。

可攜式報告的 canonical artifact 已通過 schema/build validation，且曾在實際瀏覽器確認內容、圖表與表格可載入。自動 headless verifier 在這台 Mac 的「永久顯示捲軸」設定下，因閱讀器 top bar 多出約 8–15 px 而回報 horizontal overflow；這是報告 QA 工具的環境限制，不是實驗資料通過證明，亦未被當成驗收成功。

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
