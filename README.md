# Montagewright

Montagewright 把一個毛片資料夾、可選的音樂與 Brief，整理成一支可交付的影片。

它不只是「請模型列一份 EDL」：Gemini 負責理解素材、形成剪輯意圖與挑片；本機程式負責時間、節拍、逐幀追蹤、裁切可行性、字幕、字卡、渲染與驗收紀錄。最後除了影片，還會留下足以回答「為什麼這樣剪」的結構化報表。

```bash
montagewright render RUSHES/ \
  --brief BRIEF.md \
  --music MUSIC.mp3 \
  --aspect 9:16 \
  --seconds 90 \
  --review \
  --budget 5 \
  --output CUT/
```

也可以直接開啟本機 Web 編輯器：

```bash
montagewright-web
# http://127.0.0.1:8765/
```

> 目前是積極開發中的本機剪輯工具，不是託管服務。素材、SAM 與 FFmpeg 都在執行 Montagewright 的電腦上處理；需要語意理解的影片與音訊會依工作階段上傳到 Gemini API。

## 它現在能做什麼

- 讓 Gemini 看完整 proxy，而不是只靠檔名或文字摘要挑片。
- 產生可跨專案重用的 Clip Cards：內容、可用區段、原生運鏡、人物／產品、動作與語音角色。
- 依 Brief、音樂、指定片長與比例決定方向、選片、順序、長度和鏡頭意圖。
- 以本機量測把切點落到實際節拍，而不是採信模型猜的時間碼。
- 預設使用 SAM 2.1 逐幀追蹤 Gemini 指定的主體；也可明確關閉 SAM。
- 執行定鏡、橫移、直移、推近／拉遠、多落點、跟隨主體與沿用原素材運鏡。
- 產生 SRT，或輸出 `plain`、`speakers`、`spoken`、`plate` 四種燒錄字幕。
- 產生並編輯獨立字卡軌，支援多種版型、表面、字型、位置、層級與進出動畫。
- 在 Web UI 檢查成片、原素材、裁切框、逐顆驗收、字幕、字卡、波形、成本與未採用素材。
- 不再呼叫 Gemini，即可改進出點、順序、音量、字幕和字卡並重新輸出。
- 選配逐顆與整片 Gemini review，將可定位的問題送回重規劃。
- 輸出 Premiere XML／FCPXML、原始素材 handles、成本帳本與 `report.json`。

## 核心原則

### 語意交給模型，座標與時鐘交給程式

| 問題 | 負責者 |
| --- | --- |
| 這支素材在拍什麼、是否可用 | Gemini 看影片 |
| 哪個人／產品才是 Brief 指的主體 | Gemini |
| 哪些鏡頭該出現、順序與剪輯理由 | Gemini |
| 畫面應偏安靜、資訊密集或有方向感 | Gemini |
| 主體每一幀實際在哪裡 | SAM 2.1 + 本機座標轉換 |
| 節拍、重音與音訊時間 | 本機音訊分析 |
| 裁切是否越界、要放大多少、能否完成運鏡 | 本機幾何與 renderer |
| 字卡是否撞主體／字幕、文字是否可讀 | 本機多幀排版與對比稽核 |
| 最後輸出的每一格 | FFmpeg |

模型可以要求一種編輯行為，但不能捏造執行結果。本機做不到時會降級或停止，並把原因、量測與實際結果寫進報表。

### 決策、執行與驗收分層

Montagewright 分開保存：

1. Gemini 原本想做什麼。
2. 本機實際能做什麼。
3. 最後是否真的交付了那個意圖。

因此「計畫是推近、renderer 最後只能定住」不會在報表中仍被寫成成功推近。Web UI 也分別呈現原生素材運動、數位裁切運動與兩者疊加的結果。

### 成本上限是停止條件，不是品質旋鈕

`--budget` 是整輪工作的美元上限。每次付費呼叫送出前，Montagewright 會用最大輸出 token 先保留最壞情況預算；餘額不夠就不送出，留下當下最好的成片，而不是偷偷換成較差的判斷。

目前 production model 固定為 `gemini-3.7-flash`。費率依 Google 公告，
以 UTC 日期在每次預算保留與實際結算時自動選擇：

| Token 類型 | 至 2026-12-31（含） | 2027-01-01 起 |
| --- | ---: | ---: |
| 新 input | US$0.75 | US$1.50 |
| cached input | US$0.075 | US$0.15 |
| output／thinking | US$3.75 | US$7.50 |

這是 Montagewright 用來做預算保留與報表的固定費率，不是 Google 帳戶的 quota。API 的 429 仍可能來自 rate limit、billing、shared project quota 或帳戶層級限制。

## 安裝

### 必要條件

- Python `>=3.12,<3.13`
- FFmpeg／ffprobe
- Gemini API key
- 建議：Apple Silicon Mac；SAM 2.1 與本機轉錄在這個環境最完整

macOS 可先安裝 FFmpeg：

```bash
brew install ffmpeg
```

### Python 環境

使用 uv：

```bash
uv sync
```

或使用一般 virtual environment：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

設定 API key。Web 與 CLI 會自動讀取目前工作目錄的 `.env`；如果 shell
已經 export 同名變數，shell 的值優先、不會被 `.env` 覆蓋：

```bash
export GEMINI_API_KEY='...'
# GOOGLE_API_KEY 也可使用
```

### SAM 2.1（建議安裝）

完整的主體跟隨需要 tracking 與 segmentation extras：

```bash
uv sync --extra tracking --extra segmentation
# 或
pip install -e '.[tracking,segmentation]'
```

將 checkpoint 放在預設位置：

```text
artifacts/models/sam2.1_hiera_tiny.pt
```

也可以每次明確指定：

```bash
montagewright render RUSHES/ \
  --sam-checkpoint /path/to/sam2.1_hiera_tiny.pt \
  --output CUT/
```

SAM 是預設路徑。找不到 checkpoint 時 CLI 會顯示詳細警告並退回 Gemini 的稀疏定位；只有在你確定不需要逐幀追蹤時才使用 `--no-sam-tracking`。

### 用參考圖片鎖定人、物或地點

只有文字 Brief 時，展開的摺疊機可能被模型看成一般平板。需要鎖定「這一個」實例時，可提供一張或多張正例圖片，以及跨角度仍穩定的辨識線索：

```bash
montagewright render RUSHES/ \
  --grounding-target-id foldable.hero \
  --grounding-target-description 'Brief 指定的銀色摺疊手機；展開後仍是同一台' \
  --grounding-reference ~/Pictures/folded-front.jpg \
  --grounding-reference ~/Pictures/unfolded-screen.jpg \
  --grounding-identity-cue '相機模組與鉸鏈配置' \
  --grounding-exclusion '不要用外觀相似的平板或另一款摺疊機' \
  --output CUT/
```

Gemini 先以參考圖和正／反向條件找出候選區間，再對候選的精確影片格確認身份與框；至少兩個不同時間點確認成功後，SAM 才能接手逐幀追蹤。SAM 負責延續遮罩與位置，不會自行判斷兩個相似物件是不是同一個。任何階段無法確認都會停止該身份追蹤，而不是換成外觀相似的替代品。

Web 的「開始新一輪」提供相同的簡易欄位。需要多個目標、hard negative 或已核准的完整查詢契約時，可改用 CLI `--grounding-spec PATH` 或 Web 的進階 grounding JSON；兩條入口最後都會寫成同一份 `work/grounding-spec.json`，供續跑、報表與稽核使用。

### 本機逐字稿（macOS 26）

Apple SpeechTranscriber 提供逐字時間碼；Gemini 負責校正文意，但不提供字幕時鐘。

```bash
swiftc -parse-as-library -O \
  -o tools/transcribe/transcribe \
  tools/transcribe/Transcribe.swift
```

沒有這個工具時，純 b-roll 剪輯仍可執行；需要語音內容與精準字幕的流程會受限。

## 第一次剪輯

最小命令：

```bash
montagewright render ~/Movies/rushes \
  --aspect 9:16 \
  --output ~/Movies/cut
```

較完整的 90 秒案例：

```bash
montagewright render ~/Movies/rushes \
  --brief ~/Movies/brief.md \
  --music ~/Music/track.m4a \
  --aspect 9:16 \
  --seconds 90 \
  --speech auto \
  --subtitles burn \
  --subtitle-look spoken \
  --timeline both \
  --review \
  --budget 5 \
  --output ~/Movies/cut
```

先用少量素材驗證 prompt 或環境：

```bash
montagewright render ~/Movies/rushes \
  --sample 12 \
  --seconds 20 \
  --budget 1 \
  --output ~/Movies/test-cut
```

`--sample N` 會在整個資料夾中做固定、可重現的分散取樣，不是永遠拿前 N 支；同樣的素材仍可命中 Clip Card cache。

### 常用 render 選項

| 選項 | 行為 |
| --- | --- |
| `--brief FILE` | 創意方向、必要資訊與可選的核准字卡文字 |
| `--music FILE` | 配樂；節奏階段會聽音樂並以本機 beat grid 落點 |
| `--music-map FILE` | 使用已鎖定的音樂分析結果 |
| `--aspect` | `16:9`、`1:1`、`4:5`、`9:16` |
| `--seconds N` | 硬性的目標片長；不要只把秒數寫在 prose Brief 裡 |
| `--sample N` | 只分析固定抽樣的 N 支素材 |
| `--review` | 增加逐顆與整片 Gemini review；較慢、也較貴 |
| `--budget USD` | 整輪成本上限，預設 US$5 |
| `--speech auto\|never` | 自動只轉錄「語音構成內容」的素材，或完全不轉錄 |
| `--subtitles` | `none`、`sidecar`、`burn` |
| `--subtitle-look` | `plain`、`speakers`、`spoken`、`plate` |
| `--timeline` | `none`、`premiere`、`finalcut`、`both` |
| `--library DIR` | 共用 Clip Cards 與 transcripts 的位置 |
| `--upload-cache DIR` | 共用已上傳 Gemini media URI 的內容定址快取 |

完整選項以程式為準：

```bash
montagewright render --help
```

## Brief 怎麼被使用

Brief 是判斷依據，不是要求使用者手寫 EDL。適合描述：

- 影片目的、觀眾與語氣。
- 必須涵蓋或不能出現的內容。
- 片長與平台限制；片長仍建議同時使用 `--seconds`。
- 哪些資訊適合做成字卡。
- 字卡可以避開主體、刻意壓住哪個元素，或跟內容／音樂哪個事件同步。

一般 Markdown prose 會同時提供給剪輯規劃，並由本機抽出「可能適合做字卡」的候選。候選永遠是草稿：Gemini 可以選擇時機和設計方向，但不能把 prose 自己提升成已核准、可直接燒入的文字。

```markdown
# 方向

90 秒新品總覽。開頭要快，產品規格要清楚，但不要每顆都像規格表。

Galaxy Watch Ultra2
EN13319 國際潛水標準認證
40m

（畫面：避開手錶本體；字卡節奏偏安靜）
```

### 明確核准可直接輸出的字

如果某段文字必須逐字照用，可加入一個 `montagewright-approved-copy` JSON fence：

````markdown
```montagewright-approved-copy
{
  "version": 1,
  "items": [
    {
      "copy_id": "launch-title",
      "text": "Galaxy Z Fold 系列",
      "allowed_kinds": ["opening_title"]
    },
    {
      "copy_id": "water-rating",
      "text": "EN13319 國際潛水標準認證",
      "allowed_kinds": ["feature"]
    }
  ]
}
```
````

這個區塊會被當成使用者明確核准的 immutable copy。Gemini 可以引用 `copy_id`、安排時間與設計家族，但不能改字。普通 Brief、OCR、逐字稿或模型草稿都必須經過 Web／CLI 人工核准，才能成為可輸出的 `human_review` copy。

## 實際流程

`$` 代表 Gemini 付費呼叫，`◈` 代表可由內容 hash 命中快取。

```text
    rushes
      │
  -   proxy / scene split ◈
      │
  $   Clip Cards ◈              每支素材一次；刻意不看 Brief
      │
  -   Apple ASR ◈               只有 speech=content 才需要
  $   transcript correction     只改文字，不採用模型時間碼
      │
  $   direction                 全部 proxy + 音樂 + Brief
      │
  $   selection                 挑片、順序、運鏡、字卡候選
      │
  $   rhythm                    有音樂時決定鏡頭長度與意圖
      │
  $   semantic grounding        確認要追哪個主體／落點
  -   SAM 2.1 propagation       逐幀位置與 mask
  -   crop compiler             可行裁切路徑與降級
  -   FFmpeg render             segment → concat → mix
      │
  $   shot + cut review         只有 --review
      │
  -   report / subtitles / NLE / Web artifacts
```

Clip Card 故意不讀 Brief，因為素材沒有變時，同一張卡應能服務不同剪輯。Direction 與 Selection 則會再看 proxy，而不是只讀摘要；這是為了讓模型能針對當次 Brief 判斷摘要沒有記下來的視覺細節。

## Web UI

```bash
montagewright-web
```

預設位置是 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。可用環境變數調整：

```bash
export MONTAGEWRIGHT_RUNS="$HOME/.cache/montagewright/runs"
export MONTAGEWRIGHT_HOST=127.0.0.1
export MONTAGEWRIGHT_PORT=8877
montagewright-web
```

Web 介面包含：

- **鏡頭**：逐顆查看選片理由、原計畫、實際運動、驗收與降級；可改進出點、音量、順序與刪除鏡頭。
- **預覽工作區**：切換成片、原素材＋裁切框與並排模式，確認原生運鏡和數位裁切沒有混淆。
- **字卡**：點時間軸或畫面字卡即可選取；雙擊畫面文字可直接編輯主文字。
- **逐字稿**：修改字幕 copy 與 timing，再輸出燒錄版。
- **沒用的**：查看未入選素材與原因。
- **花費**：查看各 Gemini 階段與累計成本。
- **匯出**：重新輸出目前成片、字幕版、字卡版或 NLE timeline。

Web 的 trim／reorder／字幕／字卡操作不會重新呼叫 Gemini。結構性 recut 會建立新的 current-timeline revision，重新對齊或失效與新版時間軸不一致的字卡、字幕與 preview 衍生檔，避免畫面已換但下載仍是舊版本。

## 字卡系統

字卡的 source of truth 是 `work/graphics.json`，不是 PNG。PNG 是 production renderer 編譯出的透明圖層，讓 Web 精準預覽與 FFmpeg 輸出使用完全相同的字型、斷行、底板、描邊與幾何。

拖曳時 Web 先用輕量的互動預覽保持流暢；停止操作後再取得精準 PNG。暫時性的解碼／快取錯誤會重試並標成「上一次有效預覽」；真正的對比、越界或碰撞問題才會顯示「需要調整」。

### 設計能力

目前有八個 curated design families：

- 極簡編輯
- YouTube 強調
- 雜誌專題
- 社群貼紙
- 資訊下標
- 電影標題
- 運動娛樂
- 柔和生活

可再獨立組合：

- 內容角色：開場主標、章節、產品名、功能／規格、重點標註、結尾卡。
- Template：主視覺、產品銘牌、規格徽章、置中多行、多行規格、結尾 roster 等。
- Surface：透明、實色、pill、split、ribbon、sticker、highlight、outline、glass／霧面、editorial。
- Typography：project display／body font、主副標比例、行距、block gap、對齊、最大寬度。
- Appearance：文字色、強調字、描邊、陰影、底板透明度、邊框、圓角與 padding。
- Placement：自動留白、避開主體、刻意覆蓋主體、固定位置、自由位置、縮放與旋轉。
- Motion：fade、rise、slide left／right、進出時間與距離。
- Music sync：可把進場完成點吸附到成片實際聽到的 accent 或 downbeat；只做小幅局部校正，沒有鄰近節點時保留原時間。
- Layering：z-index、允許／避免字卡彼此重疊。

Gemini 只選擇語意層的 `family`、`surface`、`motion`、`composition` 與出現鏡頭；本機 resolver 依字型、實際文字長度、畫面比例、SAM 主體、字幕 keepout 和多幀對比決定像素。過長的自動字卡可以換成相容的較寬 template；人工鎖定的卡不會被偷偷改版。

### CLI 字卡工作流

Web 與 CLI 共用同一份 GraphicsPlan、核准規則與 production renderer：

```bash
montagewright graphics CUT/ inspect
montagewright graphics CUT/ preview --graphic-id g03
montagewright graphics CUT/ approve --graphic-id g03
montagewright graphics CUT/ validate
montagewright graphics CUT/ render
```

`approve` 是明確人工核准：它會複製當下精確文字、建立 digest，並將來源記為 `human_review`。修改核准文字後必須重新核准，不能沿用舊 digest。

`render` 會同時產生燒錄版與 `graphics-overlay.mov`。後者是含完整進出動畫的透明 ProRes 4444 圖層，可在 NLE 中保留為獨立上層；字卡仍以 `graphics.json` 為可編輯來源。

## 字幕

只做逐字稿，不重剪：

```bash
montagewright transcribe VIDEO.mp4 \
  --locale zh-TW \
  --output transcript.json
```

或處理整個資料夾：

```bash
montagewright transcribe RUSHES/ --budget 2
```

字幕時鐘來自本機辨識器，Gemini 只做文字校正。燒錄字幕與字卡會由同一個 compositor 合成，字幕位於 graphics 上層，並使用實際字幕 bbox 作為字卡排版 keepout。

## NLE timeline

在 render 時要求：

```bash
montagewright render RUSHES/ \
  --timeline both \
  --output CUT/
```

或對既有輸出重新產生：

```bash
montagewright timeline CUT/ --flavour both --rushes RUSHES/
```

Timeline 指回原始素材並保留 handles、來源時間、裁切 keyframes、marker 與 laid music audio。若已輸出 `graphics-overlay.mov`，Premiere XMEML 與 Final Cut FCPXML 也會把它放在主畫面上層。Premiere 使用 XMEML，Final Cut 使用 FCPXML。

透明 overlay 可移動、裁切、關閉或整軌替換，且畫面與 Web／CLI 輸出一致；但它不是 MOGRT／Motion Template，因此文字、字型和底板不能在 Premiere／Final Cut 的原生文字檢查器逐項修改。要改內容與設計仍回到 Web／`graphics.json` 後重輸 overlay。這項限制也適用於原生字幕圖層。

## 產出內容

主要檔案會依選項略有不同：

```text
CUT/
  deliverable.mp4                    最終混音成片
  picture.mp4                        未加音樂 bed 的畫面 master
  preview.mp4                        Web／Gemini review 用預覽
  deliverable-subtitled.mp4          燒錄字幕版（若產生）
  deliverable-graphics.mp4           燒錄字卡版（若產生）
  deliverable-graphics-subtitled.mp4 字卡＋字幕合併版（若產生）
  graphics-overlay.mov               透明 ProRes 4444 字卡軌（若產生）
  report.json                        決策、驗收、降級、成本與錯誤
  segments/                          每顆已渲染鏡頭與 handles
  work/
    current-timeline.json            Web recut 後的 current truth
    crops.json                       實際裁切 keyframes
    subtitles.json                   人工字幕修改
    graphics.json                    字卡計畫、copy facts 與 revision
    graphics-render/                 字卡編譯結果與 layout report
```

`report.json` 是交付物，不只是 debug log。即使後續階段失敗，CLI 也會盡可能先寫下已完成的方向、選片、成本、審查與停止原因。

## Cache 與重跑

- Proxy、Clip Cards、transcripts 與 SAM artifacts 以內容和執行契約定址。
- 同一批素材換 Brief，不需要重做 Brief-free Clip Cards。
- Gemini File API URI 會放在共用 upload cache；素材未變時可避免重傳，但新的規劃仍會計算 input token。
- SAM cache 會檢查 checkpoint、implementation revision、shot window、seed 與 source fingerprint；不相符就拒絕沿用。
- Web preview PNG 與 metadata 原子發布，並驗證 signature；損壞 cache 不會被當成成功預覽。

## 已知限制

- 目前使用 Google Gemini API key；尚未提供 Vertex AI backend 切換。
- `--review` 目前審查的是主要剪輯成片，字卡軌是在 review loop 後 materialize；Gemini 還不能在同一輪 review 中直接提出結構化字卡修正。
- 字卡 `music_sync` 已可在本機對齊 accent／downbeat；目前只同步進場完成點，尚未提供逐字、逐行或連續音訊反應動畫。字卡動畫 easing 目前固定為 linear。
- Web 可檢查裁切框與運鏡，但尚未提供完整的 source-time crop keyframe editor。
- Web 互動預覽不是 WebGL renderer；精準結果仍由 Pillow／FFmpeg production compiler 產生。這保證輸出一致，但第一次編譯複雜字卡仍可能需要短暫等待。
- NLE timeline 會帶透明字卡軌，但目前不會建立原生可逐字編輯的 Premiere／Final Cut 字卡或字幕物件。
- SAM 追蹤品質仍取決於 seed、遮擋、鏡頭切換與素材清晰度；對焦前後、拍攝準備動作與真正 authored camera motion 仍需要可靠的 usable-window 分析。

## 專案結構

```text
src/montagewright/
  brief.py        Brief、核准 copy 與初始字卡計畫
  clipcard.py     素材語意分析與 Clip Cards
  transcript.py  本機 ASR + Gemini 文字校正
  backfill.py     將校正文字回填到本機 word clock
  planner.py      Direction、Selection、Rhythm、Grounding、Replan
  schema.py       模型輸出與剪輯契約
  grounding.py    將意圖落到節拍與可行 source window
  capabilities.py 執行能力與 prompt 共用描述
  reframe.py      裁切路徑、插值與手動 retime
  pipeline.py     Semantic intent → 本機執行層
  executor.py     EDL → RenderPlan
  renderer.py     FFmpeg segments、concat 與混音
  review.py       逐顆與整片 Gemini review
  graphics.py     字卡 schema、design resolver 與 compositor
  subtitles.py    字幕 layout、樣式與 compositor
  timeline.py     XMEML／FCPXML
  cost.py         預算保留、計價與 ledger
  webapp.py       FastAPI 與 Web editing API
  web/            單頁剪輯介面
  measure/        SAM、音樂、場景、影像與幾何量測
  prompts/        Production prompts（繁體中文）
```

`measure/` 只負責量測，不應反向依賴規劃／渲染層；測試會守住這條 dependency boundary。

## 開發與驗證

```bash
source .venv/bin/activate
pytest -q
```

只跑字卡相關測試：

```bash
pytest -q \
  tests/test_graphics.py \
  tests/test_graphics_cli.py \
  tests/test_graphics_resolver.py \
  tests/test_brief_graphics_contract.py
```

檢查 CLI surface：

```bash
montagewright --help
montagewright render --help
montagewright graphics --help
```

設計背景與踩過的坑在 [`docs/lessons.md`](docs/lessons.md)，過去的架構提案保留在 [`docs/history/`](docs/history/)。它們是歷史資料；目前行為一律以程式、schema、CLI `--help` 與本 README 為準。
