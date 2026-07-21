# macOS 影片分割追蹤評估

本文件記錄 JasCueVideoLab 在 Apple Silicon 上比較影片物件分割追蹤 runtime 的方法與目前結果。它是實驗紀錄，不是 production 選型結論，也不把任何模型輸出視為人工真值。

## 結論先行

- 目前最可稽核的基準仍是 Meta 官方 SAM 2.1 Tiny video predictor；在本次工作負載上，CPU 比 MPS 快。
- 同一個 shot 內的多物件應共用 decode、predictor 與 inference state。這不只少做重複工作，也保留每個物件獨立的 object ID、mask、bbox 與狀態。
- EdgeTAM 的官方 PyTorch video predictor 可在 MPS 上執行真正的多物件、雙向 mask propagation。本次單一 fixture 明顯較快，但其中一個物件曾連續五格輸出空 mask，所以只能列為待擴大驗證的候選。
- EfficientTAM-Ti 也通過同一個 bbox／多物件／雙向契約；本次速度介於 EdgeTAM 與官方 SAM 2.1 Tiny CPU 之間，且不需要修改 upstream source，但 MPS 會略過 CUDA-only 的小孔洞後處理。
- MLX 原生的 SAM 2.1 temporal port 有速度潛力，但目前測到的第三方實作缺少可明確套用至程式碼的授權聲明，暫時只能隔離研究。
- Apple／Hugging Face 現成的 Core ML SAM 2.1 Tiny、Small、Large package 只涵蓋單張圖片 segmentation；模型名稱中的「SAM 2.1」不代表它已包含影片 memory pipeline。
- 不採 Gemini polygon。Gemini 負責選對語意目標並提供 bbox seed；tracker 將 bbox 精煉成 seed mask，再傳播到同一 shot 的前後影格。

## 公平比較契約

所有完成 live run 的方法盡量共用以下條件：

| 項目 | 固定條件 |
|---|---|
| 素材 | 一支約 11.5 秒、同時含兩個可追蹤前景物件的經授權真實短片；公開文件不記錄原檔名或私人路徑，媒體本身未去識別化 |
| 解碼影格 | 同一份 immutable manifest，960×540、15 FPS、173 格 |
| seed | 同一張中段影格上的兩個 bbox prompt |
| session | 一個 predictor、一個 inference state、兩個 object ID |
| 傳播 | 從 seed 向前及向後，合併成 173 格／物件 |
| 輸出 | full-resolution binary mask、derived bbox、人工審核 overlay MP4 |
| 播放檔 | H.264、yuv420p、15 FPS、173 格、保留來源 AAC 音訊 |

只有同時支援 bbox prompt、多物件共享 state、雙向 propagation 的實作，才可列入相同契約的 live 比較。若某個實作只有 point prompt、只能單向傳播，或同一 seed frame 的多物件會互相覆寫，就只做能力稽核，不以不等價的數字加入排名。

相同輸入契約不代表相同 framework build：官方 SAM 2.1 測試環境使用其已鎖定的 PyTorch 版本；EdgeTAM 因相容性使用隔離的 PyTorch 2.7.1 環境；EfficientTAM 使用另一個 Python 3.12.8／PyTorch 2.7.1 隔離環境；MLX port 使用獨立的 MLX／Python 3.14 環境。因此表格比較的是每條可實際部署候選路徑的觀測成本，而不是單一 framework 的微基準。

## 單一 fixture 的實測

以下時間只適用於這台測試主機、這份短片、目前的依賴版本與 artifact 保存方式。不同 harness 在 propagation 迴圈內做的 threshold、geometry、hash 與 mask serialization 工作也不完全相同；因此 forward + reverse 是最接近的觀測值，仍不是嚴格同範圍的 apples-to-apples benchmark。右側時間只描述各自 harness 記錄的範圍，不視為統一的端到端 benchmark。

| 實作 | 裝置 | forward + reverse | harness 記錄時間（範圍不同） | 相對判讀 |
|---|---:|---:|---:|---|
| 官方 SAM 2.1 Tiny | CPU | 153.158 s | 166.255 s | reference |
| 官方 SAM 2.1 Tiny | MPS | 183.439 s | 196.432 s | propagation 比 CPU 慢約 19.8% |
| 官方 SAM 2.1 Tiny，frames 留在 MPS | MPS | 187.964 s | 203.289 s | 比預設 MPS 又慢約 2.5% |
| 官方 SAM 2.1 Small | MPS | 約 188.9 s | 201.389 s | 與 Tiny MPS 接近；沒有真值可證明較準 |
| EdgeTAM | MPS | 63.916 s | 72.391 s | propagation 約為官方 Tiny MPS 的 2.87× |
| EfficientTAM-Ti | MPS | 89.214 s | 98.237 s（inference） | 比官方 Tiny CPU 快，慢於 EdgeTAM |
| 社群 SAM 2.1 MLX port | MLX | 80.263 s | 84.293 s | 較快，但程式碼授權未釐清 |

「較快」不等於「較準」。本次只有兩個 bbox seed 與一支短片；它不足以推論長片、遮擋、切鏡、相似物件交換、透明／反光、小物件或快速運動的表現。

完成 exact-PTS 契約後，另以目前程式重跑官方 Tiny CPU：固定間距 catalog 保留原有 173 格，再額外納入 upstream seed PTS，共 174 格；forward + reverse 為 146.169 秒，harness total 為 157.165 秒。因影格 catalog 與上表的歷史 runtime 比較多一格，此數字只證明目前程式可完整執行，不加入跨 runtime 排名。

### Peer agreement，不是 accuracy

因為目前沒有逐格人工 mask ground truth，IoU 只能表示不同 runtime 是否產生相近結果：

| 候選 vs 官方 Tiny CPU | 物件 A mean mask IoU | 物件 B mean mask IoU | 重要觀察 |
|---|---:|---:|---|
| 官方 Tiny MPS | 0.999997 | 1.000000 | 幾乎完全一致 |
| 官方 Small MPS | 0.959686 | 0.983725 | 有可見差異，但不能推論誰較正確 |
| EdgeTAM MPS | 0.926467 | 0.972384 | 物件 A 曾連續五格為空 mask |
| EfficientTAM-Ti MPS | 0.962126 | 0.975425 | 兩個物件皆有 173／173 masks；仍需人工審核 |
| 社群 SAM 2.1 MLX port | 0.963924 | 0.979522 | 物件 A 接近離開畫面時差異最大 |

所有影片都必須由人檢查。尤其是空 mask、突然擴張、身份交換、遮擋後重新出現及畫面邊界附近的結果，不能以平均 IoU 掩蓋。

## 為什麼 MPS 反而比 CPU 慢

這不代表 Apple GPU 的硬體比較弱，而是目前這個 eager PyTorch SAM video graph 與 artifact pipeline 沒有充分攤銷 MPS 成本：

1. Meta 的 macOS 範例把 MPS 支援標為 preliminary；CUDA 路徑使用 autocast，MPS 路徑在目前範例中沒有等價的 CUDA bfloat16 execution path。
2. SAM video predictor 每格包含 memory attention、索引、插值及多個較小、具時序相依性的運算。Tiny 模型的單次工作量有限，MPS kernel dispatch 與同步成本可能比平行運算收益更大。
3. 為保存可稽核的逐格 mask，本實驗會把每個物件的 logits 轉回 CPU／NumPy。這形成裝置同步點；CPU 路徑沒有相同的來回成本。
4. Meta 的 MPS demo 預設讓 decoded video frames 留在 CPU，以降低 MPS memory fragmentation。本次另跑「所有 frames 留在 MPS」仍慢約 2.5%，因此至少在這個 fixture 上，輸入 frame transfer 不是主要瓶頸。
5. 本次沒有觀察到 MPS fallback warning，所以不能把「某些算子退回 CPU」當成已證實原因。它只是其他 MPS 環境可能發生的風險。

換句話說，CPU 與 MPS 要以目標機上的完整 workload 實測，不應只看模型名稱、GPU 理論算力或 CUDA 公布的 FPS。

## 各候選的採用判斷

### 官方 SAM 2.1 PyTorch

優點：官方 video predictor、bbox prompt、多物件、雙向 propagation、完整 inference state、Apache-2.0，最適合作為 reference implementation。缺點是本次 MPS 不比 CPU 快，且 173 格雙物件仍需數分鐘。

目前建議：

- Tiny CPU 作為可重現的 reference／fallback。
- Small MPS 只在失敗 fixture 顯示 Tiny 品質不足時再比較；本次沒有 ground truth 證明 Small 較準。
- Base Plus／Large 暫不以單一成功 fixture 追加成本。先建立有遮擋、相似物件交換、快速運動及小物件的人工 golden set。
- `torch.compile`／`vos_optimized` 的官方加速資料主要來自 CUDA，不直接外推到 MPS。

### EdgeTAM

EdgeTAM 官方 video predictor 具備 temporal memory、bbox prompt、多物件與 inference state，因此不是逐張圖片 segmentation 的替代說法，而是可公平測試的 tracker。這次速度最好，但仍有三個 gate：

- 一個物件連續五格出現空 mask，必須加入失敗偵測、re-seed 與人工審核。
- 目前 PyTorch 版本遇到 non-contiguous tensor 的 `view` 相容性錯誤；隔離 PoC 以 `reshape` 修正並保存 provenance，正式採用前要確認 upstream 相容方案。
- MPS 沒有官方 CUDA hole-filling extension；雖然通常不阻礙推論，仍可能影響細小孔洞的後處理。

EdgeTAM 的 Core ML export 目前列出 image encoder、prompt encoder 與 mask decoder，repository 也附有用這些元件組成的 Core ML video-tracking demo；但本次尚未證明它等同官方 PyTorch predictor 的完整 temporal-memory 行為，也未按本比較契約驗證。不要把「可匯出 Core ML」直接等同於已驗證、可直接替換的 Core ML temporal tracker。

### EfficientTAM

原作者 repository 保留影片 tracking、支援 macOS MPS backend，並採 Apache-2.0。本次 EfficientTAM-Ti 已以同一 predictor/state、兩個 bbox object ID 與正反向 propagation 跑完；每個物件都產生 173／173 masks，且不需修改 upstream source。雙向 propagation 為 89.214 秒，完整 inference 為 98.237 秒。

限制是 MPS 無法載入其 CUDA `_C` extension，所以官方 small-hole filling 後處理在 346 次逐物件輸出中都略過。這不是 device fallback；實驗明確關閉 MPS fallback，模型仍在 `mps:0`。是否影響邊緣品質，必須由人工 mask ground truth 判斷。

### MLX ports

本次跑通的 MLX port 確實具有 video memory、bbox prompt、多物件與雙向 propagation，速度也優於官方 PyTorch CPU／MPS。不過 pinned source revision 沒有 LICENSE 檔或 package license 欄位；模型卡的權重授權不能自動補足程式碼授權。因此它目前只能做隔離研究，不能成為產品依賴。

另一個宣稱高 FPS 的 MLX 實作雖有 temporal memory，但公開介面只有 point prompt、只做 forward propagation，而且同一 seed frame 的第二個物件會覆寫第一個 conditioning output。其 FPS 也指 feature preload 後的 propagation，而不是影片解碼、特徵計算、雙向傳播與輸出編碼的端到端時間，所以未加入 live 排名。

### Apple／Hugging Face Core ML SAM 2.1

Tiny、Small、Large 的現成 package 都可做單張影像 promptable segmentation，但相關 Core ML conversion branch 明確寫著目前只支援 image segmentation，video segmentation 尚在開發。缺少的不是模型大小，而是完整 temporal memory encoder／attention／state orchestration；Large 不能補上這個能力差異。

## 對 JasCueVideoLab 的下一步

1. 保留「Gemini bbox → tracker seed mask」；不把 Gemini polygon 當主要 seed，也不把 Gemini bbox 當最終逐格 geometry。
2. 同一 shot 的多目標共用一個 predictor／state；遇到切鏡就結束 track，下一個 shot 重新定位。
3. 在 tracker 輸出中保留 `tracked`、`reacquired`、`occluded`、`low_confidence`、`drift_suspected`、`lost`，並把空 mask 視為需要處理的訊號。
4. 建立逐格人工 golden subset，至少覆蓋遮擋、離畫再入畫、相似物件交會、快速運動、反光與小目標。
5. 分開評估 seed correctness、mask IoU、bbox IoU、identity switch、遮擋恢復、延遲、記憶體與人工修正時間。
6. 所有 review MP4 維持正常播放時長、H.264/yuv420p、來源音訊與完整 frame count；sample FPS 必須清楚標示，不能讓播放 FPS 冒充推論 FPS。
7. EdgeTAM 先列為最快的 licensed experimental candidate，EfficientTAM-Ti 作為不需 upstream patch 的次快候選，官方 SAM 2.1 Tiny CPU 保留為 reference。只有跨多種 fixture 的人工驗證通過後，才調整預設 runtime。

## 來源

- [Meta SAM 2 官方 repository](https://github.com/facebookresearch/sam2)
- [Meta EdgeTAM 官方 repository](https://github.com/facebookresearch/EdgeTAM)
- [EfficientTAM 原作者 repository](https://github.com/yformer/EfficientTAM)
- [Hugging Face Core ML conversion branch](https://github.com/huggingface/segment-anything-2/tree/coreml-conversion)
- [Apple Core ML SAM 2.1 Tiny](https://huggingface.co/apple/coreml-sam2.1-tiny)
- [Apple Core ML SAM 2.1 Large](https://huggingface.co/apple/coreml-sam2.1-large)

Live fixtures、mask、影片、checkpoint 與第三方隔離環境都保存在 gitignored `artifacts/`，不進 repository，也不在公開文件中保存原始素材名稱或絕對路徑。
