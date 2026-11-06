# 跨個體 NIRS 預測專案

本專案包含碩士論文使用的兩階段流程：第一階段從光子飛行時間分布（distribution of time of flight, DTOF）學習個體描述子；第二階段以漫反射光譜（diffuse reflectance spectroscopy, DRS）為主要輸入，結合個體資訊，預測灰質總血紅素濃度（tHb）與組織血氧飽和度（StO₂）。實驗比較 DRS-only baseline、32 維 DTOF 描述子及組織厚度特徵，並評估不同融合方式與訓練策略。

主要實驗採用 136 名個體的 leave-one-subject-out（LOSO）評估。以下指令均在專案根目錄（`pyproject.toml` 所在位置）執行；完整訓練需要另外準備研究資料及相應運算資源。

## 研究流程

1. **DTOF 表徵學習**：以共用 temporal CNN 編碼各 SDS 的 DTOF 形狀與相對面積，透過個體原型對比學習與個體內緊湊性目標，學習單位長度的個體表徵。每名個體的描述子由最接近其球面中心的 acquisition embedding 代表。DTOF 先將每條曲線第一段連續 5 個非零 bin 之前的孤立 bin 設為 0（低於可靠計數），再計算面積；形狀取各 bin 佔該曲線面積比例的 log10，以固定範圍 [−10, 0] 線性縮放至 [0, 1]（空 bin 落在 −10）；面積為 0 的曲線直接報錯。
2. **DRS 絕對參數預測**：以 CNN 建立 DRS-only baseline，再分別加入 DTOF 描述子或四維組織厚度特徵。融合方式包含 Concatenation、FiLM 與 Gated residual；訓練策略包含 Scratch 與 Fine-tuning。
3. **跨個體評估**：每折保留 1 名測試個體、10 名驗證個體，其餘 125 名用於訓練。前處理與特徵標準化使用訓練個體估計，驗證資料用於模型選擇，測試個體用於最終評估。

個體名單與切分由 [`configs/subject_cohort.json`](configs/subject_cohort.json) 管理。資料與模型保留原始 subject ID，因此原始陣列／設定中的 154 筆索引不等同於實際使用人數；正式 LOSO 評估涵蓋名單中的 136 名個體。

## 目錄導覽

| 目錄 | 用途 |
| --- | --- |
| `configs/` | 第一、二階段及控制實驗的 YAML 設定與個體名單。 |
| `data/` | 原始 DTOF、DRS 與已對齊的組織厚度資料。 |
| `src/subject_nirs/stage1/` | DTOF 表徵學習、描述子匯出及第一階段分析。 |
| `src/subject_nirs/stage2/` | DRS 預測、特徵融合、LOSO 訓練及第二階段分析。 |
| `scripts/` | 訓練、匯出與續跑的執行入口。 |
| `notebooks/` | 依論文分析順序整理的圖表與結果檢視。 |
| `artifacts/` | 模型權重、逐折設定及測試預測。 |
| `cache/` | 前處理資料與各訓練折的正規化紀錄。 |
| `results/tables/`、`results/figures/` | 分析表格與圖檔。 |

版本控制提供程式與設定；研究資料、快取、模型、notebook 與結果另行準備。上表也列出完整研究工作目錄中使用的分析與輸出位置。Notebook 負責串接分析與展示，主要計算及繪圖邏輯位於 `src/`。

## 安裝

訓練流程使用 Linux、Python 3.10 以上版本及 PyTorch。請依本機 GPU／CUDA 環境安裝適合的 PyTorch，再安裝專案與分析依賴：

```bash
python -m pip install -e ".[notebook]"
```

亦可使用既有 Conda 環境。Shell 執行入口可透過 `SUBJECT_NIRS_PYTHON` 指定 Python；GPU 可透過 `CUDA_VISIBLE_DEVICES` 選擇。

## 資料準備

原始資料放置方式如下；也可使用符號連結指向實際儲存位置：

```text
data/
├── raw/
│   ├── stage1/
│   │   ├── dataset_dtof.mat
│   │   └── dataset_OP.mat
│   └── stage2/
│       └── NIRS_Absolute_Dataset_grid.mat
└── metadata_aligned.csv
```

`metadata_aligned.csv` 的資料列須與原始 subject ID 對齊；Stage 2 的厚度特徵讀取第 6–9 欄（Python 切片 `5:9`）。輸入／輸出路徑與實驗參數均由 `configs/` 中的 YAML 管理。程式中的目標名稱 `hc` 對應 tHb，`sto2` 對應 StO₂。

訓練前須建立個體名單標記；Stage 1 與 Stage 2 都會檢查，缺少時會回報 `Output root requires a matching .cohort.json marker`。指紋須與 `configs/stage1_32.yaml` 的 `cohort_fingerprint` 相同：

```bash
mkdir -p artifacts cache
echo '{"fingerprint": "01deae79e5b1d272"}' > artifacts/.cohort.json
cp artifacts/.cohort.json cache/.cohort.json
```

若直接使用既有模型或快取，須使用相同個體名單、資料前處理及切分所產生的檔案。DTOF 融合需要各 fold 的 Stage 1 描述子；Fine-tuning 另需相同目標與 fold 的 DRS-only baseline checkpoint。

## 使用順序

### 1. 第一階段訓練與描述子匯出

```bash
bash scripts/run_stage1_loso.sh
```

此入口執行 136-fold LOSO，並匯出 Stage 2 所需的描述子與測試個體 acquisition embeddings。主要設定為 [`configs/stage1_32.yaml`](configs/stage1_32.yaml)，輸出目錄名稱取自設定中的 `experiment_name`：

```text
artifacts/stage1/<test_id>/relat_cons_32/
├── best_model.pt
├── subject_features_all_op.npz
└── test_acquisition_embeddings.npz
```

已完成且通過檢查的模型與匯出會保留；checkpoint 若不是目前的 DTOF 前處理版本，入口會直接報錯而不是沿用。不加 `--execute` 只列出待執行指令。

若只需要匯出既有模型的描述子，可執行：

```bash
python scripts/run_subject_cohort.py --phase exports --execute
```

### 2. 第二階段 baseline 與融合實驗

先檢查待執行的實驗，再選擇預測目標：

```bash
python scripts/run_remaining_stage2.py --dry-run
python -u scripts/run_remaining_stage2.py --target hc
python -u scripts/run_remaining_stage2.py --target sto2
```

每個目標包含一組 baseline，以及 DTOF／厚度特徵、三種融合方式與兩種訓練策略的 12 組融合實驗。執行順序為 baseline、DTOF、厚度特徵；完整的 fold 會跳過，未完成的 fold 會重新訓練。同一輸出目錄應由單一訓練程序使用。

單獨執行特定設定時，可使用：

```bash
bash scripts/run_stage2_loso.sh configs/stage2/baseline/hc.yaml
bash scripts/run_stage2_loso.sh configs/stage2/concatenation/finetune/dtof_hc.yaml
bash scripts/run_stage2_loso.sh configs/stage2/concatenation/scratch/dtof_hc.yaml
```

FiLM、Gated residual、厚度特徵及 StO₂ 的設定分別位於對應的 `configs/stage2/<fusion>/<strategy>/` 目錄。各設定直接定義初始化、學習率、融合方式及訓練參數。DTOF 實驗的每個 fold 會記錄所用描述子檔案的 SHA-256；續跑時若描述子已改變，會拒絕沿用舊 fold。

| 訓練策略 | DRS 初始化 | DRS 初始 LR | 融合初始 LR | 融合介入方式 |
| --- | --- | --- | --- | --- |
| DRS-only baseline | 隨機初始化 | `1e-4` | — | 僅使用 DRS。 |
| Scratch | 隨機初始化 | `1e-4` | `1e-4` | 首個 epoch 逐步增加融合強度，兩分支共同訓練。 |
| Fine-tuning | 對應 fold 的 baseline | `1e-6` | `3e-5` | 使用完整融合，微調 DRS 與融合參數。 |

Scratch 的融合強度在首個 epoch 由 0 增至 1，驗證與測試皆使用完整融合。三種融合方式共用此訓練流程；具體參數以所選 YAML 為準。

結果目錄結構為：

```text
artifacts/stage2/baseline/<target>/
artifacts/stage2/<fusion>/<strategy>/<feature>/<target>/
```

其中 `<feature>` 為 `dtof` 或 `metadata`，`<target>` 為 `hc` 或 `sto2`。各實驗包含 `loso_results.csv`、逐 fold checkpoint、設定及預測輸出。

### 3. 使用既有 Stage 2 快取

完整的 `cache/stage2/` 包含 validation 資料及所需的 compact training 子集時，可直接執行第二階段，無須重新讀取原始 DRS MAT。兩種目標共用含 tHb／StO₂ 的快取；請維持相同的個體名單、抽樣及前處理設定。

搬移研究工作目錄時，亦須保留 `artifacts/.cohort.json`、所需的 Stage 1 特徵、baseline 權重及 `data/metadata_aligned.csv`。快取缺檔或啟用重建時仍需要原始資料；Stage 1 訓練及使用原始 DTOF 的分析也需要對應 MAT 檔。

## 結果分析

分析內容依研究問題分工：

| 分析 | 對應 notebook |
| --- | --- |
| DTOF 個體差異 | `00_dtof_subject_variation.ipynb` |
| 個體檢索與表徵分布 | `01_stage1_retrieval.ipynb` |
| OP 穩健性與 CKA | `02_stage1_op_robustness_cka.ipynb` |
| 結構厚度 probing | `03_stage1_structure_probe.ipynb` |
| DRS-only baseline 評估 | `04_stage2_baseline_evaluation.ipynb` |
| 融合策略與模型效能比較 | `05_stage2_fusion_comparison.ipynb` |
| 困難個體與光譜分析 | `06_stage2_subject_failure.ipynb` |
| 未見資料評估 | `07_unseen_dataset_validation.ipynb`、`07_unseen_dataset_validation_sto2.ipynb` |
| 描述子控制實驗 | `08_stage2_descriptor_controls.ipynb` |

取得 notebook 與必要資料後，建議依編號閱讀。也可直接執行分析模組，例如：

```bash
python -m subject_nirs.stage2.comparison --root ./artifacts/stage2 --output-dir ./results/tables/stage2/fusion_comparison
```

第二階段效能主表以 subject RMSE 中位數 [Q25, Q75] 呈現個體誤差分布，另列平均逐人絕對 Bias、平均配對 ΔRMSE [95% CI]、改善個體比例，以及依 baseline 難度固定分組的 Q1–Q4 平均 ΔRMSE。ΔRMSE 定義為融合模型減 baseline，負值表示改善；tHb 單位為 μM，StO₂ 誤差以百分點（percentage points）呈現。

配對統計檢定與效果量保留於補充 CSV。CI 使用 subject-level 配對 bootstrap，屬於 OOF 結果的近似描述，並不涵蓋完整重訓、LOSO 訓練集重疊及超參數選擇的不確定性。分析預設採用完整的 LOSO 實驗。

已有 CKA 摘要時，可直接重畫圖，不需重新推論模型：

```bash
python -m subject_nirs.stage1.cka_plotting --analysis_dir results/tables/stage1/cka_relat_cons_32
```

自訂輸出位置時，請將 `--analysis_dir` 指向含 `full_cka_summary.h5` 與 transition 摘要表的目錄。
