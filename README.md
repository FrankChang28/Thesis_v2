# 跨個體 NIRS 預測專案（136 人）

此目錄為最終碩論專案。原始資料仍使用154筆subject的排列，`configs/subject_cohort.json`排除18筆；實際136-fold，各折125 train／10 validation／1 test。保留原始subject IDs，不將資料索引改成0–135。

## 目錄

- `configs/`：兩階段、baseline、fusion與控制實驗設定。
- `data/raw/`：原始DTOF／DRS；`data/metadata_aligned.csv`為已對齊資料的metadata。
- `src/`、`scripts/`：模型、訓練、匯出與正式分析入口。
- `notebooks/`：論文正式圖表與結果分析。
- `artifacts/stage1/<原始ID>/relat_cons_32/`：Stage1模型與特徵。
- `artifacts/stage2/<fusion>/<strategy>/<feature>/<target>/`：Stage2模型與預測；baseline位於`stage2/baseline/<target>`。
- `cache/stage2/`：共用前處理；`results/tables/`、`results/figures/`：論文結果。

目錄不再含cohort雜湊；根目錄的`.cohort.json`與各模型metadata保留名單識別，防止混用其他cohort。若日後更改名單，應指定獨立輸出／cache並更新該輸出根的marker，不可沿用舊模型。

## 執行

在專案根目錄使用既有yc_ae環境，或安裝`pip install -e ".[notebook,test]"`。可設定`SUBJECT_NIRS_PYTHON`指定Python。

```bash
bash scripts/run_stage1_loso.sh
bash scripts/run_stage2_loso.sh configs/stage2/baseline/hc.yaml
python scripts/run_remaining_stage2.py --target hc --dry-run
```

Stage1會保留已完成模型與匯出；Stage2依CSV／checkpoint續跑。不要在同一輸出目錄同時執行兩個訓練程序。

## 在另一台電腦使用 Stage2 快取

完整複製 `cache/stage2/` 後，Stage2 訓練不必攜帶原始 DRS `.mat`，也不需新增參數。須保留 validation 的三個 `.npy` 與 metadata，以及目前設定所需的每個 subject／stage 的 compact training `.npy` 與 metadata。兩種目標共用同一份含 tHb／StO₂ 的快取；請維持原 cohort、抽樣與前處理設定，且不要啟用 `rebuild_preprocessed_cache`／`rebuild_compact_cache`。

若快取缺檔或要求重建，程式仍須讀取 `mat_file`；不會跳過缺少的 subject。DTOF 融合另需各 fold 的 Stage1 特徵，fine-tune 另需對應目標的 baseline checkpoint，厚度融合另需 metadata CSV。Stage1 重訓及需要原始 DTOF 的分析仍需原始資料；`data/raw` 的符號連結只在這些情況下才需重新接上。新電腦的正規化紀錄會按現有路徑與快取內容驗證／建立，不依賴原電腦的絕對路徑。

## 目前採用的訓練設定

六組 tHb fine-tune（DTOF／metadata × Concat／FiLM／gated residual）統一採 DRS LR=1e-6、融合 LR=3e-5。使用原本的 `*_hc.yaml`，結果位於各自的 `artifacts/stage2/<fusion>/finetune/<feature>/hc`。舊 fine-tune 結果已移出專案至外部備份；不再保留 `hc_lr3e5` 或同名版本 YAML。

六組 tHb scratch 也統一使用漸進融合：隨機初始化、DRS 與融合 LR皆為1e-4，首一個epoch內 alpha由0升至1。三種方法共用 `h_out=(1-alpha)*h_DRS+alpha*h_fused`；因此FiLM的調變與gated residual的殘差也逐步加入。訓練持續共同更新DRS與融合，驗證／測試一律完整融合。使用各自 `scratch/*_hc.yaml` 與正式hc；舊五組scratch已移出專案外備份。StO₂已同步相同的新LR與漸進scratch策略，詳見下節。

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_remaining_stage2.py --target hc
```

佇列先確認 baseline，然後跑 DTOF，再跑 metadata；完成設定與完整 folds 自動跳過。加 `--dry-run` 可預覽。更新後僅baseline、DTOF Concat漸進scratch與DTOF Concat新fine-tune已完整完成，初始audit為3complete／10pending。其餘五組scratch與五組fine-tune依DTOF優先、metadata接續；partial fold會重新訓練。

未來額外版本仍可使用 `output_variant` 分開輸出。完整來源、外部備份與模型數值驗證記於 `PROJECT_MANIFEST.json`。


## StO₂：另一台電腦接續訓練

六組StO₂ fine-tune（DTOF／metadata × 三種fusion）採融合LR=3e-5、DRS LR=1e-6，不加ramp；六組scratch採兩分支LR=1e-4、首1epoch漸進融合。沿用各自 `*_sto2.yaml` 與正式 `sto2/`，不新增結果版本層。Baseline與共用cache不變。

舊StO₂ Concat fine-tune DTOF136折／metadata98折已移至專案外備份，避免resume誤用舊LR結果。同步當下主要矩陣應為1complete（baseline）／12pending；後續進度以實際audit為準。

```bash
python -u scripts/run_remaining_stage2.py --target sto2 --dry-run
python -u scripts/run_remaining_stage2.py --target sto2
```

在另一台保留完整程式／configs／metadata／cache，以及 `artifacts/.cohort.json`、Stage1每fold features、`artifacts/stage2/baseline/sto2/`。啟用相容Python環境後從專案根目錄執行。佇列先DTOF、後metadata；baseline136折會跳過。跑完將各fusion／strategy／feature底下的 `sto2/` 連同configs／CSV／checkpoint等結果整個合併回原機，保留原機 `hc/`，且避免兩台同時寫同一結果目錄。


## Stage2 performance table與Fig. 33更新

執行 `notebooks/05_stage2_fusion_comparison.ipynb` 會掃描完整136人結果，主表列pooled RMSE、subject RMSE median [Q25, Q75]、平均逐人絕對Bias、平均配對ΔRMSE [95% CI]、改善人數百分比與固定baseline Q1–Q4的平均ΔRMSE。StO₂誤差以percentage points呈現；pooled依實際n_test加權，缺sample count的舊CSV不臆測pooled分數。

原Fig. 33四格個體ΔRMSE圖與fallback預覽已從正常notebook／CLI流程移除；各完整arm的Q統計不再等待兩target的整個融合矩陣完成。既有代表性比較圖保留，舊繪圖helper只留作歷史重現。主表不放p；原配對Wilcoxon、Holm及rank-biserial保留補充數值CSV。5000次固定seed的subject-level配對bootstrap CI為OOF近似，不涵蓋完整重訓、LOSO相依及策略搜尋；矩陣未全完成前p校正仍為暫定，不補舊設定結果。
