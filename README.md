Energy Vehicle Supply Chain Finance Risk Assessment: Data and Code

## 1. Project Overview

This repository accompanies the study *Supply Chain Finance Risk Assessment of New Energy Vehicles Based on the Fuzzy Cloud Attention Neural Network*. It provides the quarterly data-processing pipeline, risk labels, model definitions, training routines, evaluation functions, and figures for the Fuzzy Cloud Attention–Bidirectional Long Short-Term Memory–Hybrid Support Vector Machine/K-Nearest Neighbors (FCA-BiLSTM-HSK) model.

The workflow combines a fuzzy cloud representation of financial and market indicators, a bidirectional Long Short-Term Memory (Bi-LSTM) neural network with temporal attention, and a cascaded Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) classifier. Project files are organized into `01_data` (data and preparation), `02_code` (models and numerical evaluation), `03_figures` (plotting), `04_outputs` (stored numerical and graphical outputs), `05_docs` (environment and file indexes), and `tests` (implementation tests).

## 2. Study Data and Sample Design

| Setting | Specification |
|---|---|
| Sample | 138 A-share listed companies in China's new energy vehicle supply chain |
| Industry groups | Vehicle manufacturing: 42; power batteries: 31; core components: 65 |
| Observation period | 2018 Q1–2023 Q4 |
| Data frequency | Quarterly |
| Sliding-window length | 8 quarters |
| Sliding-window stride | 1 quarter |
| Candidate windows | 2,346 |
| Valid windows | 2,141 |
| Training windows | 1,039; window-end quarter no later than 2021 Q4 |
| Testing windows | 1,102; window-end quarter no earlier than 2022 Q1 |
| Training-period validation segment | Windows ending from 2021 Q1 through 2021 Q4 |
| Risk classes | Low risk = 1; medium risk = 2; high risk = 3 |
| Random seeds | 1, 2, 3, 4, 5 |

Quarterly financial statements, notes, and market variables are organized from CNEquity, with event records and associated public-information data collected using AKShare-related resources. Variables are aligned to their disclosure dates, and missing values are filled using historical observations only. Z-score standardization uses parameters fitted on the training data, which are then applied to the other partitions.

The nine model input indicators are debt-to-assets ratio (`lev`), debt-to-equity ratio (`debt_equity`), current ratio (`current_ratio`), interest coverage ratio (`interest_cover`), beta coefficient (`beta`), Weighted Average Cost of Capital (`wacc`), government subsidies as a share of operating revenue (`gov_subsidy_ratio`), warranty provisions as a share of operating costs (`warranty_ratio`), and lithium-price exposure (`lithium_exposure`). Other fields in the quarterly panel support preprocessing and indicator construction rather than serving as the nine designated input variables.

Risk classes are assigned from disclosed events rather than thresholds applied to the nine model inputs. High-risk events include special treatment status (ST/*ST), modified audit opinions, substantive bond principal or interest defaults, and commercial acceptance bill overdue-list events. Medium-risk events include rating downgrades or negative outlook changes, abnormal credit-spread increases, and public regulatory inquiries. Windows without these triggers are assigned to the low-risk class. Event indicators and model input features are stored separately.

## 3. Model Architecture and Parameter Settings

### 3.1 Fuzzy Cloud Feature Mapping

An inverse normal cloud generator transforms each indicator into three cloud digital characteristics: expectation (`Ex`), entropy (`En`), and hyper-entropy (`He`). The rolling estimation window spans up to the most recent **4 quarters**, with no use of future observations:

$$
Ex=\frac{1}{n}\sum_{i=1}^{n}x_i,
\qquad
En=\sqrt{\frac{\pi}{2}}\,\frac{1}{n}\sum_{i=1}^{n}|x_i-Ex|,
$$

$$
S^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-Ex)^2,
\qquad
He=\sqrt{\max(S^2-En^2,\,0)}.
$$

Here, `n` is the number of available observations in the rolling segment; a single available observation is represented by its value with zero entropy and hyper-entropy. Mapping nine indicators to three cloud characteristics produces **27 cloud features per time step** while retaining an **8-quarter sequence**.

### 3.2 Temporal Representation and Cascaded Classification

A Bi-LSTM network learns representations in both temporal directions and concatenates the two hidden states. The temporal attention module assigns normalized weights to the time steps and produces an attention-weighted representation for downstream classification.

The SVM classifier uses a **One-vs-Rest (OvR)** multiclass formulation with a **Radial Basis Function (RBF)** kernel. Routing confidence is the difference between the largest and second-largest SVM decision scores. Windows with confidence below the fixed threshold are passed to the KNN classifier for a distance-weighted local decision; all other windows retain the SVM prediction. KNN uses Euclidean distance and inverse-distance voting without smoothing.

### 3.3 Main Model Hyperparameters

The values below follow the model specification in Table 3 of the paper and the corresponding constants in `02_code/config.py`.

| Component | Parameter | Value |
|---|---|---|
| Inverse normal cloud generator | Rolling estimation length | 4 quarters |
| Inverse normal cloud generator | Cloud output | `Ex`, `En`, `He` |
| Inverse normal cloud generator | Hyper-entropy safeguard | `max(S² − En², 0)` |
| Bi-LSTM | Hidden units per direction | 128 |
| Bi-LSTM | Concatenated feature dimension | 256 |
| Temporal attention | Attention mapping dimension | 64 |
| Attention output head | Dropout | 0.2; attention output head only |
| Optimization | Optimizer | Adam |
| Optimization | Initial learning rate | 0.001 |
| Optimization | Weight decay | `1e-5` |
| Neural-network training | Batch size | 32 |
| Neural-network training | Maximum epochs | 200 |
| Neural-network training | Validation objective | Cross-entropy loss |
| Neural-network training | Early-stopping patience | 20 epochs |
| SVM | Multiclass strategy | One-vs-Rest |
| SVM | Kernel | RBF |
| SVM | Kernel parameter, `γ` | 0.05 |
| SVM | Penalty parameter, `C` | 10.0 |
| KNN | Number of neighbors, `k` | 5 |
| KNN | Distance and voting | Euclidean distance; inverse-distance weights; no smoothing |
| Cascaded routing | Fixed confidence threshold, `θ*` | 0.35 |
| Threshold calibration | Candidate grid | 0.10–0.60 in increments of 0.05 |
| Repeated experiments | Random seeds | 1, 2, 3, 4, 5 |

### 3.4 Comparison Models and Ablation Configurations

The comparison methods include a standard Long Short-Term Memory network (LSTM), Gated Recurrent Unit network (GRU), Temporal Convolutional Network (TCN), SVM, and Extreme Gradient Boosting (XGBoost). Their architecture and model parameters are as follows:

| Comparison model | Configuration |
|---|---|
| Standard LSTM | 128 hidden units; one recurrent layer |
| GRU | 128 hidden units; one recurrent layer; no dropout |
| TCN | Kernel size 3; dilation factors 1, 2, 4, 8; 64 residual channels |
| SVM | OvR classification; RBF kernel; `C = 10.0`; `γ = 0.05` |
| XGBoost | Maximum tree depth 6; learning rate 0.05; row subsampling 0.8; column subsampling 0.8; maximum 200 boosting estimators |

The ablation configurations follow Table 5 of the paper: the basic Bi-LSTM–SVM model, the model without fuzzy cloud features, the model without temporal attention, the model without KNN correction, and the complete FCA-BiLSTM-HSK model. The corresponding variants are defined in `02_code/train.py`.

### 3.5 Computational Environment and Evaluation Metrics

The computational configuration reported in Table 3 of the paper is:

| Item | Specification |
|---|---|
| Operating system | Ubuntu 20.04 LTS |
| Processor | Intel Core i9-12900K |
| Graphics processor | NVIDIA GeForce RTX 3090 (24 GB) |
| System memory | 64 GB DDR4 |
| Python | 3.8 |
| PyTorch | 1.10.0 |
| Polars | 0.19 |
| DuckDB | 0.9 |

Evaluation functions cover classification accuracy, per-class precision, recall and F1 score, macro Area Under the Receiver Operating Characteristic Curve (AUC), and high-risk Average Precision (AP). The package also includes a paired McNemar test and a company-cluster bootstrap procedure with **1,000 resampling iterations**. Per-seed metric records and window-level predictions are stored in `04_outputs/predictions/metrics_summary.csv` and `04_outputs/predictions/predictions_all.csv`, respectively.

## 4. Execution

Execute commands from the project root directory.

Install the Python dependencies listed in the package:

```bash
python -m pip install -r 05_docs/requirements.txt
```

Recalculate evaluation statistics from the included prediction snapshot and run implementation tests:

```bash
python run_all.py --audit-snapshot
```

Build the data and run the model comparison and ablation workflows using the included historical dataset:

```bash
python run_all.py --data-mode legacy --stage both
```

Run the stored-results plotting script separately:

```bash
python 03_figures/make_figures.py
```

The `--stage main` option selects model comparisons, while `--stage ablation` selects the ablation study. New training outputs are written under `04_outputs/retrained/`, separately from the existing files in `04_outputs/predictions/`.

## 5. File Reference

Each entry below provides a one-sentence description of a file in the supplied ZIP archive. Paths are relative to the repository root.


### Project entry points

| File | Description |
|---|---|
| `run_all.py` | Coordinates data preparation, training, metric compilation, and figure generation. |
| `run_checks.py` | Recomputes evaluation values from the stored prediction files and runs consistency checks. |

### 01_data — Data and preprocessing

| File | Description |
|---|---|
| `01_data/akshare_interface_spec.csv` | Lists the fields, time coverage, and purposes of the public-information interfaces. |
| `01_data/build_data.py` | Constructs quarterly panels, event-based labels, sliding windows, and model-ready tensors. |
| `01_data/cnequity_dataset_spec.csv` | Lists export fields, filters, and time coverage for the CNEquity datasets. |
| `01_data/commercial_paper_source_review.csv` | Indexes source-reference information for commercial acceptance bill overdue events. |
| `01_data/company_universe.csv` | Lists the study companies, security identifiers, and supply-chain categories. |
| `01_data/events_firm_quarter.csv` | Stores company-quarter event flags used to assign risk categories. |
| `01_data/extract_akshare.py` | Defines functions for collecting public disclosures and market-event information. |
| `01_data/extract_cnequity.py` | Defines extraction routines for quarterly financial, market, and note-based indicators. |
| `01_data/extraction_manifest.csv` | Indexes exported data files, field counts, and associated file metadata. |
| `01_data/field_dictionary.csv` | Defines the variable names and meanings used in the data pipeline. |
| `01_data/interim/build_manifest.json` | Records the panel-building parameters and the window-partition summary. |
| `01_data/interim/events.csv` | Stores security-quarter event records prepared for risk labeling. |
| `01_data/interim/events_firm_quarter.csv` | Stores event indicators aggregated by firm and quarter. |
| `01_data/interim/firm_meta.csv` | Stores the company metadata used for modeling. |
| `01_data/interim/label_source.csv` | Links quarterly risk labels to their associated event categories. |
| `01_data/interim/panel_quarterly.csv` | Stores the aligned quarterly company-level feature panel. |
| `01_data/interim/preprocessing_report.md` | Documents the data filtering, temporal alignment, and feature-preparation settings. |
| `01_data/interim/window_tensor.npz` | Stores sliding-window feature tensors, class labels, and grouping indices. |
| `01_data/interim/windows.csv` | Records each window, its inclusion status, and its training or testing assignment. |
| `01_data/preprocess.py` | Implements causal filling, temporal splits, and other panel preprocessing operations. |
| `01_data/raw/akshare/audit_opinion.csv` | Contains audit-opinion event records with relevant disclosure details. |
| `01_data/raw/akshare/bond_interest_default.csv` | Contains publicly reported bond principal or interest default events. |
| `01_data/raw/akshare/bond_rating_change.csv` | Contains credit-rating changes and outlook events. |
| `01_data/raw/akshare/bond_yield_quarterly.csv` | Contains quarterly bond-yield and credit-spread observations. |
| `01_data/raw/akshare/commercial_paper_overdue.csv` | Contains commercial acceptance bill overdue-event observations. |
| `01_data/raw/akshare/regulatory_inquiry.csv` | Contains exchange inquiry and regulatory-attention events. |
| `01_data/raw/akshare/st_status.csv` | Contains special treatment status events and corresponding dates. |
| `01_data/raw/cnequity/beta_quarterly.csv` | Contains quarterly beta estimates and their estimation-window fields. |
| `01_data/raw/cnequity/financial_statement_items.csv` | Contains consolidated financial-statement items and disclosure dates. |
| `01_data/raw/cnequity/lithium_price.csv` | Contains quarterly lithium carbonate price observations. |
| `01_data/raw/cnequity/notes_cost_structure.csv` | Contains raw-material cost composition extracted from financial-statement notes. |
| `01_data/raw/cnequity/notes_gov_subsidy.csv` | Contains government-subsidy amounts from financial-statement notes. |
| `01_data/raw/cnequity/notes_warranty_provision.csv` | Contains warranty provision items from financial-statement notes. |
| `01_data/raw/cnequity/wacc_quarterly.csv` | Contains quarterly Weighted Average Cost of Capital estimates. |
| `01_data/raw/official/commercial_paper_verified.csv` | Defines the structured fields used for official commercial bill overdue-list records. |
| `01_data/raw/official/st_status_verified.csv` | Defines the structured fields used for official special treatment announcements. |
| `01_data/raw_file_hash_manifest.csv` | Lists original data-file sizes and SHA-256 checksums. |
| `01_data/source_audit.py` | Provides source-record field checks and document-reference validation routines. |
| `01_data/st_source_review.csv` | Indexes source references associated with special treatment status events. |

### 02_code — Modeling and statistics

| File | Description |
|---|---|
| `02_code/calibrate_theta.py` | Computes routing statistics across candidate confidence thresholds. |
| `02_code/cloud_features.py` | Implements inverse normal cloud feature calculations. |
| `02_code/compile_retrained.py` | Collects per-seed metrics and tables from newly generated training outputs. |
| `02_code/compute_numbers.py` | Computes table and figure statistics from saved prediction records. |
| `02_code/config.py` | Defines dataset periods, random seeds, model parameters, and output paths. |
| `02_code/evaluate.py` | Implements classification metrics, company-cluster resampling, and paired tests. |
| `02_code/make_tables.py` | Generates numerical tables from stored model evaluation data. |
| `02_code/models.py` | Defines neural feature extractors, baseline networks, and the SVM–KNN cascade. |
| `02_code/reconcile_review.py` | Checks agreement among per-window predictions, per-seed metrics, and loss files. |
| `02_code/train.py` | Trains specified models and saves predictions, validation histories, and model outputs. |

### 03_figures — Plotting

| File | Description |
|---|---|
| `03_figures/make_figures.py` | Generates the study figures from saved predictions and summary tables. |
| `03_figures/plot_retrained.py` | Generates figures using newly produced training outputs. |

### 04_outputs — Diagnostic files

| File | Description |
|---|---|
| `04_outputs/diagnostics/xgboost_seed45_diff.csv` | Lists window-level differences between two XGBoost seed predictions. |
| `04_outputs/diagnostics/xgboost_seed45_summary.csv` | Summarizes the differences between the corresponding XGBoost seed predictions. |

### 04_outputs — Figures

| File | Description |
|---|---|
| `04_outputs/figures/fig3_performance.pdf` | Stores Figure 3, covering classification performance and KNN correction, as a PDF. |
| `04_outputs/figures/fig3_performance.png` | Stores Figure 3, covering classification performance and KNN correction, as a PNG image. |
| `04_outputs/figures/fig4_comparison.pdf` | Stores Figure 4, covering model comparisons, ablation, and threshold analysis, as a PDF. |
| `04_outputs/figures/fig4_comparison.png` | Stores Figure 4, covering model comparisons, ablation, and threshold analysis, as a PNG image. |
| `04_outputs/figures/figA1_sample.pdf` | Stores Figure A1, showing sample windows and test-set class composition, as a PDF. |
| `04_outputs/figures/figA1_sample.png` | Stores Figure A1, showing sample windows and test-set class composition, as a PNG image. |
| `04_outputs/figures/figA2_training.pdf` | Stores Figure A2, showing validation-loss trajectories, as a PDF. |
| `04_outputs/figures/figA2_training.png` | Stores Figure A2, showing validation-loss trajectories, as a PNG image. |

### 04_outputs — Training logs

| File | Description |
|---|---|
| `04_outputs/logs/training_log_all_30.csv` | Combines the epoch-level loss records for six network configurations across five seeds. |
| `04_outputs/logs/training_log_bilstm_seed1.csv` | Records epoch-level training and validation losses for the Bi-LSTM configuration under random seed 1. |
| `04_outputs/logs/training_log_bilstm_seed2.csv` | Records epoch-level training and validation losses for the Bi-LSTM configuration under random seed 2. |
| `04_outputs/logs/training_log_bilstm_seed3.csv` | Records epoch-level training and validation losses for the Bi-LSTM configuration under random seed 3. |
| `04_outputs/logs/training_log_bilstm_seed4.csv` | Records epoch-level training and validation losses for the Bi-LSTM configuration under random seed 4. |
| `04_outputs/logs/training_log_bilstm_seed5.csv` | Records epoch-level training and validation losses for the Bi-LSTM configuration under random seed 5. |
| `04_outputs/logs/training_log_fc_bilstm_seed1.csv` | Records epoch-level training and validation losses for the fuzzy cloud Bi-LSTM configuration without attention under random seed 1. |
| `04_outputs/logs/training_log_fc_bilstm_seed2.csv` | Records epoch-level training and validation losses for the fuzzy cloud Bi-LSTM configuration without attention under random seed 2. |
| `04_outputs/logs/training_log_fc_bilstm_seed3.csv` | Records epoch-level training and validation losses for the fuzzy cloud Bi-LSTM configuration without attention under random seed 3. |
| `04_outputs/logs/training_log_fc_bilstm_seed4.csv` | Records epoch-level training and validation losses for the fuzzy cloud Bi-LSTM configuration without attention under random seed 4. |
| `04_outputs/logs/training_log_fc_bilstm_seed5.csv` | Records epoch-level training and validation losses for the fuzzy cloud Bi-LSTM configuration without attention under random seed 5. |
| `04_outputs/logs/training_log_fca_bilstm_seed1.csv` | Records epoch-level training and validation losses for the full fuzzy cloud attention Bi-LSTM configuration under random seed 1. |
| `04_outputs/logs/training_log_fca_bilstm_seed2.csv` | Records epoch-level training and validation losses for the full fuzzy cloud attention Bi-LSTM configuration under random seed 2. |
| `04_outputs/logs/training_log_fca_bilstm_seed3.csv` | Records epoch-level training and validation losses for the full fuzzy cloud attention Bi-LSTM configuration under random seed 3. |
| `04_outputs/logs/training_log_fca_bilstm_seed4.csv` | Records epoch-level training and validation losses for the full fuzzy cloud attention Bi-LSTM configuration under random seed 4. |
| `04_outputs/logs/training_log_fca_bilstm_seed5.csv` | Records epoch-level training and validation losses for the full fuzzy cloud attention Bi-LSTM configuration under random seed 5. |
| `04_outputs/logs/training_log_gru_seed1.csv` | Records epoch-level training and validation losses for the GRU configuration under random seed 1. |
| `04_outputs/logs/training_log_gru_seed2.csv` | Records epoch-level training and validation losses for the GRU configuration under random seed 2. |
| `04_outputs/logs/training_log_gru_seed3.csv` | Records epoch-level training and validation losses for the GRU configuration under random seed 3. |
| `04_outputs/logs/training_log_gru_seed4.csv` | Records epoch-level training and validation losses for the GRU configuration under random seed 4. |
| `04_outputs/logs/training_log_gru_seed5.csv` | Records epoch-level training and validation losses for the GRU configuration under random seed 5. |
| `04_outputs/logs/training_log_lstm_seed1.csv` | Records epoch-level training and validation losses for the standard LSTM configuration under random seed 1. |
| `04_outputs/logs/training_log_lstm_seed2.csv` | Records epoch-level training and validation losses for the standard LSTM configuration under random seed 2. |
| `04_outputs/logs/training_log_lstm_seed3.csv` | Records epoch-level training and validation losses for the standard LSTM configuration under random seed 3. |
| `04_outputs/logs/training_log_lstm_seed4.csv` | Records epoch-level training and validation losses for the standard LSTM configuration under random seed 4. |
| `04_outputs/logs/training_log_lstm_seed5.csv` | Records epoch-level training and validation losses for the standard LSTM configuration under random seed 5. |
| `04_outputs/logs/training_log_tcn_seed1.csv` | Records epoch-level training and validation losses for the TCN configuration under random seed 1. |
| `04_outputs/logs/training_log_tcn_seed2.csv` | Records epoch-level training and validation losses for the TCN configuration under random seed 2. |
| `04_outputs/logs/training_log_tcn_seed3.csv` | Records epoch-level training and validation losses for the TCN configuration under random seed 3. |
| `04_outputs/logs/training_log_tcn_seed4.csv` | Records epoch-level training and validation losses for the TCN configuration under random seed 4. |
| `04_outputs/logs/training_log_tcn_seed5.csv` | Records epoch-level training and validation losses for the TCN configuration under random seed 5. |

### 04_outputs — Predictions and statistics

| File | Description |
|---|---|
| `04_outputs/numbers.json` | Stores selected statistics computed from the saved predictions. |
| `04_outputs/predictions/confusion_matrix_seed3.csv` | Stores the risk-class confusion matrix for random seed 3. |
| `04_outputs/predictions/knn_subset.csv` | Stores initial and KNN-adjusted decisions for routed low-confidence windows. |
| `04_outputs/predictions/mcnemar_bootstrap.csv` | Stores paired-test and company-cluster resampling statistics. |
| `04_outputs/predictions/metrics_summary.csv` | Stores evaluation metrics indexed by model and random seed. |
| `04_outputs/predictions/paired_window_seed3_1102.csv` | Stores paired model predictions for 1,102 test windows under random seed 3. |
| `04_outputs/predictions/predictions_all.csv` | Stores test-window class predictions and probabilities for the recorded models and seeds. |
| `04_outputs/predictions/theta_sensitivity.csv` | Stores the confidence-threshold scan and its window-count and accuracy fields. |
| `04_outputs/predictions/training_log.csv` | Stores epoch-level training and validation loss records used in summary plots. |
| `04_outputs/predictions/window_labels_expected.csv` | Stores window-level risk labels and corresponding event-category fields. |
| `04_outputs/predictions/window_split.csv` | Stores window identifiers, risk classes, and dataset partitions. |
| `04_outputs/tables/confusion_matrix_median_seed.csv` | Stores the risk-class confusion matrix for the selected median seed. |
| `04_outputs/tables/knn_contribution_two_scopes.csv` | Summarizes the KNN contribution under two evaluation scopes. |
| `04_outputs/tables/paired_counts_by_seed.csv` | Stores paired correct-versus-incorrect prediction counts for each seed. |
| `04_outputs/tables/statistics_unrounded.csv` | Stores unrounded means, sample standard deviations, and ranges. |
| `04_outputs/tables/table2_sample.csv` | Stores the company and sliding-window counts corresponding to Table 2. |
| `04_outputs/tables/table4_comparison.csv` | Stores the model-comparison metrics corresponding to Table 4. |
| `04_outputs/tables/table5_ablation.csv` | Stores the ablation configurations and metrics corresponding to Table 5. |
| `04_outputs/tables/table6_seed_reconciliation.csv` | Collects per-seed accuracy, recall, and paired prediction counts. |
| `04_outputs/tables/tableA1_events_by_year.csv` | Stores the annual event-category counts for Appendix Table A1. |
| `04_outputs/tables/tableA2_best_epoch.csv` | Stores the minimum-loss epoch and associated summaries by network and seed. |
| `04_outputs/tables/tableA3_theta.csv` | Stores threshold-sensitivity values corresponding to Appendix Table A3. |
| `04_outputs/tables/theta_check.csv` | Stores values from a separate confidence-threshold calculation. |
| `04_outputs/tables/theta_reconciliation.csv` | Stores the correspondence between two threshold-calculation datasets. |

### 05_docs — Environment and indexes

| File | Description |
|---|---|
| `05_docs/file_integrity_sha256.csv` | Indexes file sizes and SHA-256 checksums for the package. |
| `05_docs/original_snapshots/commercial_paper_overdue_original.csv` | Preserves a source snapshot of commercial acceptance bill overdue events. |
| `05_docs/original_snapshots/extraction_manifest_original.csv` | Preserves a source snapshot of the data extraction manifest. |
| `05_docs/original_snapshots/mcnemar_bootstrap_original.csv` | Preserves a source snapshot of the paired-test summary. |
| `05_docs/original_snapshots/st_status_original.csv` | Preserves a source snapshot of special treatment event records. |
| `05_docs/output_manifest.csv` | Maps output files to generated tables, figures, and scripts. |
| `05_docs/requirements.txt` | Lists the Python dependency requirements and version constraints. |
| `05_docs/session_info.txt` | Records runtime and session metadata for the project. |
| `05_docs/statistical_reconciliation.json` | Stores a structured summary of numerical cross-checks. |
| `05_docs/training_logs_sha256_index.csv` | Indexes epoch logs by seed and includes file hashes. |
| `05_docs/待核验证据清单.csv` | Lists document-source reference items and their categorization. |

### Tests

| File | Description |
|---|---|
| `tests/test_implementation.py` | Tests cloud features, model dimensions, temporal handling, and numerical calculations. |

All file paths are relative to the project root; filenames are preserved exactly as they appear in the supplied archive.
