# Parkinson’s Voice Regression: Predicting Motor UPDRS Scores

Can a regression model use voice measurements to estimate a motor UPDRS score for a patient it has never seen during training? This project compares Ridge regression and Extra Trees against a simple baseline: predict the average score of the training recordings for every new recording.

**Main finding:** The training-mean baseline had the lowest mean validation error. Selecting four features instead of using all 16 slightly reduced the average error of Extra Trees, but the improvement occurred in only one of three folds. At the patient level, the four-feature model performed better than the baseline for 10 of 33 validation patients.

## Dataset and target

The [UCI Parkinsons Telemonitoring dataset](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) contains **5,875 voice recordings from 42 people with Parkinson’s disease**. Each row represents one recording. The notebooks use acoustic measurements extracted from the recordings rather than raw audio.

| Component | Use in this project |
|---|---|
| Inputs | 16 voice features: jitter and shimmer measures, NHR, HNR, RPDE, DFA and PPE |
| Target | `motor_UPDRS`: the linearly interpolated motor score on the Unified Parkinson’s Disease Rating Scale |
| Patient ID | `subject#`: used to group recordings for evaluation, never as a predictor |
| Excluded predictors | Age, sex, test time and total UPDRS |

The goal is to estimate a motor score for a recording. The dataset contains people already diagnosed with Parkinson’s disease, so this experiment does not evaluate diagnosis. Because the target is interpolated, each recording does not necessarily correspond to a separate clinical assessment.

## Evaluation design

One patient contributes many recordings. Splitting individual rows at random could put the same patient in both training and evaluation. To measure performance on **unseen patients**, all recordings from a patient stayed on the same side of each split.

1. **Initial holdout:** `GroupShuffleSplit` with `test_size=0.20` and `random_state=42` separated 33 training patients from 9 test patients. Ridge and the baseline were evaluated on the 9 held-out patients.
2. **Three-fold cross-validation:** Only the 33 training patients entered `GroupKFold`. Each round used 22 patients for training and 11 for validation. Every patient appeared in validation exactly once, giving 4,640 validation recordings across the three folds.
3. **Patient-level analysis:** For each of the 33 patients, the baseline and four-feature Extra Trees predictions were compared on exactly the same validation recordings.

The same folds were used across model comparisons. The baseline mean, Ridge feature scaling and feature selection were calculated using only the training recordings in each round.

## Models and feature selection

| Method | Implementation |
|---|---|
| Training-mean baseline | Predicts the mean `motor_UPDRS` of the current training recordings for every validation recording |
| Ridge regression | `StandardScaler` followed by `Ridge(alpha=1.0)`, using all 16 voice features |
| Extra Trees: all features | `ExtraTreesRegressor` with 100 trees and `random_state=42`, using all 16 features |
| Extra Trees: selected features | The same regressor and settings, using four features selected within each training fold |

For feature selection, each voice feature was represented by its **Pearson correlations with all 16 features**. K-Means grouped these correlation profiles into four clusters. Within each cluster, the feature with the highest mutual information with the training target was selected. This produced four features per fold; their names could differ between folds.

## Results

### 1. Initial holdout: 9 test patients

| Method | MAE | RMSE | R² |
|---|---:|---:|---:|
| Training-mean baseline | **6.9069** | **7.8154** | **−0.0476** |
| Ridge regression | 7.0605 | 8.7180 | −0.3035 |

MAE and RMSE are reported in motor UPDRS points; lower values indicate smaller prediction errors. On this split, Ridge did not improve on the training-mean baseline. **The Extra Trees results below are validation results, not results from these 9 test patients.**

### 2. Model comparison: three patient-wise validation folds

| Method | Fold 1 MAE | Fold 2 MAE | Fold 3 MAE | Mean fold MAE |
|---|---:|---:|---:|---:|
| Training-mean baseline | **6.9911** | 8.1967 | **6.6242** | **7.2706** |
| Ridge regression | 7.0022 | **7.7026** | 9.8480 | 8.1843 |
| Extra Trees: 16 features | 7.5672 | 8.6373 | 8.6090 | 8.2712 |
| Extra Trees: 4 selected features | 8.0865 | 8.6399 | 8.0324 | 8.2529 |

Each fold MAE averages the absolute errors of that fold’s validation recordings. The final column gives equal weight to each of the three folds.

Reducing the inputs from 16 features to four lowered the mean Extra Trees MAE from **8.2712 to 8.2529 points**. The fold results explain this small difference: error increased in Fold 1, changed little in Fold 2 and decreased in Fold 3. Feature selection therefore did not produce a consistent improvement.

### 3. Patient-level comparison

| Method with lower patient-level MAE | Patients | Size of the difference |
|---|---:|---|
| Four-feature Extra Trees | 10 | Median reduction of 0.7281 points relative to baseline |
| Training-mean baseline | 23 | Median increase of 1.4782 points for Extra Trees relative to baseline |
| Equal MAE | 0 | — |

Each patient counts once in this comparison, regardless of recording count. For example, across **patient 4’s 137 recordings**, Extra Trees had a lower MAE than the baseline (**6.4616 vs. 7.1945**), even though the baseline performed better across Fold 1 overall. A fold average can therefore hide differences between patients.

## Notebook guide

Run the notebooks in order. Later notebooks use result files created by earlier ones.

| Notebook | Question or task |
|---|---|
| [01 — Data understanding](notebooks/01_data_understanding.ipynb) | How many patients and recordings are present, and how are motor scores distributed? |
| [02 — Ridge baseline](notebooks/02_ridge_baseline.ipynb) | Does Ridge improve on the training mean for the 9 held-out patients? |
| [03 — Patient-wise cross-validation](notebooks/03_patient_wise_cross_validation.ipynb) | How does the Ridge comparison change across different validation patients? |
| [04 — Extra Trees regression](notebooks/04_extra_trees_regression.ipynb) | Does changing the model while keeping the features and folds fixed reduce error? |
| [05 — Feature selection](notebooks/05_feature_selection_extra_trees.ipynb) | Does selecting four voice features improve Extra Trees on the same folds? |
| [06 — Patient-wise error analysis](notebooks/06_patient_wise_error_analysis.ipynb) | For which patients does Extra Trees outperform the baseline, and by how much? |

## Run locally

Clone the repository:

```bash
git clone https://github.com/WaqasGurmani/parkinsons-voice-updrs-regression.git
cd parkinsons-voice-updrs-regression
```

Install the Python packages used for the analysis:

```bash
python -m pip install jupyterlab pandas numpy matplotlib scikit-learn
```

Download and extract the [UCI dataset](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring). Create a `data` folder in the repository root and place the data file at:

```text
data/parkinsons_updrs.data
```

Launch JupyterLab from the repository root:

```bash
jupyter lab
```

Open `notebooks/01_data_understanding.ipynb` and run the notebooks sequentially. The notebooks use relative paths such as `../data/` and `../results/`. Raw data are kept locally; result tables and plots are included in the repository.

## Saved outputs

| Output | Contents |
|---|---|
| [Holdout comparison](results/baseline_vs_ridge.csv) | Baseline and Ridge metrics for the 9 test patients |
| [Fold-wise model comparison](results/patient_wise_model_comparison.csv) | Baseline, Ridge and 16-feature Extra Trees MAE on the same folds |
| [Feature selection comparison](results/feature_selection_fold_comparison.csv) | Selected features and change in Extra Trees MAE for each fold |
| [Patient-wise error comparison](results/patient_wise_error_comparison.csv) | Recording count, baseline MAE, Extra Trees MAE and error change for each validation patient |
| [Plots](plots/) | Motor score distribution and Ridge predictions against reference scores |

## Limitations

- The 4,640 validation recordings came from **33 people**. Repeated recordings allow detailed analysis within patients but do not replace evaluation on more independent patients.
- The model comparison used one three-fold partition of one dataset. It does not establish that the baseline will outperform voice-based models in another cohort.
- A lower patient-level MAE describes a difference in prediction error. It does not explain why one method performed better for that patient or establish clinical usefulness.
- The feature selection experiment tested one configuration of a published idea. It did not reproduce the complete published pipeline.

## References

- **Dataset and original study:** Tsanas, A., Little, M. A., McSharry, P. E., & Ramig, L. O. (2010). *Accurate Telemonitoring of Parkinson’s Disease Progression by Noninvasive Speech Tests*. See the [UCI dataset record and original study details](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring).
- **Feature selection inspiration:** Essam et al. (2026). [Adaptive regression model for Parkinson’s disease diagnosis from speech signals using Box-Cox-based clustering and extremely randomization](https://doi.org/10.1038/s41598-026-49065-2). *Scientific Reports*.

**Author:** [Muhammad Waqas](https://github.com/WaqasGurmani) · Undergraduate BS IT student