# Chess Game Outcome and Length Prediction (Statistical Learning Final Assignment)

## Overview

This repository contains a Jupyter notebook (`analysis.ipynb`) for a Statistical Learning final assignment. The project predicts:
- Classification: the winner of a chess game (white, black, draw)
- Regression: the game length (number of turns)

The analysis is framed as conditional on a known opening. That is, predictions are made using information available before or very early in the game (e.g., ratings, time control, opening category/ply). All modeling follows cross-validation hygiene and avoids target leakage.

## Problem Framing

- Conditional-on-opening winner prediction: Opening information is assumed known (e.g., declared or observed in the first moves), alongside pre-game features like player ratings and time control.
- No post-game features (e.g., victory status) are used in supervised models.
- Opening-derived features (e.g., `opening_category`, interactions, popularity) are allowed; outcome-derived aggregates/clusters are not used as features in supervised models.

## Data

- Source: Lichess game dataset (Kaggle). Columns include players’ ratings, openings, game outcome (`winner`), `victory_status`, time controls, number of `turns`, and more.
- File: `games.csv` in the repository root.

## Repository Structure

- `analysis.ipynb`: Main analysis notebook
- `games.csv`: Input dataset
- `Report/`: Markdown and HTML report (optional)
- `Assignment context/`: Assignment description and learning outcomes

## Environment and Setup

The notebook runs with Python 3.12 (see notebook metadata). Install core dependencies:

```bash
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

Optional (recommended) packages:
```bash
pip install imbalanced-learn xgboost
```

macOS + XGBoost note:
- If you see an OpenMP error, install libomp and then reinstall xgboost:
```bash
brew install libomp
pip install --upgrade --force-reinstall xgboost
```

## How to Run

1. Launch Jupyter and open `analysis.ipynb`.
2. Run all cells from top to bottom. The notebook is linear and designed for “Run All”.
3. The final sections include a “Results Summary” cell and concise Discussion/Conclusion.

## Methodology (What’s in the Notebook)

### 1) EDA and Feature Engineering
- Rating-based features: `rating_difference`, `abs_rating_difference`, `avg_rating`, etc.
- Time control parsing from `increment_code` into minutes/seconds and total time per player.
- Opening features: `opening_category` (sanitized opening name), `opening_ply`.
- Interaction and popularity signals: rating × opening, rating × time, opening frequency/rarity.
- Strict leakage controls: no outcome-based variables (e.g., `victory_status`) in supervised feature sets.

### 2) Unsupervised Learning (Exploratory)
- PCA for dimensionality insight and visualization.
- Clustering (K-Means and hierarchical) to explore patterns across opening categories.
- Important: Clusters that embed outcome statistics are not used as supervised features to avoid leakage.

### 3) Supervised Learning: Classification (Winner)
- Baseline: Majority-class (`DummyClassifier`) with cross-validated balanced accuracy.
- Logistic Regression (pipeline): `StandardScaler` + (if available) `SMOTE` + classifier, all inside cross-validation.
- Random Forest (pipeline): (if available) `SMOTE` + `RandomForestClassifier` inside cross-validation; also a tuned variant via `GridSearchCV`.
- XGBoost (optional): If available, tuned in a CV-safe pipeline, with target label encoding for multi-class.
- Metrics: Accuracy, Balanced Accuracy (primary, due to imbalance), weighted F1. Cross-validation scores reported (mean ± std where applicable).

### 4) Supervised Learning: Regression (Turns)
- Models: Linear Regression, Ridge, Lasso, Random Forest Regressor.
- Enhancements: Log-transformed target Linear Regression (predictions back-transformed), Poisson Regression for count data.
- Metrics: RMSE (primary), MAE, R².
- A consolidated comparison table ranks models by test RMSE and visualizes comparisons.

### 5) Cross-Validation Hygiene and Imbalance Handling
- All preprocessing (e.g., `StandardScaler`) and resampling (`SMOTE`) occur inside CV pipelines, fit on training folds only.
- No information from test folds leaks into training via global fit/transform.
- Class imbalance is handled via either `SMOTE` (preferred) or `class_weight` fallback when `imbalanced-learn` is unavailable.

## Results and Reporting

The notebook includes:
- A majority baseline for classification and a consolidated comparison (with CV metrics) for all classification models.
- An extended regression comparison including log-transform and Poisson models.
- An auto-generated “RESULTS SUMMARY” cell that prints the top classification model (by balanced accuracy) and best regression model (by RMSE), including improvement over baseline.
- Concise Discussion and Conclusion aligned with the conditional-on-opening framing.

Tip: When presenting results, emphasize balanced accuracy and per-class behavior (notably the minority “draw” class), and describe the rationale for conditional framing and leakage prevention.

## Reproducibility

- Fixed random seeds (`random_state=42`) for splits and models wherever applicable.
- Stratified train-test split for classification.
- 5-fold (`StratifiedKFold`) CV used for model evaluation and tuning.
- If you re-run the notebook on a different machine, ensure the same library versions for exact reproducibility.

## Known Caveats

- XGBoost may require OpenMP runtime on macOS (see setup note).
- Opening popularity features are derived from opening categories; they are outcome-free but still reflect corpus frequencies. In high-stakes settings, consider fitting such transforms on training folds only within a pipeline for distributional purity.

## What to Highlight in a Report

- Clear problem statement: conditional-on-opening prediction avoids unrealistic information usage.
- Leak-proof modeling pipelines and appropriate metrics for imbalance.
- Baselines and ablations: show gains vs majority and discuss what each feature family contributes.
- Error analysis: confusion matrix, per-class performance, and which openings are challenging.
- Regression: handling of skew with log-transform and the Poisson alternative; compare RMSE/MAE/R².

## Acknowledgements

- Dataset credit: Lichess games (Kaggle). Please cite appropriately according to the dataset’s terms.


