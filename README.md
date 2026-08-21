# Job Opportunity Prediction — Identifying Candidates Likely to advance to the next stage of the hiring process.

A Machine learning project that predicts whether a candidate who has completed data science training is likely to advance to the next stage of the hiring process, so recruitment and training resources can be targeted more effectively.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models & Results](#models--results)
- [Feature Importance](#feature-importance)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

## Overview

Companies that fund employee training programs often want to know which trainees are likely to leave for a new job afterward. This project builds and compares several classification models to predict that outcome (`target` column: 1 = looking for a job change, 0 = not looking) using candidate demographic, education, and employment-history data.

The final notebook also produces a ranked shortlist of the top candidates most likely to be looking for a new role, based on predicted probability.

## Dataset
The data consists of two files, `aug_train.csv` and `aug_test.csv`, with the following fields:

| Column | Description |
|---|---|
| `enrollee_id` | Unique candidate ID |
| `city` | City code |
| `city_development_index` | Development index of the candidate's city |
| `gender` | Candidate gender |
| `relevent_experience` | Whether the candidate has relevant work experience |
| `enrolled_university` | Type of university enrollment (if any) |
| `education_level` | Highest education level |
| `major_discipline` | Field of study |
| `experience` | Years of total work experience |
| `company_size` | Size of current employer |
| `company_type` | Type of current employer |
| `last_new_job` | Years since last job change |
| `training_hours` | Hours of training completed |
| `target` | 1 = looking for a job change, 0 = not looking (train set only) |

> **Note:** Add the dataset source/citation here (e.g. the Kaggle dataset it was downloaded from), and confirm the license permits redistribution before committing the CSVs to the repo — otherwise link to the source instead of uploading the raw data.

## Project Workflow

1. **Exploratory Data Analysis** — shape, dtypes, missing-value profiling (`missingno`), duplicate checks, distributions of numerical/categorical columns, outlier detection (IQR method), and target relationships across key features.
2. **Preprocessing** — missing values filled as `"Unknown"`; outliers in `training_hours` capped using the IQR method (bound computed on train, applied to test).
3. **Feature Engineering / Encoding**
   - Ordinal encoding: `education_level`, `relevent_experience`, `experience`, `company_size`, `last_new_job`, `enrolled_university`
   - Frequency encoding: `city`
   - One-hot encoding: `major_discipline`, `gender`, `company_type`
4. **Modeling** — trained and cross-validated (5-fold `StratifiedKFold`) with ROC-AUC as the primary metric:
   - Logistic Regression (scaled, class-balanced)
   - Random Forest (baseline, then tuned via `RandomizedSearchCV`)
   - XGBoost (baseline, then tuned via `RandomizedSearchCV`, with `scale_pos_weight` for class imbalance)
   - Soft-voting Ensemble (Logistic Regression + Random Forest + XGBoost)
5. **Evaluation** — classification reports, confusion matrices, and a master comparison table across models.
6. **Candidate Ranking** — the trained model scores the test set and outputs the top 10 candidates by predicted probability of looking for a job change.

## Models & Results

Cross-validated performance (5-fold, ROC-AUC), sorted best to worst:

| Model | ROC-AUC | Class 1 Recall | Class 1 Precision | Class 1 F1 |
|---|---|---|---|---|
| **Tuned XGBoost** | **0.7903** | 0.683 | 0.584 | 0.6350 |
| Tuned Random Forest | 0.8003 | 0.724 | 0.576 | 0.631 |
| Logistic Regression | 0.7615 | 0.698 | 0.486 | 0.564 |
| Ensemble | 0.7918 | 0.7194 | 0.576 | 0.658 |

Additional models evaluated during experimentation: baseline Random Forest (CV ROC-AUC 0.784), baseline XGBoost (CV ROC-AUC 0.784)  — none outperformed the tuned XGBoost or tuned Random Forest above.

Class 1 (candidates looking for a job change) is the minority class (~25% of the data), so recall/precision/F1 on class 1 are reported alongside ROC-AUC to reflect performance on the class that matters most for this use case.

## Feature Importance

Top predictors from the tuned Random Forest model:

| Feature | Importance |
|---|---|
| `city_development_index` | 0.327 |
| `company_size_encoded` | 0.228 |
| `city_freq` | 0.121 |
| `experience_encoded` | 0.082 |
| `training_hours_capped` | 0.075 |
| `education_level_encoded` | 0.066 |
| `last_new_job_encoded` | 0.040 |
| `enrolled_university_encoded` | 0.035 |
| `relevent_experience_encoded` | 0.027 |

The city's development index and current company size are by far the strongest signals of whether a candidate is looking for a new role.


## Repository Structure
```
Main
│
├── datasets/
│   ├── aug_train.csv
│   ├── aug_test.csv
│   ├── processed-aug_train.csv
│   ├── processed-aug_test.csv
│   └── final_candidate_predictions.csv
│
├── Source-Code/
│   └── Source Code.ipynb
│
├── trained-models/
│   ├── best_xgb_model.pkl        
│   └── best_rf_model.pkl     
│
├── Predicting_Candidate_Job_Change_Likelihood.pdf
├── Project-Presentation.pptx
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting Started

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

Place `aug_train.csv` and `aug_test.csv` in a `data/` folder (or update the file paths in the notebook if you keep them elsewhere — the original notebook reads them from Google Drive when run in Colab).

## Usage

Run the notebook top to bottom in Jupyter or Google Colab. The final cells:
- Train and tune each model
- Compare all models on cross-validated ROC-AUC, precision, recall, and F1
- Export the trained model as a `.pkl` file for reuse
- Score the test set and print the top 10 candidates most likely to be looking for a job change

To reuse the exported model elsewhere:

```python
import pickle

with open('models/best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

## Tech Stack

- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn, missingno
- **Modeling:** scikit-learn, XGBoost
- **Environment:** Google Colab

## Future Improvements

- Address class imbalance further (e.g. SMOTE) and compare against the current `class_weight`/`scale_pos_weight` approach
- Try LightGBM and a neural network (Keras/TensorFlow) as additional candidate models
- Hyperparameter-tune the ensemble's component weights instead of using unweighted soft voting
- Add SHAP values for more interpretable, per-candidate explanations
- Wrap preprocessing + model into a single `sklearn.Pipeline` for simpler deployment

---

*Add license and contributor information here.*
