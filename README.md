# Employee Attrition Early Warning with Responsible Machine Learning

Employee turnover can be expensive and disruptive, but a prediction is only useful when its uncertainty and intended use are clear. This project asks whether employee attrition can be predicted meaningfully from workplace data and how the results could support earlier, more helpful retention conversations.

## At a Glance

| Measure | Verified result |
|---|---:|
| Employee records | 1,470 |
| Employees who left | 237 (16.1%) |
| Training / test records | 1,176 / 294 |
| Selected model | Logistic Regression |
| Test ROC-AUC | 0.802 |
| Test precision-recall AUC | 0.557 |
| Departures identified | 30 of 47 (63.8% recall) |
| Precision | 34.9% |

The model ranked attrition risk better than chance and found almost two-thirds of the departures in the untouched test set. It also produced 56 false alerts and missed 17 departures. That tradeoff is central to the project's conclusion: this can be a screening aid, but it is not an automated employment decision system.

## Business Question

Can the available employee and workplace characteristics identify attrition risk early enough to support constructive retention efforts?

## Data and Quality Checks

The analysis uses the fictional **IBM HR Analytics Employee Attrition and Performance** dataset distributed through Kaggle. It contains demographic, job, compensation, satisfaction, travel, overtime, and tenure fields.

I inspected the data before analysis and found no missing values or duplicate records. I preserved all 1,470 observations and removed only four fields:

- `EmployeeCount`, `Over18`, and `StandardHours` because each was constant.
- `EmployeeNumber` because it was an identifier, not a meaningful predictor.

The cleaned data contained 30 predictors and one target. I used a stratified 80/20 split and verified that no record appeared in both sets. Encoding, imputation, and scaling were placed inside the modeling pipeline so that each transformation was learned from training data only.

## Modeling Approach

I compared Logistic Regression and Random Forest with a non-predictive baseline using five-fold stratified cross-validation. Model selection used only training data and considered ROC-AUC, precision-recall AUC, balanced accuracy, precision, recall, and F1 score. The untouched test set was used once for final evaluation.

Logistic Regression produced the strongest validation results:

- Mean validation ROC-AUC: **0.827**
- Mean validation precision-recall AUC: **0.605**
- Mean validation recall: **73.7%**, compared with **40.0%** for Random Forest

The imbalance matters. Predicting that every employee would stay would be 84.0% accurate while identifying no departures. For that reason, accuracy was not treated as the main success measure.

## Findings

On the test set, Logistic Regression achieved a ROC-AUC of 0.802 and a precision-recall AUC of 0.557. At the default 0.50 threshold, it identified 30 of the 47 employees who left. Its 63.8% recall had an approximate 95% bootstrap confidence interval of 48.9% to 76.6%, reflecting the limited number of departure cases.

Overtime carried the strongest predictive signal in the permutation analysis. Shuffling `OverTime` reduced test ROC-AUC by approximately 0.105, more than four times the decrease associated with the next group of predictors. Marital status, years since last promotion, number of companies worked, business travel, and job satisfaction also contributed, but their importance was smaller or less stable.

These are predictive associations, not evidence that any factor causes an employee to leave.

## Responsible Use

I would use the model to support voluntary check-ins, workload reviews, career-development conversations, and a closer review of overtime or travel expectations. A risk score should never be used to penalize an employee, restrict opportunities, or make termination, promotion, or compensation decisions.

Before practical use, the model should be validated with current organizational data, evaluated across departments and demographic groups, calibrated to the organization's capacity to respond, and monitored for performance drift.

## Repository Structure

```text
.
├── employee_attrition_analysis.ipynb  # Executed analysis and interpretation
├── data/
│   └── README.md                      # Data source and download instructions
├── requirements.txt                   # Reproducible Python dependencies
├── LICENSE                            # License for original code and writing
└── README.md
```

## Reproduce the Analysis

1. Clone this repository.
2. Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
3. Place the CSV in the `data` folder without renaming it.
4. Create and activate a Python environment.
5. Install the dependencies and run the notebook from top to bottom.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab employee_attrition_analysis.ipynb
```

The repository does not redistribute the source CSV. Kaggle identifies the dataset as a fictional dataset and lists an open database/content license; the source page remains the authoritative location for its terms and attribution.

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and Jupyter

## Data Source

IBM. (n.d.). *IBM HR analytics employee attrition and performance* [Data set]. Kaggle. https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
