
# Loan Approval Classification Using Machine Learning

## Project Overview
This project applies supervised machine learning models to classify loan applications as **approved or rejected** based on applicant financial and demographic attributes.

Several classification algorithms were implemented and compared to determine which model performs best in predicting loan approval outcomes. The models were evaluated using **Accuracy** and **F1 Score**, and feature importance was analyzed to understand which applicant characteristics most influence approval decisions.

The primary objective was to determine:

- Which machine learning model performs best for loan approval prediction
- Which applicant features most strongly influence approval decisions

---

# Dataset

The dataset used is the **Loan Approval Prediction Dataset** from Kaggle.

The dataset contains **4,269 observations and 13 variables**, representing applicant information and loan attributes.

### Key Variables

| Variable | Description |
|---|---|
| no_of_dependents | Number of dependents |
| education | Graduate or Not Graduate |
| self_employed | Self-employment status |
| income_annum | Annual income |
| loan_amount | Requested loan amount |
| loan_term | Loan duration |
| cibil_score | Credit score |
| residential_assets_value | Residential asset value |
| commercial_assets_value | Commercial asset value |
| luxury_assets_value | Luxury asset value |
| bank_asset_value | Bank asset value |
| loan_status | Target variable (Approved or Rejected) |

---

# Data Exploration

Initial exploratory analysis focused on understanding data completeness, distributions, and relationships between variables.

Key findings:

- No missing values were found in the dataset.
- No duplicate records were detected.
- Loan approvals were moderately imbalanced:

| Status | Percentage |
|---|---|
| Approved | 62.2% |
| Rejected | 37.8% |

Several exploratory visualizations were created including:

- Histograms of continuous variables
- Boxplots for outlier detection
- Count plots for categorical variables
- Correlation heatmaps

Highly correlated variables were removed to reduce redundancy. For example:

- `loan_amount` was strongly correlated with `income_annum`
- `luxury_assets_value` was also strongly correlated with `income_annum`

These features were removed to avoid multicollinearity.

---

# Data Preprocessing

Several preprocessing steps were performed before training models:

- Removed whitespace from column names
- Encoded categorical variables using one-hot encoding
- Converted the target variable:

```python
df["loan_status"] = df["loan_status"].str.strip().map({"Approved":0, "Rejected":1})
```

- Split the dataset into training (80%) and testing (20%) sets
- Applied StandardScaler to numerical features using a Scikit-Learn pipeline

---

# Machine Learning Models

Four classification algorithms were implemented and compared:

| Model | Description |
|---|---|
| CART | Decision tree classifier |
| Support Vector Machine | Margin-based classifier |
| Random Forest | Ensemble of decision trees |
| K-Nearest Neighbors | Instance-based classifier |

Hyperparameters were optimized using GridSearchCV with 5-fold cross-validation.

---

# Model Evaluation

Models were evaluated using:

### Accuracy
Percentage of correctly classified observations.

### F1 Score
Harmonic mean of precision and recall, useful when class imbalance exists.

---

# Model Performance

| Model | Accuracy | F1 Score |
|---|---|---|
| Random Forest | **0.967** | **0.956** |
| CART | 0.957 | 0.943 |
| SVM | 0.934 | 0.915 |
| KNN | 0.911 | 0.883 |

Random Forest achieved the best performance across both metrics.

Model ranking (best → worst):

1. Random Forest
2. CART
3. Support Vector Machine
4. K-Nearest Neighbors

---

# Feature Importance Analysis

Feature importance was calculated using the Random Forest model.

Top predictors of loan approval:

| Feature | Importance |
|---|---|
| cibil_score | **84%** |
| loan_term | 6% |
| residential_assets_value | 2% |
| commercial_assets_value | 2% |
| bank_asset_value | 2% |

All other variables contributed less than 2% to predictions.

### Key Insight

Credit score (**cibil_score**) overwhelmingly dominates the decision process, accounting for **84% of impurity reduction** in the model.

This suggests that **loan approval decisions are largely driven by credit score**, with other financial variables playing a relatively minor role.

---

# Conclusion

The **Random Forest classifier** produced the strongest performance:

- **Accuracy:** 96.7%
- **F1 Score:** 0.956

This model significantly outperformed the other classifiers and provided interpretable feature importance results.

The analysis revealed that **credit score is the dominant predictor of loan approval**, with loan term being the only other moderately influential variable.

---

# Future Improvements

Several enhancements could improve the analysis:

- Expand hyperparameter tuning using RandomizedSearchCV
- Implement additional models such as Gradient Boosting or XGBoost
- Apply further feature reduction techniques
- Explore class imbalance strategies such as SMOTE
- Perform deeper cross-validation model comparison

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Author

**Aidan Daley**  
M.S. Applied Data Science  
Eastern Connecticut State University
