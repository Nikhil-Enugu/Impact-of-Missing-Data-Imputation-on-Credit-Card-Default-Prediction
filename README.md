# Impact of Missing Data Imputation on Credit Card Default Prediction

## Author 

Name: Nikhil Enugu

Roll No: DA25M010

---

This project investigates how different techniques for handling missing data impact the performance of a machine learning model. Using the UCI Credit Card Default Clients Dataset, we explore a real-world scenario where feature columns have missing values.

The primary goal is to compare four distinct strategies for handling this missing data and to evaluate their effect on a Logistic Regression classifier built to predict credit card default.

- **Dataset:** UCI Credit Card Default Clients
- **Model:** Logistic Regression
- **Target Variable:** `default.payment.next.month`

---

## 📈 Project Workflow

The analysis is structured in three parts: data imputation, model training, and comparative analysis.

### 1. Data Preprocessing and Imputation

1.  **Data Loading:** The `UCI_Credit_Card.csv` dataset (30,000 samples) is loaded.
2.  **MAR Simulation:** To simulate a real-world scenario, **5% Missing At Random (MAR)** values (1,500 samples each) are artificially introduced into three numerical columns: `AGE`, `BILL_AMT1`, and `BILL_AMT2`.
3.  **Handling Strategies:** Four new datasets are created based on different handling methods:
    * **Dataset A (Median Imputation):** Missing values in all three columns (`AGE`, `BILL_AMT1`, `BILL_AMT2`) are filled with their respective **medians**. This method is robust to outliers. (Total samples: 30,000)
    * **Dataset B (Linear Regression Imputation):** Missing `AGE` values are predicted using a **Linear Regression** model trained on other complete features. *Note: Rows with remaining missing values in `BILL_AMT1` and `BILL_AMT2` are subsequently dropped.* (Total samples: 27,074)
    * **Dataset C (Decision Tree Imputation):** Missing `AGE` values are predicted using a non-linear **Decision Tree Regressor**. *Note: Rows with remaining missing values in `BILL_AMT1` and `BILL_AMT2` are subsequently dropped.* (Total samples: 27,074)
    * **Dataset D (Listwise Deletion):** All rows containing *any* missing values in `AGE`, `BILL_AMT1`, or `BILL_AMT2` are dropped. This is the baseline "do-nothing" approach. (Total samples: 25,728)

---

### 2. Model Training and Assessment

1.  **Train/Test Split:** Each of the four datasets (A, B, C, D) is split into an 80% training set and a 20% testing set. The split is **stratified** to maintain the same proportion of default/non-default cases in both sets.
2.  **Feature Scaling:** `StandardScaler` is fit on each training set and used to transform both the training and test sets.
3.  **Classification:** A **Logistic Regression** classifier is trained on each of the four preprocessed training datasets.
    * `class_weight="balanced"` is used to counteract the high class imbalance in the target variable.
4.  **Evaluation:** Each model's performance is evaluated on its respective test set using a full classification report (Accuracy, Precision, Recall, F1-score).

---

## 📊 Results

The performance of the Logistic Regression model varied significantly based on the imputation strategy used. Since the dataset is imbalanced (more non-defaults than defaults), the **F1-score for the positive class (1)** is the most critical metric for comparing models.

### Comparative Performance Metrics

| Model | Imputation Strategy | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A** | Median Imputation | 0.6795 | 0.3674 | 0.6225 | 0.4621 |
| **B** | **Linear Regression Imputation** | 0.6903 | 0.3829 | **0.6558** | **0.4835** |
| **C** | Decision Tree Imputation | 0.6899 | 0.3824 | 0.6550 | 0.4829 |
| **D** | Listwise Deletion | **0.6935** | 0.3833 | 0.6373 | 0.4787 |

---

## 💡 Analysis and Conclusion

1.  **Listwise Deletion (D) vs. Imputation (A, B, C):**
    * Listwise Deletion (Model D) achieved the **highest accuracy** (0.6935). However, this came at the cost of **deleting over 4,000 samples (~14% of the data)**. This data loss can lead to a model that generalizes poorly, which is reflected in its F1-score being lower than the regression-based imputation models.
    * Simple Median Imputation (Model A) performed the worst on all metrics. This suggests that filling missing values with a simple static number (like the median) distorts the relationships between features and weakens the model's predictive power.

2.  **Linear (B) vs. Non-Linear (C) Imputation:**
    * The **Linear Regression Imputation (Model B)** and **Decision Tree Imputation (Model C)** performed almost identically.
    * Model B (Linear) had a slightly higher F1-score (0.4835 vs 0.4829). This suggests that the relationship between `AGE` and the other predictor variables is **mostly linear**. The added complexity of the non-linear Decision Tree model did not provide any significant benefit and may have slightly overfit the imputation task.

### Recommendation

Based on this analysis, the **Linear Regression Imputation (Model B)** is the recommended strategy for this dataset.

It provides the **best F1-score (0.4835)** and **highest Recall (0.6558)**, indicating it is the most effective at identifying the minority "default" class, which is the primary goal in a risk-assessment scenario. It successfully preserves more of the dataset than listwise deletion while capturing the underlying feature relationships more effectively than simple median imputation.

---
