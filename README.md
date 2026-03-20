# comparing-classifiers
For a bank marketing dataset, build a model and tune it to compare performance of the selected classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines)

## Project Overview
This project aims to optimize the efficiency of direct marketing campaigns for a Portuguese banking institution. By analyzing historical client data, a machine learning pipeline is built to predict whether a customer will subscribe to a term deposit or not.

Here is the data source that has been used for this analysis: [Bank Marketing -UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing)

## The Business Problem
The bank faces a significant class imbalance:
- Success Rate: ~11.3% "Yes".
- Rejection Rate: ~88.7% "No".

The Baseline issue: A model that simply guesses "No" for everyone would be 88.7% accurate but provides zero business value because it identifies zero subscribers. The goal is to beat this baseline by focusing on Recall and F1-score.

## Data & Features
- Dataset: bank-additional-full.csv (41,188 records).
- Target: y (Subscribed? 'yes'/'no' mapped to 1/0).
- Features Used: Focused strictly on bank client info: age, job, marital, education, default, housing, and loan.
- Feature Exclusion: The duration attribute was intentionally excluded to prevent data leakage, as its value is only known after a call is completed.

### Preprocessing:
- Categorical: One-hot encoding for all non-numeric attributes.
- Numerical: StandardScaler is applied to age to ensure mean-centering for sensitive models like SVM and KNN.
- Data Integrity: Remaining features are dropped to ensure model trains on the intended "bank client" demographic profile

## Baseline
Majority-class baseline accuracy (always predict “no”): **~0.887**

## Final Model Performance (Tuned)
After performing hyperparameter tuning using  GridSearchCV the models are re-evaluated and F-1 scores are optimized. While accuracy decreased slightly, the models' ability to identify subscribers (Recall) improved significantly.

| Model | Test Accuracy | Test F1-Score | Train Time | Best Parameters |
| :--- | :---: | :---: | :---: | :--- |
| **Decision Tree** | **0.7013** | **0.2705** | 7.24s | `max_depth: 5` |
| **Logistic Regression** | 0.5895 | 0.2584 | **2.21s** | `C: 1` |
| **SVC (SVM)** | 0.6259 | 0.2581 | 610.25s | `C: 1`, `kernel: 'rbf'` |
| **KNN** | 0.8838 | 0.0736 | 87.77s | `n_neighbors: 9` |

Also, it shows the huge difference in training time taken by different models.

## Models plot before and after tuning:
<img width="1784" height="737" alt="image" src="https://github.com/user-attachments/assets/17943308-c01e-4c72-bd84-e2719668012b" />

## Key findings: 
- It cleary indicates the improvement over baseline. While KNN shows the highest accuracy (0.884), it sits right at the Baseline (88.7%). This confirms that KNN is essentially guessing "No" for every client, providing zero business value despite its high score.

- After fine-tuning F-1 scores improved. Decision Tree showed the most significant improvement, increasing its ability to identify subscribers from 0.233 to 0.271.

- Logistic Regression achieved nearly identical performance to the SVC but trained in 2 seconds vs. the SVC’s 10 minutes, making it the most scalable choice for larger databases.

## Recommendations:
- Target additional features like economic indicators (e.g., euribor3m, emp.var.rate) and campaign context (e.g., month, day_of_week) will help model understand the market conditions better.
- Use Logistic Regression as a cost-effective production alternative. It also works great for bank compliance teams.
- Use full feature set to run GridSearchCV and see if the increased complexity of SVM and other algorithms benefits once more data is available.
- Prioritize precisions and recall to avoid imbalanced data trap.
- Operationalize the Best Model to reach out to new clients and achieve better results.


