# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecommunications
customer is likely to churn. The project explores customer data,
preprocesses features, compares classification models, tunes selected
models, and provides a Streamlit interface for making predictions from
customer information.

## Project Goal

Customer churn can reduce recurring revenue. This project aims to
identify customers who may leave so a business can consider retention
actions, such as targeted offers or discounts.

The project emphasizes **recall for the churn class** because missing a
customer who is likely to leave (a false negative) can mean losing an
opportunity to retain them. Precision and F1-score are also considered
because predicting too many non-churning customers as churners can waste
retention resources.

## Dataset

The notebook loads `Telco-Customer-Churn.csv`, the Telco Customer Churn
dataset.

-   **Target:** `Churn` (`Yes` / `No`)
-   **Input features:** customer demographics, subscribed services,
    contract and payment details, tenure, monthly charges, and total
    charges.
-   **Identifier removed:** `customerID`
-   `TotalCharges` is converted to a numeric type. Values that cannot be
    converted become missing, and rows with missing values are removed.

The notebook reports 7,032 rows after cleaning. The target is
imbalanced, so accuracy alone is not sufficient for evaluating the
models.

> Place `Telco-Customer-Churn.csv` in the working directory before
> running the notebook. Confirm the dataset's usage terms and source
> before redistributing it.

## Workflow

1.  **Data inspection and cleaning**
    -   Inspect columns, data types, category distributions, and missing
        values.
    -   Convert `TotalCharges` to numeric.
    -   Remove rows containing missing values.
    -   Drop `customerID`.
    -   Explore numeric distributions and relationships with churn.
2.  **Train/test split**
    -   Use an 80/20 train/test split.
    -   Use stratification to preserve the target distribution.
    -   Set `random_state=42` for reproducibility.
3.  **Preprocessing**
    -   One-hot encode categorical input features.
    -   Standardize numeric features for models that are sensitive to
        feature scale.
    -   Encode the target labels as numeric values.
    -   Fit preprocessing on the training data and apply the fitted
        transformations to the test data.
4.  **Model training and tuning**
    -   Compare Logistic Regression, Decision Tree, Random Forest,
        K-Nearest Neighbors (KNN), Support Vector Classifier (SVC),
        AdaBoost, and XGBoost.
    -   Use `GridSearchCV` with cross-validation to tune selected
        models.
    -   Experiment with soft-voting ensembles.
5.  **Evaluation**
    -   Evaluate models using recall, precision, and F1-score.
    -   Give particular attention to recall for the churn class.
    -   Review the reported results critically and verify the final
        evaluation cells before using the results in a portfolio or
        business decision.
6.  **Deployment**
    -   Save the fitted model and encoder together in `churn_model.pkl`.
    -   Use a Streamlit app to collect customer details and display a
        churn prediction.

## Models and Reported Results

The notebook includes the following comparison table:

  Model                   F1-score   Recall   Precision
  --------------------- ---------- -------- -----------
  Logistic Regression         0.60     0.79        0.48
  Decision Tree               0.61     0.74        0.52
  Random Forest               0.63     0.76        0.53
  KNN                         0.54     0.53        0.54
  SVC                         0.59     0.55        0.63
  AdaBoost                    0.62     0.79        0.51
  XGBoost                     0.62     0.79        0.51

These are the rounded values written in the notebook's comparison table,
not independently re-evaluated results. **Verify the final XGBoost
metrics before treating this table as definitive:** the notebook's
evaluation cell immediately after fitting `best_xgb` calls
`best_ada.predict(x_test)`, which evaluates AdaBoost rather than
XGBoost.

The notebook also evaluates soft-voting ensembles. The reported
AdaBoost + XGBoost voting model has:

-   **F1-score:** 0.6220
-   **Recall:** 0.6952
-   **Precision:** 0.5628

Based on the project's stated priority of finding as many churners as
possible, compare models using recall alongside precision and F1-score,
rather than selecting a model by accuracy alone. The final model saved
by the notebook is `best_xgb`; its final test metrics should be verified
as noted above.

## Streamlit App

The app in `my_app.py` provides dropdowns for categorical features and
numeric inputs for:

-   `tenure`
-   `MonthlyCharges`
-   `TotalCharges`

When the user selects **Predict Churn**, the app loads
`churn_model.pkl`, applies the saved one-hot encoder, and passes the
transformed input to the saved model.

### Run locally

1.  Create and activate a Python virtual environment (recommended).

2.  Install the required libraries:

    ``` bash
    python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib jupyter
    ```

3.  Ensure these files are in the app's working directory:

    -   `my_app.py`
    -   `churn_model.pkl`

4.  Start the app:

    ``` bash
    python -m streamlit run my_app.py
    ```

5.  Open the local URL printed in the terminal by Streamlit (usually
    `http://localhost:8501`).

### Create the model artifact

If `churn_model.pkl` is not available, run `clsf.ipynb` through the
model-saving section to generate it. The notebook expects
`Telco-Customer-Churn.csv` to be available at the path specified in the
notebook.

**Important:** The encoder used in the app must be the same fitted
encoder used during training, and the input feature names, types, and
order must match the model's training data. The app and notebook should
be tested together after any preprocessing changes.

## Project Files

A suggested repository layout is:

``` text
.
├── clsf.ipynb                 # Data exploration, preprocessing, training, and evaluation
├── my_app.py                  # Streamlit prediction interface
├── churn_model.pkl            # Saved encoder and trained model (generated by notebook)
├── Telco-Customer-Churn.csv   # Dataset 
└── README.md                  # Project documentation
```

Only include files that actually exist in your repository. Avoid
committing private data, credentials, or other sensitive files. If the
dataset's license or redistribution terms are unclear, document where it
can be obtained instead of uploading it.

## Technologies

-   Python
-   pandas and NumPy
-   Matplotlib and Seaborn
-   scikit-learn
-   XGBoost
-   Joblib
-   Streamlit
-   Jupyter Notebook

## Possible Next Steps

-   Recheck and standardize the final evaluation for every tuned model
    using the same test split.
-   Add a confusion matrix and classification report for the selected
    model.
-   Tune the prediction threshold using validation data if higher churn
    recall is needed, then evaluate once on the held-out test set.
-   Improve input validation and user-facing error messages in the
    Streamlit app.
-   Add screenshots of the app and document the dataset source.
