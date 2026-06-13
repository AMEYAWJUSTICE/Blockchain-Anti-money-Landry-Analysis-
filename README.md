# AML Transaction Analysis in Elliptic Dataset

## Project Overview
This project focuses on detecting illicit activities, specifically Anti-Money Laundering (AML), within cryptocurrency transactions using the Elliptic Dataset. It employs a comprehensive methodology involving data acquisition, graph construction, advanced feature engineering (including graph-based features), and the application of various machine learning models (Logistic Regression, Random Forest, XGBoost, and Graph Neural Networks).

The ultimate goal is to build an effective classification system that can identify suspicious transactions and provide a user-friendly Streamlit dashboard for real-time analysis and alerting.

## Dataset
The project utilizes the **Elliptic Dataset**, which comprises transaction data from a real-world cryptocurrency network. It includes:
- `elliptic_txs_features.csv`: Anonymized features for each transaction.
- `elliptic_txs_classes.csv`: Labels indicating whether a transaction is legitimate (2), suspicious (1), or unknown.
- `elliptic_txs_edgelist.csv`: Defines the transaction graph, showing the flow of transactions between addresses.

## Key Components & Workflow
1.  **Data Acquisition, EDA, and Preprocessing**:
    - Loading and merging transaction features and class labels.
    - Initial exploratory data analysis (EDA) to understand data distributions, missing values, and class imbalance.
    - Generating synthetic date-based features for time-series analysis.
    - Encoding categorical labels and mapping `txId`s to graph-compatible integer IDs.

2.  **Graph Construction and Feature Engineering**:
    - Building a directed transaction graph using `networkx`.
    - Extracting graph-based features for each wallet, including:
        - **Degree Centrality**: In-degree, Out-degree, Total-degree.
        - **PageRank**: To measure influence within the network.
        - **Clustering Coefficient**: To quantify local connectivity.
        - **Betweenness Centrality**: To identify critical bridging nodes (sampled due to computational intensity).
    - Aggregating transaction-level features (e.g., mean transaction amount, transaction count) per wallet.
    - Creating a custom `risk_exposure_score` and `wallet_aml_label` for each unique wallet.

3.  **Feature Selection and Data Splitting**:
    - Analyzing feature correlations to identify high-impact indicators and remove redundant features.
    - Filtering out 'unknown' labels for model training and evaluation.
    - Splitting the data into training (80%) and testing (20%) sets, ensuring stratification for class balance.

4.  **Model Selection and Training**:
    - Training and evaluating four classification models:
        - Logistic Regression
        - Random Forest Classifier
        - XGBoost Classifier (with hyperparameter tuning using `RandomizedSearchCV`)
        - Graph Neural Network (GNN) using `PyTorch Geometric`
    - Evaluating models based on Accuracy, Precision, Recall, F1-Score, and ROC AUC, with a strong emphasis on **Precision** to minimize false positives.

5.  **Streamlit Dashboard**:
    - Developing an interactive Streamlit application (`app.py`) to visualize the analysis, display transaction details, wallet activity, and AML alerts.
    - Integrating the best-performing model (XGBoost) for real-time predictions.

## Model Performance Summary
Among the evaluated models, the **Tuned XGBoost model** demonstrated the best performance in terms of precision for identifying suspicious transactions, which was the primary objective of this project. It also achieved a strong ROC AUC score.

| Model                 | Accuracy | Precision (Suspicious) | Recall (Suspicious) | F1-Score (Suspicious) | ROC AUC Score |
|:----------------------|:---------|:-----------------------|:--------------------|:----------------------|:--------------|
| Logistic Regression   | 0.2587   | 0.1204                 | 0.9763              | 0.2143                | 0.6131        |
| Random Forest         | 0.6280   | 0.1826                 | 0.7458              | 0.2934                | 0.7551        |
| XGBoost (Tuned)       | **0.6862**   | **0.1974**             | 0.6621              | **0.3042**                | **0.7560**    |
| GNN                   | 0.3410   | 0.1259                 | 0.9028              | 0.2211                | 0.5559        |

**Conclusion**: The **Tuned XGBoost model** is selected as the primary model for the Streamlit dashboard due to its superior precision and balanced overall performance in detecting suspicious transactions.

## Setup and Running the Streamlit App

### Prerequisites
- Python 3.8+
- `pip` (Python package installer)

### Installation
1.  **Clone the repository (if applicable) or ensure you have all necessary files.**
2.  **Install dependencies** using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### Required Files to Run the Dashboard Locally
To run the Streamlit dashboard, ensure you have the following files in the same directory:
-   `app.py` (The Streamlit application code)
-   `xgboost_model.joblib` (The trained XGBoost model)
-   `cleaned_elliptic_txs.csv` (The preprocessed transaction data)
-   `wallet_feature_vectors_selected.csv` (The selected wallet features for prediction)
-   `elliptic_txs_edgelist.csv` (The original edgelist for graph recreation and mapping)
-   `elliptic_txs_features.csv` (The original features file)
-   `elliptic_txs_classes.csv` (The original classes file)

### Running the Dashboard
1.  **Navigate to the directory** containing the above files in your terminal or command prompt.
2.  **Execute the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

3.  Your web browser will automatically open to the Streamlit dashboard (usually at `http://localhost:8501`).

## File Structure
```
.
├── app.py
├── requirements.txt
├── xgboost_model.joblib
├── gnn_model_state_dict.pth
├── logistic_regression_model.joblib
├── random_forest_model.joblib
├── cleaned_elliptic_txs.csv
├── wallet_feature_vectors.csv
├── wallet_feature_vectors_selected.csv
├── elliptic_txs_classes.csv
├── elliptic_txs_edgelist.csv
├── elliptic_txs_features.csv
├── transaction_graph.graphml
└── README.md
