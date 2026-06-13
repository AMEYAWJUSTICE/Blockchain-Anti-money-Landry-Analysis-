import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# --- Configuration and Data Loading (Cached) ---
@st.cache_data
def load_data():
    # Load wallet features (used for model prediction)
    wallets_features_df = pd.read_csv('wallet_feature_vectors_selected.csv')

    # Load all transactions data (for transaction details)
    all_transactions_df = pd.read_csv('cleaned_elliptic_txs.csv')

    # Rename column '8' to 'transaction_amount' for clarity and consistency
    # Ensure to check for string '8' as pandas often reads numeric headers as strings
    if '8' in all_transactions_df.columns and 'transaction_amount' not in all_transactions_df.columns:
        all_transactions_df.rename(columns={'8': 'transaction_amount'}, inplace=True)
    
    # Fix: Ensure column '8' is always treated as 'transaction_amount' before any other operations
    if '8' in all_transactions_df.columns:
        all_transactions_df.rename(columns={'8': 'transaction_amount'}, inplace=True)


    # Load edgelist to re-create wallet_id_mapping
    edges_df = pd.read_csv('elliptic_txs_edgelist.csv')

    # Re-create wallet_id_mapping
    unique_wallet_ids = pd.Series(list(set(edges_df['txId1']).union(set(edges_df['txId2']))))
    wallet_id_mapping = {id: i for i, id in enumerate(unique_wallet_ids)}

    # Add txId_mapped to all_transactions_df
    all_transactions_df['txId_mapped'] = all_transactions_df['txId'].map(wallet_id_mapping)
    all_transactions_df.dropna(subset=['txId_mapped'], inplace=True)
    all_transactions_df['txId_mapped'] = all_transactions_df['txId_mapped'].astype(int)

    # Convert 'date' column to datetime objects
    all_transactions_df['date'] = pd.to_datetime(all_transactions_df[['year', 'month', 'day']])

    return wallets_features_df, all_transactions_df

@st.cache_resource
def load_model():
    # Load the trained XGBoost model
    model_xgb = joblib.load('xgboost_model.joblib')
    return model_xgb

@st.cache_data
def make_predictions(wallets_features_df, _model_xgb):
    X_wallets = wallets_features_df.drop(['txId_mapped', 'wallet_aml_label'], axis=1)

    # Ensure column order matches training data if necessary (though joblib should preserve it)
    # If you encounter issues, save/load X_train.columns and reindex X_wallets

    wallets_features_df['predicted_aml_label_num'] = _model_xgb.predict(X_wallets)
    wallets_features_df['prediction_proba_suspicious'] = _model_xgb.predict_proba(X_wallets)[:, 1]

    # Map numerical labels to readable strings
    label_map = {0: 'Legitimate', 1: 'Suspicious'} 
    wallets_features_df['predicted_aml_label'] = wallets_features_df['predicted_aml_label_num'].map(label_map)

    # True labels (for evaluation or comparison)
    wallets_features_df['true_aml_label'] = wallets_features_df['wallet_aml_label'].map({0: 'Legitimate', 1: 'Suspicious', 2: 'Unknown'})
    return wallets_features_df

# --- Load Data and Model ---
wallets_features_df_raw, all_transactions_df_raw = load_data()
model_xgb = load_model()

# Make predictions and get the full wallets DataFrame with predictions
wallets_df = make_predictions(wallets_features_df_raw.copy(), model_xgb)

# Merge wallet predictions into transaction data
dashboard_data_df = all_transactions_df_raw.merge(
    wallets_df[['txId_mapped', 'predicted_aml_label_num', 'predicted_aml_label', 'prediction_proba_suspicious']],
    on='txId_mapped',
    how='left'
)

# Fill NaN for predicted columns where txId_mapped might not have been in wallets_df (e.g., 'Unknown' wallets from original data that weren't in selected_features)
dashboard_data_df['predicted_aml_label'] = dashboard_data_df['predicted_aml_label'].fillna('Not Predicted')
dashboard_data_df['predicted_aml_label_num'] = dashboard_data_df['predicted_aml_label_num'].fillna(-1) # Use -1 for not predicted
dashboard_data_df['prediction_proba_suspicious'] = dashboard_data_df['prediction_proba_suspicious'].fillna(0)

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="AML Transaction Analysis Dashboard")

st.title("AML Transaction Analysis Dashboard")
st.write("Explore and analyze cryptocurrency transactions for potential money laundering activities using an XGBoost model.")

# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")

# Date Range Filter
min_date = dashboard_data_df['date'].min().date()
max_date = dashboard_data_df['date'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = dashboard_data_df[(dashboard_data_df['date'].dt.date >= start_date) & (dashboard_data_df['date'].dt.date <= end_date)]
else:
    filtered_df = dashboard_data_df.copy()

# Risk Level Filter
risk_level_options = ['All', 'Suspicious', 'Legitimate']
selected_risk_level = st.sidebar.selectbox("Filter by Predicted Risk Level", risk_level_options)

if selected_risk_level != 'All':
    filtered_df = filtered_df[filtered_df['predicted_aml_label'] == selected_risk_level]

# Wallet Search
search_wallet_id = st.sidebar.text_input("Search Wallet ID (Mapped)")

if search_wallet_id:
    try:
        search_wallet_id_int = int(search_wallet_id)
        filtered_df = filtered_df[filtered_df['txId_mapped'] == search_wallet_id_int]
    except ValueError:
        st.sidebar.error("Please enter a valid integer for Wallet ID.")

# --- Main Content Tabs ---
overview_tab, transactions_tab, wallets_tab, analysis_tab, alerts_tab = st.tabs(["Dashboard Overview", "Transactions", "Wallets", "Analysis", "Alerts"])

with overview_tab:
    st.header("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Wallets Analyzed", len(wallets_df['txId_mapped'].unique()))
    with col2:
        st.metric("Total Transactions", len(filtered_df))
    with col3:
        suspicious_wallets_count = wallets_df[wallets_df['predicted_aml_label'] == 'Suspicious'].shape[0]
        st.metric("Predicted Suspicious Wallets", suspicious_wallets_count)
    with col4:
        st.metric("Avg. Suspicious Probability", f"{wallets_df['prediction_proba_suspicious'].mean():.2f}")

    st.subheader("Predicted AML Label Distribution (Wallets)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='predicted_aml_label', data=wallets_df, palette='viridis', ax=ax)
    ax.set_title('Predicted Wallet AML Label Distribution')
    ax.set_xlabel('Predicted AML Label')
    ax.set_ylabel('Number of Wallets')
    st.pyplot(fig)

    st.subheader("Model Performance Summary (XGBoost)")
    # Retrieve metrics from the model training phase or re-calculate for display
    # For simplicity, we'll re-calculate on the full `wallets_df` for known labels
    true_labels = wallets_df[wallets_df['true_aml_label'].isin(['Legitimate', 'Suspicious'])]['wallet_aml_label']
    predicted_labels = wallets_df[wallets_df['true_aml_label'].isin(['Legitimate', 'Suspicious'])]['predicted_aml_label_num']

    if not true_labels.empty and len(true_labels.unique()) > 1:
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
        roc_auc = roc_auc_score(true_labels, wallets_df[wallets_df['true_aml_label'].isin(['Legitimate', 'Suspicious'])]['prediction_proba_suspicious'])

        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision (Suspicious): {precision:.4f}")
        st.write(f"Recall (Suspicious): {recall:.4f}")
        st.write(f"F1-Score (Suspicious): {f1:.4f}")
        st.write(f"ROC AUC Score: {roc_auc:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(true_labels, predicted_labels)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Predicted Legitimate', 'Predicted Suspicious'],
                    yticklabels=['Actual Legitimate', 'Actual Suspicious'])
        ax_cm.set_title('Confusion Matrix (XGBoost)')
        st.pyplot(fig_cm)
    else:
        st.write("Not enough data with known labels to display full model performance metrics.")


with transactions_tab:
    st.header("Transaction Details")
    st.write(f"Displaying {len(filtered_df)} transactions based on current filters.")
    st.dataframe(filtered_df[['txId', 'date', 'transaction_amount', 'class_encoded', 'predicted_aml_label', 'prediction_proba_suspicious']].head(1000))

    st.subheader("Daily Transaction Volume with Predicted Risk")
    daily_volume = filtered_df.groupby('date').agg(
        transaction_count=('txId', 'count'),
        suspicious_count=('predicted_aml_label_num', lambda x: (x == 1).sum())
    ).reset_index()

    fig_daily, ax_daily = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='date', y='transaction_count', data=daily_volume, label='Total Transactions', ax=ax_daily)
    sns.lineplot(x='date', y='suspicious_count', data=daily_volume, label='Suspicious Transactions', color='red', ax=ax_daily)
    ax_daily.set_title('Daily Transaction Volume and Suspicious Activity')
    ax_daily.set_xlabel('Date')
    ax_daily.set_ylabel('Number of Transactions')
    ax_daily.legend()
    st.pyplot(fig_daily)

with wallets_tab:
    st.header("Wallet Analysis")
    st.write(f"Displaying {len(filtered_df['txId_mapped'].unique())} unique wallets based on current filters.")

    # Aggregate filtered_df to wallet level for display, if filters are applied to transactions
    filtered_wallets_df = wallets_df[wallets_df['txId_mapped'].isin(filtered_df['txId_mapped'].unique())]

    st.dataframe(filtered_wallets_df[['txId_mapped', 'total_degree', 'pagerank', 'clustering_coefficient',
                                       'mean_transaction_amount', 'wallet_transaction_count',
                                       'predicted_aml_label', 'prediction_proba_suspicious', 'true_aml_label']].head(1000))

    st.subheader("Distribution of Transaction Count per Wallet")
    fig_wallet_count, ax_wallet_count = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_wallets_df['wallet_transaction_count'], bins=50, kde=True, ax=ax_wallet_count)
    ax_wallet_count.set_title('Distribution of Transaction Count per Wallet')
    ax_wallet_count.set_xlabel('Number of Transactions')
    ax_wallet_count.set_ylabel('Number of Wallets')
    ax_wallet_count.set_yscale('log')
    st.pyplot(fig_wallet_count)

with analysis_tab:
    st.header("Feature Analysis")
    st.subheader("XGBoost Feature Importance")
    # X_wallets is defined inside make_predictions, so it's not directly accessible here.
    # We need to re-create X_wallets or pass it from make_predictions if we want to use its columns.
    # For now, let's assume wallets_features_df_raw still holds the correct feature names.
    X_wallets_for_importance = wallets_features_df_raw.drop(['txId_mapped', 'wallet_aml_label'], axis=1)
    feature_importances = pd.Series(model_xgb.feature_importances_, index=X_wallets_for_importance.columns).sort_values(ascending=False)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax_fi, palette='magma')
    ax_fi.set_title('XGBoost Feature Importance')
    ax_fi.set_xlabel('Importance Score')
    ax_fi.set_ylabel('Features')
    st.pyplot(fig_fi)

    st.subheader("Correlation Matrix of Selected Wallet Features")
    numeric_cols_for_corr = filtered_wallets_df.select_dtypes(include=np.number).columns.drop(['txId_mapped', 'predicted_aml_label_num', 'wallet_aml_label'])
    corr_matrix = filtered_wallets_df[numeric_cols_for_corr].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title('Correlation Matrix of Selected Wallet Features')
    st.pyplot(fig_corr)

with alerts_tab:
    st.header("AML Alerts")
    st.write("List of transactions associated with wallets predicted as 'Suspicious'.")

    alerts_df = filtered_df[filtered_df['predicted_aml_label'] == 'Suspicious'].sort_values(by='prediction_proba_suspicious', ascending=False)

    if not alerts_df.empty:
        st.write(f"Found {len(alerts_df)} potential suspicious transactions.")
        st.dataframe(alerts_df[['txId', 'date', 'transaction_amount', 'txId_mapped', 'prediction_proba_suspicious']].head(1000))

        st.subheader("Most Suspicious Wallets by Probability")
        top_suspicious_wallets = alerts_df.drop_duplicates(subset='txId_mapped').sort_values(by='prediction_proba_suspicious', ascending=False).head(10)
        st.dataframe(top_suspicious_wallets[['txId_mapped', 'prediction_proba_suspicious', 'predicted_aml_label']].set_index('txId_mapped'))
    else:
        st.info("No suspicious transactions found based on current filters.")
