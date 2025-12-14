import pandas as pd
import joblib
import json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

def evaluate(input_path, model_path, metrics_path):
    print('Model evaluation: Calculating selected metrics...')
    
    # Load data and model
    df = pd.read_csv(input_path)
    model = joblib.load(model_path)
    
    # Define features (X) and target (y)
    col = ['Risk_Label', 'CustomerId', 'WoE']
    X = df.drop(col, axis=1)
    y = df['Risk_Label']  # Ensure y is encoded as 0 and 1
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)  # Positive class is 1
    recall = recall_score(y_test, y_pred, pos_label=1)  # Positive class is 1
    f1 = f1_score(y_test, y_pred, pos_label=1)  # Positive class is 1
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    # Save metrics to file
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Evaluation metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--output", required=True, help="Path to save the metrics JSON file")
    args = parser.parse_args()
    evaluate(args.input, args.model, args.output)
