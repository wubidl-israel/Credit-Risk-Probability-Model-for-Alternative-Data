import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import argparse
import joblib

def train(input_path):
   
    df= pd.read_csv(input_path)
    # Columns to exclude
    col = ['Risk_Label', 'CustomerId','WoE']
    x = df.drop(col, axis=1)
    y = df['Risk_Label']
    
  
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    
    # Define hyperparameter grid
    param_grid = {
        'min_samples_split': [2, 10, 30, 50, 100, 200, 300, 700],
        'max_depth': [1, 2, 3, 4, 8, 16, 32, 64, None],
        'n_estimators': [10, 50, 100, 500]
    }
    
    # Initialize RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    # Fit GridSearchCV
    grid_search.fit(x_train, y_train)
    joblib.dump(best_model, 'model.pkl')

if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help='Path to read csv')
    
    args=parser.parse_args()
    train(args.input)








"""  # Best parameters and model performance
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f'Best Parameters: {best_params}')
    print(f'Best Cross-Validation Score: {best_score:.2f}')
    
    # Test the best model on the test set
    best_model = grid_search.best_estimator_
    yhat = best_model.predict(x_test)
    train_pred = best_model.predict(x_train)
    
    print(f'Test set accuracy: {accuracy_score(y_test, yhat):.2f}')
    print(f'Train set accuracy: {accuracy_score(y_train, train_pred):.2f}')
"""
