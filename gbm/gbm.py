# Written with assistance of Gemini
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class GenericTabularClassifier:
    def __init__(self, model_params=None):
        # Default parameters optimized for general tabular data
        default_params = {
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 4,
            "random_state": 42,
            "eval_metric": "logloss"
        }
        
        # Override defaults if user provides custom parameters
        if model_params:
            default_params.update(model_params)
            
        self.model = xgb.XGBClassifier(**default_params)
        self.features = None
        self.target = None

    def prepare_and_split_data(self, df, target_column, test_size=0.2, random_state=42):
        """Dynamically splits any DataFrame into Train/Test subsets."""
            
        # Dynamically separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        self.features = X.columns
        self.target = target_column
        
        # Stratify ensures balanced class representation in train and test sets
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def train(self, X_train, y_train):
        """Fits the XGBoost model onto the training data."""
        self.model.fit(X_train, y_train)
        print("Model training completed successfully.")


    def evaluate(self, X_test, y_test):
        """Prints performance reports and dynamic feature importance scores."""
        predictions = self.model.predict(X_test)
        
        print("\n" + "="*20 + " EVALUATION METRICS " + "="*20)
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_test, predictions))
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, predictions))
        
        print("\n--- Feature Importances ---")
        importances = self.model.feature_importances_
        # Sort features from most important to least important
        sorted_indices = np.argsort(importances)[::-1]
        
        for idx in sorted_indices:
            print(f"Feature: {self.features[idx]:<30} Importance: {importances[idx]:.4f}")


if __name__ == "__main__":
    np.random.seed(123)
    sample_size = 600
    data = {
        "transaction_amount_usd": np.random.uniform(1, 5000, size=sample_size),
        "is_international": np.random.randint(0, 2, size=sample_size),
        "failed_login_attempts": np.random.randint(0, 10, size=sample_size),
        "cardholder_age": np.random.randint(18, 90, size=sample_size),
        "is_fraudulent_label": np.random.choice([0, 1], size=sample_size, p=[0.92, 0.08]) 
    }
    df_sample = pd.DataFrame(data)

    # Initialise the pipeline with optional hyperparameter tweaks
    pipeline = GenericTabularClassifier(model_params={"n_estimators": 100, "max_depth": 5})

    # Process the variables dynamically by targeting 'is_fraudulent_label'
    X_train, X_test, y_train, y_test = pipeline.prepare_and_split_data(
        df=df_sample, 
        target_column="is_fraudulent_label"
    )

    # Execute and inspect outputs
    pipeline.train(X_train, y_train)
    pipeline.evaluate(X_test, y_test)
