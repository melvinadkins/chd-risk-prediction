import joblib 
import pandas as pd 
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.config import (TRAIN_PATH, 
                        TEST_PATH, 
                        MODEL_PATH, 
                        RANDOM_STATE, 
                        THRESHOLD)
from src.feature_engineering import CAT_FEATURES, NUM_FEATURES
from src.preprocessing import build_pipeline


def load_train_test_data():
    train_df = pd.read_csv(TRAIN_PATH) 
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop("TenYearCHD", axis=1)
    y_train = train_df["TenYearCHD"]
    X_test = test_df.drop("TenYearCHD", axis=1)
    y_test = test_df["TenYearCHD"]
    
    return X_train, X_test, y_train, y_test

def chd_risk_model():
    """
    Instantiates LogisticRegression with tuned hyperparameters identified from RandomizedSearchCV.
    """
    return LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=10000,
        tol=1e-3,
        C=2.077480993119357,
        l1_ratio=0.631578947368421,
        random_state=RANDOM_STATE,
    )

def evaluate_model(model, X_test, y_test, threshold=THRESHOLD):
    """
    Evaluates model performance on the test set using the specified threshold.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")

def main():
    # Load data
    print("Loading train and test data...")
    X_train, X_test, y_train, y_test = load_train_test_data()

    # Train model
    print("Training model...")

    # Load CHD risk model
    model = chd_risk_model()

    # Build and fit pipeline
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(pipeline, X_test, y_test)

    # Save model pipeline and threshold
    print("Saving model pipeline...")
    joblib.dump({"pipeline": pipeline, 
                 "threshold": THRESHOLD,
                 "features": NUM_FEATURES + CAT_FEATURES,
                 "model_type": "LogisticRegression"}, 
                 MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()