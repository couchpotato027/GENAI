import os
import joblib
from exam_difficulty_predictor import (
    load_all_splits,
    assign_difficulty_labels,
    simulate_student_scores,
    build_features,
    train_models
)
from sklearn.preprocessing import LabelEncoder

def main():
    print("=" * 50)
    print("Pre-training Machine Learning Models...")
    print("=" * 50)

    # 1. Load Data
    print("Loading data...")
    train_df, _, _ = load_all_splits()
    train_df = assign_difficulty_labels(train_df, seed=42)
    train_df = simulate_student_scores(train_df, seed=42)

    # 2. Build Features
    print("Building features...")
    X_train, tfidf, scaler = build_features(train_df, fit=True)

    # 3. Encode Labels
    print("Encoding labels...")
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["difficulty"])

    # 4. Train Models
    print("Training models...")
    models = train_models(X_train, y_train)

    # 5. Save Artifacts
    print("Saving artifacts to backend/...")
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    joblib.dump(models["Logistic Regression"], os.path.join(model_dir, "model_lr.joblib"))
    joblib.dump(tfidf, os.path.join(model_dir, "tfidf.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.joblib"))

    print("=" * 50)
    print("✅ Pre-training complete. Artifacts saved successfully.")
    print("=" * 50)

if __name__ == "__main__":
    main()
