"""
model_train.py
Trains a TF-IDF + LogisticRegression classifier on the waste dataset.
Saves vectorizer and model to models/ directory.
Run: python src/model_train.py
"""
import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

base = Path(__file__).resolve().parents[1]
data_path = base / "data" / "waste_dataset_india_synthetic.csv"
models_dir = base / "models"
models_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)
X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.9)),
    ("clf", LogisticRegression(max_iter=1000, solver="saga", multi_class="multinomial"))
])

# Quick gridsearch for C parameter (keeps runtime reasonable)
param_grid = {
    "clf__C": [0.1, 1.0, 5.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
best = grid.best_estimator_

# Evaluate
y_pred = best.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save artifacts
joblib.dump(best.named_steps["tfidf"], models_dir / "vectorizer.joblib")
joblib.dump(best.named_steps["clf"], models_dir / "classifier.joblib")
# Also save the full pipeline
joblib.dump(best, models_dir / "pipeline_full.joblib")
print("Saved vectorizer and classifier to", models_dir)
