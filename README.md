# Waste Classifier (Text) — Indian Basics (TF-IDF + LogisticRegression)

This project contains a complete text-based waste classifier tailored with an "Indian basics" synthetic dataset.
It includes:
- dataset: `data/waste_dataset_india_synthetic.csv` (10000 rows)
- training script: `src/model_train.py`
- Streamlit UI: `src/app.py`
- trained artifacts (if you run training): `models/vectorizer.joblib`, `models/classifier.joblib`, `models/pipeline_full.joblib`

### Folder structure
```
waste_classifier_project/
├─ data/
│  └─ waste_dataset_india_synthetic.csv
├─ models/
│  └─ (empty until you train or use provided artifacts)
├─ src/
│  ├─ model_train.py
│  └─ app.py
├─ requirements.txt
└─ README.md
```

### Quick setup (Linux / macOS / Windows WSL)
1. Create and activate a virtual environment (optional but recommended):
   - `python -m venv venv && source venv/bin/activate` (mac/linux)
   - `python -m venv venv && venv\Scripts\activate` (windows)

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (creates `models/pipeline_full.joblib`):
   ```bash
   python src/model_train.py
   ```
   This will print classification report and save vectorizer + classifier.

4. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

### Notes & suggestions
- The dataset is synthetic but tuned for Indian common items (banana peel, coconut husk, plastic bottle, mobile phone battery, etc.).
- For production, replace the dataset with a curated real dataset of labeled items and retrain.
- You can improve accuracy by trying different classifiers (RandomForest, XGBoost), more hyperparameter search, or pre-processing rules.
- The Streamlit app uses the saved pipeline for prediction; run training first to create artifacts, or I can provide trained artifacts too.

Enjoy! — Generated automatically.
