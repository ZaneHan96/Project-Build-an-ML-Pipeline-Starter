# NYC Airbnb ML Pipeline

## 🔗 Links for Review

- **W&B Project:** [https://wandb.ai/zanehan-udacity/nyc_airbnb](https://wandb.ai/zanehan-udacity/nyc_airbnb)  
- **W&B View-Only Report:** [https://wandb.ai/zanehan-udacity/nyc_airbnb/reports/NYC-Airbnb-ML-Pipeline-Project---VmlldzoxNDczMzMyNw?accessToken=buausdl0xfsbwtjo5yucyz80g38u4asntme1l7q7tpt4rqlwhzhl5qzopxt6e55c](https://wandb.ai/zanehan-udacity/nyc_airbnb/reports/NYC-Airbnb-ML-Pipeline-Project---VmlldzoxNDczMzMyNw?accessToken=buausdl0xfsbwtjo5yucyz80g38u4asntme1l7q7tpt4rqlwhzhl5qzopxt6e55c)
- **GitHub Repo:** [https://github.com/ZaneHan96/Project-Build-an-ML-Pipeline-Starter](https://github.com/ZaneHan96/Project-Build-an-ML-Pipeline-Starter)

---

## What’s Included

- **End-to-end ML pipeline:**  
  `download → basic_cleaning → data_check → data_split → train_random_forest (+ evaluation)`
- **Artifacts tracked in W&B:**  
  - `raw_data/sample.csv`  
  - `clean_data/clean_sample.csv`  
  - `trainval_data.csv`  
  - `test_data.csv`  
  - `model_export/random_forest_export`
- **Logged metrics:**  
  - R² and MAE for training and evaluation (available in W&B runs)
- **Tools used:**  
  - MLflow for orchestration  
  - Hydra for configuration  
  - Weights & Biases (W&B) for experiment tracking and artifact management

---

## Best Model Summary

- **Run name:** `efficient-silence-17`  
- **R²:** 0.5640  
- **MAE:** 33.85  
- **Parameters:**  
  - `n_estimators = 100`  
  - `max_depth = 10`  
  - `min_samples_split = 4`  
  - `min_samples_leaf = 3`  
  - `max_features = 0.5`  
  - `max_tfidf_features = 5`
- **Promoted Model Artifact (@prod):**  
  [random_forest_export:v10 @prod](https://wandb.ai/zanehan-udacity/nyc_airbnb/artifacts/model_export/random_forest_export/v10)

---

## How to Run Locally

```bash
# 1. Activate environment
conda activate nyc_airbnb_dev

# 2. Run the full pipeline
mlflow run . --env-manager=local -P steps=all

# 3. (Optional) Reproduce the best model training only
mlflow run . --env-manager=local -P steps=train_random_forest   -P hydra_options="modeling.random_forest.n_estimators=100                     modeling.random_forest.max_depth=10                     modeling.random_forest.min_samples_split=4                     modeling.random_forest.min_samples_leaf=3                     modeling.random_forest.max_features=0.5                     modeling.max_tfidf_features=5"
```

---

## Project Description

You are working for a property management company renting rooms and properties for short periods of time on various rental platforms.  
The goal is to estimate the typical nightly price for a given property based on similar listings in NYC.  

Your company receives **new data in bulk every week**, so the model must be **retrainable and fully automated** via an **ML pipeline**.

This project delivers:
- A reproducible ML pipeline built with MLflow, Hydra, and W&B  
- Automated data ingestion, cleaning, validation, splitting, model training, and evaluation  
- Versioned data and models via W&B Artifacts  
- Model promotion workflow for deployment (`@prod` model)

---

## Tech Stack
- Python 3.10
- Conda (environment managed via `environment.yml`)
- MLflow
- Hydra
- Weights & Biases (W&B)
- Scikit-learn
- Pandas, NumPy

## Results Summary
The best-performing model (`efficient-silence-17`) achieved an R² of 0.5640 and MAE of 33.85 on the validation data, showing stable performance across training and evaluation.





## License

[License](LICENSE.txt)
