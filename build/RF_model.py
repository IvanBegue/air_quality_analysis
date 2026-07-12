import pickle
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ===================================================
# 1. PROJECT PATHS
# ===================================================

project_path = Path(
    r"C:\Users\ivans\Desktop\All\MSC\Data_Processsing"
    r"\assignement\air_quality_analysis"
)

dataset_path = (
    project_path
    / "data"
    / "processed"
    / "feature_engineered_air_quality_data.xlsx"
)

models_folder = project_path / "models"
models_folder.mkdir(parents=True, exist_ok=True)

pickle_path = (
    models_folder
    / "random_forest_air_quality.pkl"
)


# ===================================================
# 2. LOAD DATASET
# ===================================================

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{dataset_path}"
    )

df = pd.read_excel(dataset_path)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ===================================================
# 3. DEFINE FEATURES AND TARGET
# ===================================================

feature_columns = [
    "country_name",
    "year",
    "pm10_concentration",
    "no2_concentration",
    "latitude",
    "longitude",
]

target_column = "air_quality_level"
required_columns = feature_columns + [target_column]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise KeyError(
        f"Missing columns: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Selected features or target contain missing values."
    )

X = df[feature_columns].copy()
y = df[target_column].copy()


# ===================================================
# 4. TRAIN/TEST SPLIT
# ===================================================

requested_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=requested_test_size,
    random_state=42,
    stratify=y,
)

train_samples = int(len(X_train))
test_samples = int(len(X_test))
total_samples = int(len(X))

# Store the actual proportion after row-count rounding.
actual_test_fraction = test_samples / total_samples

print("\nDataset split:")
print(f"Training samples: {train_samples:,}")
print(f"Testing samples: {test_samples:,}")
print(f"Total samples: {total_samples:,}")
print(f"Actual test proportion: {actual_test_fraction:.2%}")


# ===================================================
# 5. PREPROCESSING
# ===================================================

categorical_features = [
    "country_name",
]

numerical_features = [
    "year",
    "pm10_concentration",
    "no2_concentration",
    "latitude",
    "longitude",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "country_encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ],
    remainder="passthrough",
)


# ===================================================
# 6. RANDOM FOREST MODEL
# ===================================================

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    criterion="gini",
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ===================================================
# 7. COMPLETE PIPELINE
# ===================================================

random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            random_forest_model,
        ),
    ]
)


# ===================================================
# 8. TRAIN MODEL
# ===================================================

print("\nTraining final Random Forest pipeline...")

random_forest_pipeline.fit(
    X_train,
    y_train,
)


# ===================================================
# 9. TEST FINAL PIPELINE
# ===================================================

y_prediction = random_forest_pipeline.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_prediction,
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_prediction,
)

macro_f1 = f1_score(
    y_test,
    y_prediction,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_test,
    y_prediction,
    average="weighted",
    zero_division=0,
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_prediction,
        zero_division=0,
    )
)


# ===================================================
# 10. CREATE DEPLOYMENT BUNDLE
# ===================================================

country_list = sorted(
    df["country_name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

model_bundle = {
    "model": random_forest_pipeline,
    "countries": country_list,
    "feature_columns": feature_columns,
    "target_column": target_column,
    "model_name": "Random Forest",

    # Model performance
    "test_accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),

    # Dataset split
    "train_samples": train_samples,
    "test_samples": test_samples,
    "total_samples": total_samples,
    "requested_test_fraction": float(requested_test_size),
    "actual_test_fraction": float(actual_test_fraction),
}


# ===================================================
# 11. SAVE AS PICKLE
# ===================================================

with open(
    pickle_path,
    "wb",
) as pickle_file:
    pickle.dump(
        model_bundle,
        pickle_file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print("\nPickle bundle saved successfully:")
print(pickle_path)


# ===================================================
# 12. VERIFY SAVED PICKLE
# ===================================================

with open(
    pickle_path,
    "rb",
) as pickle_file:
    loaded_bundle = pickle.load(
        pickle_file
    )

loaded_model = loaded_bundle["model"]
sample_input = X_test.iloc[[0]].copy()

original_prediction = (
    random_forest_pipeline.predict(
        sample_input
    )[0]
)

loaded_prediction = (
    loaded_model.predict(
        sample_input
    )[0]
)

if original_prediction != loaded_prediction:
    raise RuntimeError(
        "The loaded model prediction does not match "
        "the original model prediction."
    )

print("\nSaved bundle verification:")
print("Sample prediction:", loaded_prediction)
print(
    "Saved test proportion:",
    f'{loaded_bundle["actual_test_fraction"]:.2%}',
)
print(
    "Saved testing samples:",
    f'{loaded_bundle["test_samples"]:,}',
)
print(
    "Saved Weighted F1-score:",
    f'{loaded_bundle["weighted_f1"]:.4f}',
)
