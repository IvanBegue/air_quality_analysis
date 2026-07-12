import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)



# 1. IMPORT DATASET


df =pd.read_excel(r"C:\Users\ivans\Desktop\All\MSC\Data_Processsing\assignement\air_quality_analysis\data\processed\air_quality_data.xlsx")



# 2.SELECT FEATURES AND TARGET


feature_columns = [
    "country_name",
    "year",
    "pm10_concentration",
    "no2_concentration",
    "latitude",
    "longitude"
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
        f"Required columns are missing: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "The selected features or target contain missing values."
    )

X = df[feature_columns].copy()
y = df[target_column].copy()

print("\nSelected features:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())



#TRAIN/TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nTraining feature shape:")
print(X_train.shape)

print("\nTesting feature shape:")
print(X_test.shape)


# 4. ENCODE COUNTRY_NAME


X_train_encoded = pd.get_dummies(
    X_train,
    columns=["country_name"],
    drop_first=False,
    dtype=int
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns=["country_name"],
    drop_first=False,
    dtype=int
)

# Ensure the test set has the same columns
# and column order as the training set
X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)

print("\nTraining shape after encoding:")
print(X_train_encoded.shape)

print("\nTesting shape after encoding:")
print(X_test_encoded.shape)



# 5. CREATE DECISION TREE MODEL


decision_tree_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)



# 6. TRAIN THE MODEL




decision_tree_model.fit(
    X_train_encoded,
    y_train
)



print("\nTree depth:")
print(decision_tree_model.get_depth())

print("\nNumber of leaf nodes:")
print(decision_tree_model.get_n_leaves())


# 7. MAKE PREDICTIONS


y_prediction = decision_tree_model.predict(
    X_test_encoded
)

# Training predictions are used to check overfitting
y_train_prediction = decision_tree_model.predict(
    X_train_encoded
)



# 8. EVALUATE MODEL PERFORMANCE


training_accuracy = accuracy_score(
    y_train,
    y_train_prediction
)

accuracy = accuracy_score(
    y_test,
    y_prediction
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_prediction
)

macro_precision = precision_score(
    y_test,
    y_prediction,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_prediction,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_prediction,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_prediction,
    average="weighted",
    zero_division=0
)


class_labels = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy"
]


conf_matrix = confusion_matrix(
    y_test,
    y_prediction,
    labels=class_labels
)

class_report = classification_report(
    y_test,
    y_prediction,
    labels=class_labels,
    zero_division=0
)


# 9. DISPLAY RESULTS




print(f"Training Accuracy: {training_accuracy:.4f}")
print(f"Testing Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)



# 10. DISPLAY CONFUSION MATRIX


ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_prediction,
    labels=class_labels,
    display_labels=class_labels,
    xticks_rotation=45,
    values_format="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - Decision Tree")
plt.tight_layout()



# 11. CREATE RESULTS FOLDER


results_folder = Path(
    r"C:\Users\ivans\Desktop\All\MSC\Data_Processsing"
    r"\assignement\air_quality_analysis\results"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    results_folder / "decision_tree_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()



# 12. CREATE AND SAVE RESULT TABLE


decision_tree_results = pd.DataFrame({
    "Model": [
        "Decision Tree"
    ],

    "Accuracy": [
        accuracy
    ],

    "Balanced Accuracy": [
        balanced_accuracy
    ],

    "Macro Precision": [
        macro_precision
    ],

    "Macro Recall": [
        macro_recall
    ],

    "Macro F1-score": [
        macro_f1
    ],

    "Weighted F1-score": [
        weighted_f1
    ]
})


print(
    decision_tree_results.round(4)
)

decision_tree_results.to_csv(
    results_folder / "decision_tree_results.csv",
    index=False
)

