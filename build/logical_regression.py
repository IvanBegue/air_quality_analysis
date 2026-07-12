import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
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
df =pd.read_excel(r"C:\Users\ivans\Desktop\All\MSC\Data_Processsing\assignement\air_quality_analysis\data\processed\air_quality_data.xlsx")


feature_columns = [
    "country_name",
    "year",
    "pm10_concentration",
    "no2_concentration",
    "latitude",
    "longitude"
]

target_column = "air_quality_level"


X =df[feature_columns].copy()

y=df[target_column].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nTrain/test split completed.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Convert categorical country names into dummy variables

X_train_encoded = pd.get_dummies(
    X_train,
    columns=["country_name"],
    drop_first=False
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns=["country_name"],
    drop_first=False
)


# Ensure that the testing set has exactly the same

X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)

print("\nTraining shape after encoding:")
print(X_train_encoded.shape)

print("\nTesting shape after encoding:")
print(X_test_encoded.shape)



# 5. FEATURE SCALING


# StandardScaler transforms the features so that
# they have approximately mean = 0 and standard deviation = 1
scaler = StandardScaler()

# Learn scaling values from training data only
X_train_scaled = scaler.fit_transform(
    X_train_encoded
)

# Apply the same scaling to testing data
X_test_scaled = scaler.transform(
    X_test_encoded
)



# 6. CREATE LOGISTIC REGRESSION MODEL


# class_weight="balanced" gives more importance
# to minority air-quality classes
log_model = LogisticRegression(
    class_weight="balanced",
    max_iter=3000,
    solver="lbfgs",
    random_state=42
)


# 7. TRAIN THE MODEL



log_model.fit(
    X_train_scaled,
    y_train
)

print("Model training completed.")



# 8. MAKE PREDICTIONS

y_prediction = log_model.predict(
    X_test_scaled
)



# 9. EVALUATE MODEL PERFORMANCE


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


# Define class order for the report and confusion matrix
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


# 10. DISPLAY RESULTS


print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)


# 11. DISPLAY CONFUSION MATRIX


ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_prediction,
    labels=class_labels,
    display_labels=class_labels,
    xticks_rotation=45,
    values_format="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.tight_layout()
plt.show()


# 12. CREATE RESULT TABLE


logistic_regression_results = pd.DataFrame({
    "Model": [
        "Logistic Regression"
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

print("\nModel Result Table:")
print(
    logistic_regression_results.round(4)
)


# 13. SAVE RESULTS
results_path = (
    r"C:\Users\ivans\Desktop\All\MSC\Data_Processsing"
    r"\assignement\air_quality_analysis\results"
    r"\logistic_regression_results.csv"
)

logistic_regression_results.to_csv(
    results_path,
    index=False
)

