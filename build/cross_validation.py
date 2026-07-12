

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline



# 1. FILE PATHS


DATA_PATH = Path(
    r"C:\Users\ivans\Desktop\All\MSC\Data_Processsing"
    r"\assignement\air_quality_analysis\data\processed"
    r"\feature_engineered_air_quality_data.xlsx"
)

OUTPUT_PATH = DATA_PATH.parent / "cross_validation_results.xlsx"



# 2. LOAD THE FEATURE-ENGINEERED DATASET


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}\n\n"
        "Check the DATA_PATH value."
    )

df = pd.read_excel(DATA_PATH)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)
print("\nAvailable columns:")
print(df.columns.tolist())



# 3. CREATE TARGET IF IT DOES NOT ALREADY EXIST


target_column = "air_quality_level"

if target_column not in df.columns:

    if "pm25_concentration" not in df.columns:
        raise KeyError(
            "Neither air_quality_level nor pm25_concentration "
            "was found in the dataset."
        )

    air_quality_bins = [
        -np.inf,
        10,
        25,
        35,
        55,
        np.inf
    ]

    air_quality_labels = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy"
    ]

    df[target_column] = pd.cut(
        df["pm25_concentration"],
        bins=air_quality_bins,
        labels=air_quality_labels,
        include_lowest=True,
        ordered=True
    )

    print("\nThe air_quality_level target was created.")



# 4. CREATE LOG PM10 FEATURE IF IT DOES NOT EXIST


if "log_pm10_concentration" not in df.columns:

    if "pm10_concentration" not in df.columns:
        raise KeyError(
            "pm10_concentration is required to create "
            "log_pm10_concentration."
        )

    # Negative concentrations are invalid for log1p
    df.loc[
        df["pm10_concentration"] < 0,
        "pm10_concentration"
    ] = np.nan

    df["log_pm10_concentration"] = np.log1p(
        df["pm10_concentration"]
    )

    print("The log_pm10_concentration feature was created.")



# 5. DEFINE MODEL FEATURES


# Original feature dataset:
# Logistic Regression, Decision Tree and Random Forest
regular_features = [
    "country_name",
    "year",
    "pm10_concentration",
    "no2_concentration",
    "latitude",
    "longitude"
]

# Log-transformed PM10 dataset:
# SVM and KNN
log_features = [
    "country_name",
    "year",
    "log_pm10_concentration",
    "no2_concentration",
    "latitude",
    "longitude"
]

required_columns = set(
    regular_features +
    log_features +
    [target_column]
)

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise KeyError(
        "The following required columns are missing:\n"
        f"{sorted(missing_columns)}"
    )



# 6. PREPARE THE MODELLING DATASET


# Remove only records without a target.
# Predictor missing values are handled inside each CV fold.
df_model = (
    df.dropna(subset=[target_column])
      .copy()
      .reset_index(drop=True)
)

# Convert the target into text labels
df_model[target_column] = (
    df_model[target_column]
    .astype(str)
    .str.strip()
)

valid_classes = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy"
]

# Keep only recognised target classes
df_model = (
    df_model[
        df_model[target_column].isin(valid_classes)
    ]
    .copy()
    .reset_index(drop=True)
)

print("\nFinal modelling dataset shape:", df_model.shape)

print("\nTarget class distribution:")
class_distribution = (
    df_model[target_column]
    .value_counts()
    .reindex(valid_classes)
)

print(class_distribution)



# 7. CREATE ONE COMMON TRAIN-TEST SPLIT


train_data, test_data = train_test_split(
    df_model,
    test_size=0.30,
    random_state=42,
    stratify=df_model[target_column]
)

train_data = train_data.copy().reset_index(drop=True)
test_data = test_data.copy().reset_index(drop=True)

y_train = train_data[target_column]
y_test = test_data[target_column]

print("\nTraining records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining target distribution:")
print(
    y_train.value_counts(normalize=True)
    .reindex(valid_classes)
    .round(4)
)



# 8. CREATE A COMPATIBLE ONE-HOT ENCODER


def create_one_hot_encoder():
    """
    Create an encoder compatible with new and older
    scikit-learn versions.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )



# 9. CREATE MODEL-SPECIFIC PREPROCESSING


categorical_features = ["country_name"]

regular_numeric_features = [
    "year",
    "pm10_concentration",
    "no2_concentration",
    "latitude",
    "longitude"
]

log_numeric_features = [
    "year",
    "log_pm10_concentration",
    "no2_concentration",
    "latitude",
    "longitude"
]


def create_preprocessor(
    numerical_features,
    scale_numeric=False
):
    """
    Create preprocessing that is fitted separately
    inside each cross-validation fold.
    """

    if scale_numeric:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

    else:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                create_one_hot_encoder()
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numeric_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop",
        sparse_threshold=1.0
    )

    return preprocessor



# 10. CREATE THE FIVE MODEL PIPELINES


# Logistic Regression
logistic_regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            create_preprocessor(
                regular_numeric_features,
                scale_numeric=True
            )
        ),
        (
            "model",
            LogisticRegression(
                solver="lbfgs",
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# Decision Tree
decision_tree_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            create_preprocessor(
                regular_numeric_features,
                scale_numeric=False
            )
        ),
        (
            "model",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# Random Forest
random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            create_preprocessor(
                regular_numeric_features,
                scale_numeric=False
            )
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,

                # Cross-validation already runs in parallel.
                n_jobs=1
            )
        )
    ]
)


# Support Vector Machine
svm_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            create_preprocessor(
                log_numeric_features,
                scale_numeric=True
            )
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",

                # False makes cross-validation faster.
                probability=False,

                cache_size=2000,
                random_state=42
            )
        )
    ]
)


# K-Nearest Neighbours
# Oversampling occurs inside each CV training fold.
knn_pipeline = ImbalancedPipeline(
    steps=[
        (
            "preprocessing",
            create_preprocessor(
                log_numeric_features,
                scale_numeric=True
            )
        ),
        (
            "oversampling",
            RandomOverSampler(
                random_state=42
            )
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="minkowski",
                p=2,
                n_jobs=1
            )
        )
    ]
)



# 11. STORE MODELS AND THEIR REQUIRED FEATURE SETS


experiments = {
    "Logistic Regression": {
        "pipeline": logistic_regression_pipeline,
        "features": regular_features
    },

    "Decision Tree": {
        "pipeline": decision_tree_pipeline,
        "features": regular_features
    },

    "Random Forest": {
        "pipeline": random_forest_pipeline,
        "features": regular_features
    },

    "Support Vector Machine": {
        "pipeline": svm_pipeline,
        "features": log_features
    },

    "K-Nearest Neighbours": {
        "pipeline": knn_pipeline,
        "features": log_features
    }
}



# 12. CONFIGURE STRATIFIED FIVE-FOLD CROSS-VALIDATION


stratified_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring_metrics = {
    "accuracy": make_scorer(
        accuracy_score
    ),

    "balanced_accuracy": make_scorer(
        balanced_accuracy_score
    ),

    "macro_precision": make_scorer(
        precision_score,
        average="macro",
        zero_division=0
    ),

    "macro_recall": make_scorer(
        recall_score,
        average="macro",
        zero_division=0
    ),

    "macro_f1": make_scorer(
        f1_score,
        average="macro",
        zero_division=0
    ),

    "weighted_f1": make_scorer(
        f1_score,
        average="weighted",
        zero_division=0
    )
}



# 13. RUN CROSS-VALIDATION FOR ALL FIVE MODELS


cross_validation_summary = []
individual_fold_results = {}

for model_name, experiment in experiments.items():

    print("\n" + "=" * 65)
    print(f"Running cross-validation: {model_name}")
    print("=" * 65)

    model_pipeline = experiment["pipeline"]
    feature_columns = experiment["features"]

    X_train_model = train_data[feature_columns]

    cv_output = cross_validate(
        estimator=model_pipeline,
        X=X_train_model,
        y=y_train,
        cv=stratified_cv,
        scoring=scoring_metrics,
        return_train_score=True,

        # Uses available processor cores.
        n_jobs=-1,

        error_score="raise"
    )

    fold_table = pd.DataFrame({
        "Fold": range(1, 6),

        "Accuracy":
            cv_output["test_accuracy"],

        "Balanced Accuracy":
            cv_output["test_balanced_accuracy"],

        "Macro Precision":
            cv_output["test_macro_precision"],

        "Macro Recall":
            cv_output["test_macro_recall"],

        "Macro F1-score":
            cv_output["test_macro_f1"],

        "Weighted F1-score":
            cv_output["test_weighted_f1"],

        "Fit Time":
            cv_output["fit_time"]
    })

    individual_fold_results[model_name] = fold_table

    cross_validation_summary.append({
        "Model": model_name,

        "Accuracy Mean":
            cv_output["test_accuracy"].mean(),

        "Accuracy Standard Deviation":
            cv_output["test_accuracy"].std(ddof=1),

        "Balanced Accuracy Mean":
            cv_output["test_balanced_accuracy"].mean(),

        "Balanced Accuracy Standard Deviation":
            cv_output["test_balanced_accuracy"].std(ddof=1),

        "Macro Precision Mean":
            cv_output["test_macro_precision"].mean(),

        "Macro Precision Standard Deviation":
            cv_output["test_macro_precision"].std(ddof=1),

        "Macro Recall Mean":
            cv_output["test_macro_recall"].mean(),

        "Macro Recall Standard Deviation":
            cv_output["test_macro_recall"].std(ddof=1),

        "Macro F1 Mean":
            cv_output["test_macro_f1"].mean(),

        "Macro F1 Standard Deviation":
            cv_output["test_macro_f1"].std(ddof=1),

        "Weighted F1 Mean":
            cv_output["test_weighted_f1"].mean(),

        "Weighted F1 Standard Deviation":
            cv_output["test_weighted_f1"].std(ddof=1),

        "Training Macro F1 Mean":
            cv_output["train_macro_f1"].mean(),

        "Average Fit Time":
            cv_output["fit_time"].mean()
    })

    print("\nIndividual fold results:")
    display(fold_table.round(4))



# 14. CREATE THE CROSS-VALIDATION SUMMARY TABLE


cv_results = pd.DataFrame(
    cross_validation_summary
)

# Macro F1 is used as the main ranking metric because
# the air-quality classes are imbalanced.
cv_results = (
    cv_results
    .sort_values(
        by="Macro F1 Mean",
        ascending=False
    )
    .reset_index(drop=True)
)

cv_results.insert(
    0,
    "Rank",
    range(1, len(cv_results) + 1)
)

print("\n" + "=" * 70)
print("FINAL CROSS-VALIDATION SUMMARY")
print("=" * 70)

display(cv_results.round(4))



#15. SELECT THE BEST MODEL


best_model_name = cv_results.loc[0, "Model"]

best_pipeline = clone(
    experiments[best_model_name]["pipeline"]
)

best_features = experiments[best_model_name]["features"]

print("\nBest model selected using mean Macro F1-score:")
print(best_model_name)



# 16. TRAIN THE BEST MODEL ON ALL TRAINING DATA


X_train_best = train_data[best_features]
X_test_best = test_data[best_features]

best_pipeline.fit(
    X_train_best,
    y_train
)

y_test_prediction = best_pipeline.predict(
    X_test_best
)



# 17. FINAL UNTOUCHED TEST-SET EVALUATION


final_test_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-score",
        "Weighted F1-score"
    ],

    "Score": [
        accuracy_score(
            y_test,
            y_test_prediction
        ),

        balanced_accuracy_score(
            y_test,
            y_test_prediction
        ),

        precision_score(
            y_test,
            y_test_prediction,
            average="macro",
            zero_division=0
        ),

        recall_score(
            y_test,
            y_test_prediction,
            average="macro",
            zero_division=0
        ),

        f1_score(
            y_test,
            y_test_prediction,
            average="macro",
            zero_division=0
        ),

        f1_score(
            y_test,
            y_test_prediction,
            average="weighted",
            zero_division=0
        )
    ]
})

print("\n" + "=" * 70)
print(f"FINAL TEST RESULTS: {best_model_name}")
print("=" * 70)

display(final_test_results.round(4))



# 18. CLASSIFICATION REPORT


classification_report_dictionary = classification_report(
    y_test,
    y_test_prediction,
    labels=valid_classes,
    output_dict=True,
    zero_division=0
)

classification_report_table = (
    pd.DataFrame(
        classification_report_dictionary
    )
    .transpose()
)

print("\nClassification Report:")
display(classification_report_table.round(4))



# 19. CONFUSION MATRIX TABLE


confusion_matrix_values = confusion_matrix(
    y_test,
    y_test_prediction,
    labels=valid_classes
)

confusion_matrix_table = pd.DataFrame(
    confusion_matrix_values,

    index=[
        f"Actual: {class_name}"
        for class_name in valid_classes
    ],

    columns=[
        f"Predicted: {class_name}"
        for class_name in valid_classes
    ]
)

print("\nConfusion Matrix:")
display(confusion_matrix_table)



# 20. DISPLAY CONFUSION MATRIX GRAPH


figure, axis = plt.subplots(
    figsize=(12, 9)
)

display_matrix = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix_values,
    display_labels=valid_classes
)

display_matrix.plot(
    ax=axis,
    values_format="d",
    xticks_rotation=35
)

plt.title(
    f"Confusion Matrix – {best_model_name}"
)

plt.tight_layout()
plt.show()



# 21. CREATE CV RESULT GRAPH


plot_data = cv_results.sort_values(
    by="Macro F1 Mean",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_data["Model"],
    plot_data["Macro F1 Mean"],
    xerr=plot_data["Macro F1 Standard Deviation"],
    capsize=4
)

plt.xlabel("Mean Macro F1-score")
plt.ylabel("Model")

plt.title(
    "Stratified Five-Fold Cross-Validation Results"
)

plt.xlim(
    0,
    min(
        1.0,
        plot_data["Macro F1 Mean"].max() + 0.15
    )
)

plt.tight_layout()
plt.show()



# 22. SAVE ALL RESULTS TO ONE EXCEL FILE


with pd.ExcelWriter(
    OUTPUT_PATH,
    engine="openpyxl"
) as writer:

    cv_results.to_excel(
        writer,
        sheet_name="CV Summary",
        index=False
    )

    final_test_results.to_excel(
        writer,
        sheet_name="Final Test Results",
        index=False
    )

    classification_report_table.to_excel(
        writer,
        sheet_name="Classification Report"
    )

    confusion_matrix_table.to_excel(
        writer,
        sheet_name="Confusion Matrix"
    )

    class_distribution.to_frame(
        name="Count"
    ).to_excel(
        writer,
        sheet_name="Class Distribution"
    )

    for model_name, fold_table in individual_fold_results.items():

        safe_sheet_name = (
            model_name
            .replace("Support Vector Machine", "SVM Folds")
            .replace("Logistic Regression", "Logistic Folds")
            .replace("K-Nearest Neighbours", "KNN Folds")
            .replace("Decision Tree", "Decision Tree Folds")
            .replace("Random Forest", "Random Forest Folds")
        )

        safe_sheet_name = safe_sheet_name[:31]

        fold_table.to_excel(
            writer,
            sheet_name=safe_sheet_name,
            index=False
        )



# 23. COMPLETION MESSAGE


print("\n" + "=" * 70)
print("CROSS-VALIDATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nBest model: {best_model_name}")

print(
    "Best cross-validation Macro F1-score:",
    round(
        cv_results.loc[0, "Macro F1 Mean"],
        4
    )
)

print(
    "Final test accuracy:",
    round(
        accuracy_score(
            y_test,
            y_test_prediction
        ),
        4
    )
)

print(
    "Final test Macro F1-score:",
    round(
        f1_score(
            y_test,
            y_test_prediction,
            average="macro",
            zero_division=0
        ),
        4
    )
)

