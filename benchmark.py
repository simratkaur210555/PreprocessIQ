
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_validate,
)
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# --------------------------------------------------
# 1. VALIDATE AND INSPECT THE DATASET
# --------------------------------------------------

def validate_dataset(data):
    if data is None:
        raise ValueError("No dataset was provided.")

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("The dataset is empty.")

    if data.shape[1] < 2:
        raise ValueError(
            "At least two columns are required: features and target."
        )

    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names were found.")

    if data.columns.isna().any():
        raise ValueError("Some column names are missing.")

    if (data.columns.astype(str).str.strip() == "").any():
        raise ValueError("Some column names are blank.")

    return True


def inspect_dataset(data):
    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "numerical_columns": data.select_dtypes(
            include="number"
        ).columns.tolist(),
        "categorical_columns": data.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist(),
        "datetime_columns": data.select_dtypes(
            include=["datetime"]
        ).columns.tolist(),
    }


# --------------------------------------------------
# 2. GENERAL-PURPOSE DATA CLEANING
# --------------------------------------------------

def clean_dataset(data, threshold=0.8):
    """
    Clean common missing labels, convert numeric-looking text
    columns, and extract components from recognizable date columns.

    threshold: proportion of non-missing values that must parse
    successfully before a text column is converted.
    """

    validate_dataset(data)

    if not 0 < threshold <= 1:
        raise ValueError(
            "threshold must be greater than 0 and at most 1."
        )

    cleaned = data.copy()

    missing_labels = {
        "", "na", "n/a", "nan", "null", "none",
        "unknown", "not available", "not applicable"
    }

    # Normalize text columns and replace missing markers with np.nan.
    for column in cleaned.select_dtypes(
        include=["object", "string", "category"]
    ).columns:

        values = cleaned[column].astype("string").str.strip()

        values = values.mask(
            values.str.lower().isin(missing_labels)
        )

        # Explicitly use object dtype and numpy NaN for missing cells.
        cleaned[column] = values.astype(object).where(
            values.notna(),
            np.nan,
        )

    # Detect date-like text columns and extract useful components.
    original_text_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    for column in original_text_columns:
        values = cleaned[column].astype("string").str.strip()
        valid = values.notna()
        total = int(valid.sum())

        if total == 0:
            continue

        date_like = values.str.contains(
            r"[-/:]",
            regex=True,
            na=False,
        )

        if int(date_like.sum()) / total < threshold:
            continue

        parsed = pd.to_datetime(
            values.where(date_like),
            errors="coerce",
        )

        date_ratio = int(parsed.notna().sum()) / total

        if date_ratio >= threshold:
            # Avoid overwriting existing column names.
            for suffix, component in [
                ("year", parsed.dt.year),
                ("month", parsed.dt.month),
                ("day", parsed.dt.day),
                ("dayofweek", parsed.dt.dayofweek),
            ]:
                new_name = f"{column}_{suffix}"

                if new_name not in cleaned.columns:
                    cleaned[new_name] = component

            cleaned = cleaned.drop(columns=[column])

    # Convert numeric-looking text without using domain-specific names.
    for column in cleaned.select_dtypes(
        include=["object", "string"]
    ).columns:

        values = cleaned[column].astype("string").str.strip()
        valid = values.notna()
        total = int(valid.sum())

        if total == 0:
            continue

        # Remove common thousands separators.
        numeric_text = values.str.replace(
            ",", "", regex=False
        )

        # Remove surrounding currency/unit text while retaining
        # digits, decimal points, signs and scientific notation.
        numeric_text = numeric_text.str.replace(
            r"[^\d.eE+-]",
            "",
            regex=True,
        )

        converted = pd.to_numeric(
            numeric_text,
            errors="coerce",
        )

        conversion_ratio = int(
            (converted.notna() & valid).sum()
        ) / total

        if conversion_ratio >= threshold:
            cleaned[column] = converted

    # Drop columns with no usable values.
    empty_columns = cleaned.columns[
        cleaned.isna().all()
    ].tolist()

    if empty_columns:
        cleaned = cleaned.drop(columns=empty_columns)

    if cleaned.shape[1] < 2:
        raise ValueError(
            "Fewer than two usable columns remain after cleaning."
        )

    # Final safety normalization for object columns.
    for column in cleaned.select_dtypes(
        include=["object", "string"]
    ).columns:
        series = cleaned[column].astype(object)
        cleaned[column] = series.where(
            pd.notna(series),
            np.nan,
        )

    return cleaned


def prepare_dataset(data):
    validate_dataset(data)
    cleaned = clean_dataset(data)
    report = inspect_dataset(cleaned)
    return cleaned, report


# --------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

def split_features_target(data, target_column):
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    # Do not train a supervised model on unknown target values.
    usable_data = data.dropna(subset=[target_column]).copy()

    if usable_data.empty:
        raise ValueError(
            "No rows remain after removing missing target values."
        )

    X = usable_data.drop(columns=[target_column])
    y = usable_data[target_column].copy()

    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain at least two distinct values."
        )

    return X, y


# --------------------------------------------------
# 4. BUILD PREPROCESSING PIPELINES
# --------------------------------------------------

def build_preprocessor(
    X,
    scaling="standard",
    numeric_imputation="median",
    categorical_imputation="most_frequent",
):
    if scaling not in ["standard", "minmax", "none"]:
        raise ValueError(
            "scaling must be 'standard', 'minmax', or 'none'."
        )

    if numeric_imputation not in ["median", "mean"]:
        raise ValueError(
            "numeric_imputation must be 'median' or 'mean'."
        )

    if categorical_imputation not in [
        "most_frequent", "constant"
    ]:
        raise ValueError(
            "categorical_imputation must be "
            "'most_frequent' or 'constant'."
        )

    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    datetime_columns = X.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    if datetime_columns:
        raise ValueError(
            "Datetime columns are not supported directly. "
            "Convert them into date components first."
        )

    unsupported = [
        column for column in X.columns
        if column not in numeric_columns
        and column not in categorical_columns
    ]

    if unsupported:
        raise ValueError(
            f"Unsupported feature types: {unsupported}"
        )

    transformers = []

    if numeric_columns:
        numeric_steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy=numeric_imputation,
                    keep_empty_features=True,
                ),
            )
        ]

        if scaling == "standard":
            numeric_steps.append(
                ("scaler", StandardScaler())
            )
        elif scaling == "minmax":
            numeric_steps.append(
                ("scaler", MinMaxScaler())
            )

        transformers.append(
            (
                "numerical",
                Pipeline(steps=numeric_steps),
                numeric_columns,
            )
        )

    if categorical_columns:
        if categorical_imputation == "most_frequent":
            categorical_imputer = SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True,
            )
        else:
            categorical_imputer = SimpleImputer(
                strategy="constant",
                fill_value="Missing",
                keep_empty_features=True,
            )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", categorical_imputer),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No supported numerical or categorical features found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# --------------------------------------------------
# 5. RUN BENCHMARK EXPERIMENTS
# --------------------------------------------------

def run_benchmark(
    X,
    y,
    task_type="classification",
    test_size=0.2,
):
    if task_type not in ["classification", "regression"]:
        raise ValueError(
            "task_type must be 'classification' or 'regression'."
        )

    if len(X) != len(y):
        raise ValueError(
            "Features and target must have the same row count."
        )

    if y.isna().any():
        raise ValueError("The target contains missing values.")

    if len(X) < 5:
        raise ValueError("At least 5 rows are required.")

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if task_type == "classification":
        if y.nunique() < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        }

        class_counts = y.value_counts()
        stratify_data = (
            y if len(class_counts) > 1
            and class_counts.min() >= 2
            else None
        )

    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                "Regression requires a numeric target. "
                "Check the target's data format."
            )

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42,
            ),
        }

        stratify_data = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_data,
    )
    
    if task_type == "classification":
        min_class_count = int(y_train.value_counts().min())
        cv_splits = min(5, min_class_count)

        if cv_splits < 2:
            raise ValueError(
                "Cross-validation requires at least two training "
                "examples in every class."
            )

        cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=42,
        )

    else:
        cv_splits = min(5, len(X_train))

        if cv_splits < 2:
            raise ValueError(
                "At least two training rows are required "
                "for cross-validation."
            )

        cv = KFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=42,
        )

    configurations = []

    for scaling in ["standard", "minmax", "none"]:
        for numeric_imputation in ["median", "mean"]:
            for categorical_imputation in [
                "most_frequent", "constant"
            ]:
                configurations.append({
                    "scaling": scaling,
                    "numeric_imputation": numeric_imputation,
                    "categorical_imputation": categorical_imputation,
                })

    results = []

    for config in configurations:
        strategy_name = (
            f"{config['scaling']} | "
            f"numeric={config['numeric_imputation']} | "
            f"categorical={config['categorical_imputation']}"
        )

        for model_name, model in models.items():
            try:
                preprocessor = build_preprocessor(
                    X_train,
                    scaling=config["scaling"],
                    numeric_imputation=config["numeric_imputation"],
                    categorical_imputation=(
                        config["categorical_imputation"]
                    ),
                )

                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", clone(model)),
                    ]
                )

                
                if task_type == "classification":
                    scoring = {
                        "accuracy": "accuracy",
                        "precision": "precision_weighted",
                        "recall": "recall_weighted",
                        "f1_score": "f1_weighted",
                    }
                else:
                    scoring = {
                        "mae": "neg_mean_absolute_error",
                        "rmse": "neg_root_mean_squared_error",
                        "r2_score": "r2",
                    }

                scores = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    error_score="raise",
                )

                if task_type == "classification":
                    metrics = {
                        "accuracy": float(
                            scores["test_accuracy"].mean()
                        ),
                        "precision": float(
                            scores["test_precision"].mean()
                        ),
                        "recall": float(
                            scores["test_recall"].mean()
                        ),
                        "f1_score": float(
                            scores["test_f1_score"].mean()
                        ),
                    }
                else:
                    metrics = {
                        "mae": float(
                            -scores["test_mae"].mean()
                        ),
                        "rmse": float(
                            -scores["test_rmse"].mean()
                        ),
                        "r2_score": float(
                            scores["test_r2_score"].mean()
                        ),
                    }


                results.append({
                    "model": model_name,
                    "preprocessing": strategy_name,
                    "scaling": config["scaling"],
                    "numeric_imputation": (
                        config["numeric_imputation"]
                    ),
                    "categorical_imputation": (
                        config["categorical_imputation"]
                    ),
                    "status": "Success",
                    **metrics,
                })

            except Exception as error:
                results.append({
                    "model": model_name,
                    "preprocessing": strategy_name,
                    "scaling": config["scaling"],
                    "numeric_imputation": (
                        config["numeric_imputation"]
                    ),
                    "categorical_imputation": (
                        config["categorical_imputation"]
                    ),
                    "status": "Failed",
                    "error": str(error),
                })

    return pd.DataFrame(results)


# --------------------------------------------------
# 6. SELECT THE BEST EXPERIMENT
# --------------------------------------------------

def find_best_combination(results, task_type="classification"):
    if task_type not in ["classification", "regression"]:
        raise ValueError(
            "task_type must be 'classification' or 'regression'."
        )

    if results.empty:
        raise ValueError("No benchmark results are available.")

    successful = results[
        results["status"] == "Success"
    ].copy()

    if successful.empty:
        raise ValueError("All benchmark experiments failed.")

    metric = (
        "f1_score"
        if task_type == "classification"
        else "r2_score"
    )

    if metric not in successful.columns:
        raise ValueError(f"Required metric '{metric}' is missing.")

    successful = successful.dropna(subset=[metric])

    if successful.empty:
        raise ValueError(f"No valid values were found for '{metric}'.")

    return successful.sort_values(
        by=metric,
        ascending=False,
    ).iloc[0]
