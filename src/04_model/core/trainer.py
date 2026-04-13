from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def create_model(model_name: str, model_params: dict[str, Any] | None = None) -> Any:

    if model_params is None:
        model_params = {}

    model_registry = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "xgboost": XGBClassifier,
        "decision_tree": DecisionTreeClassifier,
        "svm": SVC,
    }

    if model_name not in model_registry:
        valid_models = ", ".join(model_registry.keys())
        raise ValueError(f"Unsupported model_name: '{model_name}'. Available options: {valid_models}")

    model_class = model_registry[model_name]
    model = model_class(**model_params)

    return model

def split_features_and_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing in the DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def fit_model(
    model: Any,
    train_df: pd.DataFrame,
    target_column: str,
    drop_columns: list[str] | None = None,
) -> Any:

    if drop_columns is None:
        drop_columns = []

    train_input_df = train_df.drop(columns=drop_columns, errors="ignore").copy()
    X_train, y_train = split_features_and_target(train_input_df, target_column)

    model.fit(X_train, y_train)

    return model


def predict_labels(
    model: Any,
    test_df: pd.DataFrame,
    target_column: str,
    drop_columns: list[str] | None = None,
) -> pd.Series:

    if drop_columns is None:
        drop_columns = []

    test_input_df = test_df.drop(columns=drop_columns, errors="ignore").copy()
    X_test, _ = split_features_and_target(test_input_df, target_column)

    predictions = model.predict(X_test)

    return pd.Series(predictions, index=test_df.index, name="y_pred")


def train_and_predict(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    model_params: dict[str, Any] | None = None,
    drop_columns: list[str] | None = None,
) -> tuple[Any, pd.Series]:

    model = create_model(model_name=model_name, model_params=model_params)
    
    trained_model = fit_model(
        model=model,
        train_df=train_df,
        target_column=target_column,
        drop_columns=drop_columns,
    )
    
    y_pred = predict_labels(
        model=trained_model,
        test_df=test_df,
        target_column=target_column,
        drop_columns=drop_columns,
    )

    return trained_model, y_pred
