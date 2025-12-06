from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).parent.parent / "dataset" / "SCDB_2025_01_caseCentered_Citation.csv"

FEATURES = [
    "issue", "issueArea",
    "petitioner", "petitionerState", "respondent", "respondentState",
    "jurisdiction", "adminAction",
    "caseOrigin", "caseOriginState", "caseSource", "caseSourceState",
    "lcDisagreement", "certReason", "lcDisposition", "lcDispositionDirection",
    "declarationUncon",
    "authorityDecision1", "authorityDecision2", "lawType", "lawSupp", "lawMinor",
    "term", "naturalCourt", "chief",
]


def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)
    df["dateArgumentYear"] = pd.to_datetime(df["dateArgument"], errors="coerce").dt.year
    df = df.dropna(subset=["decisionDirection", "dateArgumentYear"])
    X = df[FEATURES + ["dateArgumentYear"]]
    y = df["decisionDirection"]
    return X, y


def build_pipeline():
    cat_features = FEATURES
    num_features = ["dateArgumentYear"]

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical, cat_features),
            ("num", numeric, num_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )

    return Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])


def train_and_evaluate(random_state: int = 42):
    X, y = load_data()
    years = X["dateArgumentYear"]
    train_mask = years <= 2019
    recent_mask = years >= 2020

    X_train, y_train = X[train_mask], y[train_mask]
    X_recent, y_recent = X[recent_mask], y[recent_mask]

    X_val, X_test, y_val, y_test = train_test_split(
        X_recent,
        y_recent,
        test_size=0.5,
        stratify=y_recent,
        random_state=random_state,
    )

    pipeline = build_pipeline()

    param_distributions = {
        "model__max_depth": [None, 15, 20, 25],
        "model__min_samples_split": [5, 10, 15, 20, 30, 40],
        "model__min_samples_leaf": [2, 5, 10, 15, 20],
        "model__max_features": ["sqrt", "log2", 0.2, 0.4, 0.6],
        "model__n_estimators": [200, 300, 400, 500, 600],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )

    search.fit(X_train, y_train)
    pipeline = search.best_estimator_

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)
    print(f"Train accuracy: {accuracy_score(y_train, train_pred):.3f}")
    print(f"Validation accuracy: {accuracy_score(y_val, val_pred):.3f}")
    print(f"Best params: {search.best_params_}")

    return pipeline


if __name__ == "__main__":
    train_and_evaluate()
