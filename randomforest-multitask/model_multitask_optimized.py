from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).parent.parent / "SCDB_2025_01_caseCentered_Citation.csv"

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

TARGETS = [
    "decisionDirection",
    "decisionType",
    "caseDisposition",
    "majVotes",
]


def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)
    df["dateArgumentYear"] = pd.to_datetime(df["dateArgument"], errors="coerce").dt.year
    df = df.dropna(subset=TARGETS + ["dateArgumentYear"])
    X = df[FEATURES + ["dateArgumentYear"]]
    y = df[TARGETS]
    return X, y


def build_preprocessor():
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
    return preprocessor


def train_and_evaluate(random_state: int = 42):
    X, y = load_data()
    print(f"Total samples: {len(X)}")
    print(f"Primary target: {TARGETS[0]}")
    print(f"Auxiliary tasks: {TARGETS[1:]}")
    print(f"\nTarget distributions:")
    for target in TARGETS:
        print(f"  {target}: {len(y[target].unique())} classes")

    years = X["dateArgumentYear"]
    train_mask = years <= 2019
    recent_mask = years >= 2020

    X_train, y_train = X[train_mask], y[train_mask]
    X_recent, y_recent = X[recent_mask], y[recent_mask]

    print(f"\nTrain samples: {len(X_train)} (<=2019)")
    print(f"Recent samples: {len(X_recent)} (>=2020)")

    X_val, X_test, y_val, y_test = train_test_split(
        X_recent,
        y_recent,
        test_size=0.5,
        stratify=y_recent["decisionDirection"],
        random_state=random_state,
    )

    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}\n")

    preprocessor = build_preprocessor()

    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)

    print("Training task-specific models...")
    print("=" * 60)

    models = {}
    results = {}

    task_params = {
        "decisionDirection": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 30,
            "min_samples_leaf": 15,
            "max_features": "sqrt",
        },
        "decisionType": {
            "n_estimators": 150,
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
        },
        "caseDisposition": {
            "n_estimators": 250,
            "max_depth": 20,
            "min_samples_split": 15,
            "min_samples_leaf": 5,
            "max_features": 0.4,
        },
        "majVotes": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 25,
            "min_samples_leaf": 10,
            "max_features": "log2",
        },
    }

    for i, target in enumerate(TARGETS):
        print(f"\nTraining {target}...")

        params = task_params[target]
        model = RandomForestClassifier(
            **params,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        )

        model.fit(X_train_prep, y_train.iloc[:, i])

        train_pred = model.predict(X_train_prep)
        val_pred = model.predict(X_val_prep)

        train_acc = accuracy_score(y_train.iloc[:, i], train_pred)
        val_acc = accuracy_score(y_val.iloc[:, i], val_pred)

        models[target] = model
        results[target] = {"train": train_acc, "val": val_acc}

        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Validation accuracy: {val_acc:.3f}")

    print("\n" + "=" * 60)
    print("MULTI-TASK RESULTS (Task-Specific Models)")
    print("=" * 60)

    for target in TARGETS:
        is_primary = " (PRIMARY)" if target == "decisionDirection" else ""
        print(f"\n{target}{is_primary}:")
        print(f"  Train accuracy: {results[target]['train']:.3f}")
        print(f"  Validation accuracy: {results[target]['val']:.3f}")

    train_acc_avg = np.mean([results[t]["train"] for t in TARGETS])
    val_acc_avg = np.mean([results[t]["val"] for t in TARGETS])

    print(f"\n{'='*60}")
    print(f"AVERAGE ACROSS ALL TASKS:")
    print(f"  Train accuracy: {train_acc_avg:.3f}")
    print(f"  Validation accuracy: {val_acc_avg:.3f}")
    print(f"{'='*60}")

    return preprocessor, models


if __name__ == "__main__":
    train_and_evaluate()
