from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier

DATA_PATH = Path(__file__).parent.parent / "SCDB_2025_01_caseCentered_Citation.csv"

# Pre-argument features only, focused on likely predictors
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

# Multi-task targets (all related to case outcome)
TARGETS = [
    "decisionDirection",        # Conservative (1) vs Liberal (2) - primary target
    "decisionType",             # Type of decision (opinions, per curiam, etc.)
    "caseDisposition",          # Final disposition of case
    "majVotes",                 # Number of votes in majority coalition
]


def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)
    # derive year of argument; drop rows missing ANY target
    df["dateArgumentYear"] = pd.to_datetime(df["dateArgument"], errors="coerce").dt.year
    df = df.dropna(subset=TARGETS + ["dateArgumentYear"])
    X = df[FEATURES + ["dateArgumentYear"]]
    y = df[TARGETS]
    return X, y


def build_preprocessor():
    cat_features = FEATURES  # treat all base features as categorical
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


def build_pipeline():
    preprocessor = build_preprocessor()

    # Multi-task: one Random Forest per target, sharing same features
    # Using best params from single-task model to prevent overfitting
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )

    multi_model = MultiOutputClassifier(base_model, n_jobs=-1)

    return Pipeline([
        ("prep", preprocessor),
        ("model", multi_model),
    ])


def train_and_evaluate(random_state: int = 42):
    X, y = load_data()
    print(f"Total samples: {len(X)}")
    print(f"Targets: {TARGETS}")
    print(f"Target distributions:")
    for target in TARGETS:
        print(f"  {target}: {y[target].value_counts().to_dict()}")

    # Date-based split: train <=2019; validation/test randomly split from 2020+ cases (equal size).
    years = X["dateArgumentYear"]
    train_mask = years <= 2019
    recent_mask = years >= 2020

    X_train, y_train = X[train_mask], y[train_mask]
    X_recent, y_recent = X[recent_mask], y[recent_mask]

    print(f"\nTrain samples: {len(X_train)} (<=2019)")
    print(f"Recent samples: {len(X_recent)} (>=2020)")

    # Split recent cases into equal-sized validation and test sets.
    X_val, X_test, y_val, y_test = train_test_split(
        X_recent,
        y_recent,
        test_size=0.5,
        random_state=random_state,
    )

    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}\n")

    pipeline = build_pipeline()

    # Train without hyperparameter search for faster results
    # (Can add custom multi-output scorer later if needed)
    print("Training multi-task model...")
    pipeline.fit(X_train, y_train)
    print("Training complete!\n")

    # Evaluate each task separately
    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)

    print("=" * 60)
    print("MULTI-TASK RESULTS")
    print("=" * 60)

    for i, target in enumerate(TARGETS):
        train_acc = accuracy_score(y_train.iloc[:, i], train_pred[:, i])
        val_acc = accuracy_score(y_val.iloc[:, i], val_pred[:, i])
        print(f"\n{target}:")
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Validation accuracy: {val_acc:.3f}")

    # Overall average
    train_acc_avg = sum(
        accuracy_score(y_train.iloc[:, i], train_pred[:, i])
        for i in range(len(TARGETS))
    ) / len(TARGETS)
    val_acc_avg = sum(
        accuracy_score(y_val.iloc[:, i], val_pred[:, i])
        for i in range(len(TARGETS))
    ) / len(TARGETS)

    print(f"\n{'='*60}")
    print(f"AVERAGE ACROSS ALL TASKS:")
    print(f"  Train accuracy: {train_acc_avg:.3f}")
    print(f"  Validation accuracy: {val_acc_avg:.3f}")
    print(f"{'='*60}")

    # Uncomment to evaluate on held-out test set after picking hyperparameters.
    # print("\n" + "="*60)
    # print("TEST SET RESULTS (held-out)")
    # print("="*60)
    # test_pred = pipeline.predict(X_test)
    # for i, target in enumerate(TARGETS):
    #     test_acc = accuracy_score(y_test.iloc[:, i], test_pred[:, i])
    #     print(f"\n{target}:")
    #     print(f"  Test accuracy: {test_acc:.3f}")
    #     print(classification_report(y_test.iloc[:, i], test_pred[:, i], digits=3))

    return pipeline


if __name__ == "__main__":
    train_and_evaluate()
