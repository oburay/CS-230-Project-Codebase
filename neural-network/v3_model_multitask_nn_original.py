from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from justice_ideology import add_ideology_features

DATA_PATH = Path(__file__).parent.parent / "dataset" / "SCDB_2025_01_caseCentered_Citation.csv"

CATEGORICAL_FEATURES = [
    "issue", "issueArea",
    "petitioner", "petitionerState", "respondent", "respondentState",
    "jurisdiction", "adminAction",
    "caseOrigin", "caseOriginState", "caseSource", "caseSourceState",
    "lcDisagreement", "certReason", "lcDisposition", "lcDispositionDirection",
    "declarationUncon",
    "authorityDecision1", "authorityDecision2", "lawType", "lawSupp", "lawMinor",
    "naturalCourt", "chief",
]

NUMERIC_FEATURES = ["term", "median_ideology", "conservative_era", "liberal_era"]

TARGETS = [
    "decisionDirection",
    "decisionType",
    "caseDisposition",
    "majVotes"
]


class MultiTaskDataset(Dataset):

    def __init__(self, categorical_data, numeric_data, labels=None):
        self.categorical_data = categorical_data
        self.numeric_data = numeric_data
        self.labels = labels

    def __len__(self):
        return len(self.categorical_data)

    def __getitem__(self, idx):
        cat_features = torch.LongTensor(self.categorical_data[idx])
        num_features = torch.FloatTensor(self.numeric_data[idx])

        if self.labels is not None:
            labels = [torch.LongTensor([self.labels[i][idx]])[0] for i in range(4)]
            return cat_features, num_features, labels
        return cat_features, num_features


class MultiTaskSCOTUSModel(nn.Module):

    def __init__(self, vocab_sizes, embedding_dims, n_numeric,
                 hidden_dims=[256, 128, 64], dropout=0.3, n_tasks=4):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim)
            for vocab_size, emb_dim in zip(vocab_sizes, embedding_dims)
        ])

        total_embed_dim = sum(embedding_dims)
        input_dim = total_embed_dim + n_numeric

        shared_layers = []

        shared_layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

        for i in range(len(hidden_dims) - 1):
            shared_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        self.shared_network = nn.Sequential(*shared_layers)

        self.task_heads = None

    def set_task_heads(self, n_classes_per_task, hidden_dim):
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_classes) for n_classes in n_classes_per_task
        ])

    def forward(self, categorical, numeric):
        embedded = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)

        x = torch.cat([embedded, numeric], dim=1)

        shared_features = self.shared_network(x)

        outputs = [head(shared_features) for head in self.task_heads]

        return outputs


def load_and_prepare_data():
    print("Loading data...")

    df = pd.read_csv(DATA_PATH)
    df["dateArgumentYear"] = pd.to_datetime(df["dateArgument"], errors="coerce").dt.year
    df = add_ideology_features(df)

    df = df.dropna(subset=TARGETS + ["dateArgumentYear"])

    print(f"Total samples: {len(df)}")

    for col in CATEGORICAL_FEATURES:
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < 5].index
        df[col] = df[col].replace(rare_categories, "RARE_CATEGORY")

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("MISSING").astype(str)
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        encoders[col] = le

    scaler = StandardScaler()
    df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES])

    target_encoders = {}
    target_classes = {}
    for target in TARGETS:
        le = LabelEncoder()
        df[target + "_encoded"] = le.fit_transform(df[target])
        target_encoders[target] = le
        target_classes[target] = len(le.classes_)
        print(f"{target}: {len(le.classes_)} classes")

    encoded_cols = [col + "_encoded" for col in CATEGORICAL_FEATURES]
    cat_data = df[encoded_cols].values
    num_data = df[NUMERIC_FEATURES].values

    labels = [df[target + "_encoded"].values for target in TARGETS]

    years = df["dateArgumentYear"]

    train_mask = years <= 2019
    recent_mask = years >= 2020

    cat_train = cat_data[train_mask]
    num_train = num_data[train_mask]
    labels_train = [label[train_mask] for label in labels]

    cat_recent = cat_data[recent_mask]
    num_recent = num_data[recent_mask]
    labels_recent = [label[recent_mask] for label in labels]

    print(f"Training samples: {len(cat_train)} (≤2019)")
    print(f"Recent samples: {len(cat_recent)} (≥2020)")

    indices = np.arange(len(cat_recent))
    val_idx, test_idx = train_test_split(
        indices, test_size=0.5, random_state=42,
        stratify=labels_recent[0]
    )

    cat_val = cat_recent[val_idx]
    cat_test = cat_recent[test_idx]
    num_val = num_recent[val_idx]
    num_test = num_recent[test_idx]
    labels_val = [label[val_idx] for label in labels_recent]
    labels_test = [label[test_idx] for label in labels_recent]

    print(f"Validation samples: {len(cat_val)}")
    print(f"Test samples: {len(cat_test)}")

    vocab_sizes = [len(encoders[col].classes_) for col in CATEGORICAL_FEATURES]
    embedding_dims = [min(20, (size + 1) // 2) for size in vocab_sizes]

    return {
        'train': (cat_train, num_train, labels_train),
        'val': (cat_val, num_val, labels_val),
        'test': (cat_test, num_test, labels_test),
        'vocab_sizes': vocab_sizes,
        'embedding_dims': embedding_dims,
        'target_classes': target_classes,
        'encoders': encoders,
        'target_encoders': target_encoders,
        'scaler': scaler,
    }


def train_multitask_model(hidden_dims=[256, 128, 64], dropout=0.3,
                          learning_rate=0.001, weight_decay=1e-4,
                          batch_size=64, epochs=100, patience=15):

    data = load_and_prepare_data()

    train_dataset = MultiTaskDataset(*data['train'])
    val_dataset = MultiTaskDataset(*data['val'])
    test_dataset = MultiTaskDataset(*data['test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTaskSCOTUSModel(
        vocab_sizes=data['vocab_sizes'],
        embedding_dims=data['embedding_dims'],
        n_numeric=len(NUMERIC_FEATURES),
        hidden_dims=hidden_dims,
        dropout=dropout,
        n_tasks=4
    )

    n_classes_per_task = [data['target_classes'][t] for t in TARGETS]
    model.set_task_heads(n_classes_per_task, hidden_dims[-1])

    print(f"\nModel architecture:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Shared layers: {hidden_dims}")
    print(f"  Task heads: {n_classes_per_task}")

    criterions = [nn.CrossEntropyLoss() for _ in range(4)]

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = 0
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_acc_primary': []
    }

    print("\nTraining...")
    for epoch in range(epochs):
        model.train()
        train_losses = [0.0] * 4

        for cat_batch, num_batch, labels_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(cat_batch, num_batch)

            losses = [criterions[i](outputs[i], labels_batch[i]) for i in range(4)]

            total_loss = sum(losses)

            total_loss.backward()
            optimizer.step()

            for i in range(4):
                train_losses[i] += losses[i].item()

        model.eval()
        val_preds = [[] for _ in range(4)]
        val_labels = [[] for _ in range(4)]

        with torch.no_grad():
            for cat_batch, num_batch, labels_batch in val_loader:
                outputs = model(cat_batch, num_batch)

                for i in range(4):
                    preds = outputs[i].argmax(dim=1).numpy()
                    val_preds[i].extend(preds)
                    val_labels[i].extend(labels_batch[i].numpy())

        val_accs = [accuracy_score(val_labels[i], val_preds[i]) for i in range(4)]
        primary_acc = val_accs[0]

        history['train_loss'].append(sum(train_losses) / len(train_loader))
        history['val_acc_primary'].append(primary_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            for i, target in enumerate(TARGETS):
                print(f"  {target}: {val_accs[i]:.4f}", end="")
                if i == 0:
                    print(" (PRIMARY)", end="")
                print()

        if primary_acc > best_val_acc:
            best_val_acc = primary_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_sizes': data['vocab_sizes'],
                'embedding_dims': data['embedding_dims'],
                'hidden_dims': hidden_dims,
                'dropout': dropout,
                'target_classes': data['target_classes'],
            }, Path(__file__).parent / "best_multitask_nn.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    import os
    os.makedirs('plots', exist_ok=True)

    checkpoint = torch.load(Path(__file__).parent / "best_multitask_nn.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    model.eval()
    val_preds = [[] for _ in range(4)]
    val_labels = [[] for _ in range(4)]

    with torch.no_grad():
        for cat_batch, num_batch, labels_batch in val_loader:
            outputs = model(cat_batch, num_batch)
            for i in range(4):
                preds = outputs[i].argmax(dim=1).numpy()
                val_preds[i].extend(preds)
                val_labels[i].extend(labels_batch[i].numpy())

    print("\nValidation Set:")
    for i, target in enumerate(TARGETS):
        acc = accuracy_score(val_labels[i], val_preds[i])
        marker = " (PRIMARY)" if i == 0 else ""
        print(f"  {target}{marker}: {acc:.4f}")

    test_preds = [[] for _ in range(4)]
    test_labels = [[] for _ in range(4)]

    with torch.no_grad():
        for cat_batch, num_batch, labels_batch in test_loader:
            outputs = model(cat_batch, num_batch)
            for i in range(4):
                preds = outputs[i].argmax(dim=1).numpy()
                test_preds[i].extend(preds)
                test_labels[i].extend(labels_batch[i].numpy())

    print("\nTest Set:")
    for i, target in enumerate(TARGETS):
        acc = accuracy_score(test_labels[i], test_preds[i])
        marker = " (PRIMARY)" if i == 0 else ""
        print(f"  {target}{marker}: {acc:.4f}")

    print("\n" + "="*80)
    print("PRIMARY TASK (decisionDirection) - Detailed Report")
    print("="*80)
    class_names = [str(c) for c in data['target_encoders'][TARGETS[0]].classes_]
    print(classification_report(
        test_labels[0], test_preds[0],
        target_names=class_names,
        digits=4
    ))

    return model, data


if __name__ == "__main__":
    model, data = train_multitask_model()
