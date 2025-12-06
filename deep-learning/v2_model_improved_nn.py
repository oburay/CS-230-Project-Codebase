from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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

TARGET = "decisionDirection"


class SCOTUSDataset(Dataset):

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
            label = torch.LongTensor([self.labels[idx]])[0]
            return cat_features, num_features, label
        return cat_features, num_features


class ImprovedSCOTUSModel(nn.Module):

    def __init__(self, vocab_sizes, embedding_dims, n_numeric, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim)
            for vocab_size, emb_dim in zip(vocab_sizes, embedding_dims)
        ])

        total_embed_dim = sum(embedding_dims)

        layers = []

        input_dim = total_embed_dim + n_numeric
        layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout * 0.7),
            ])

        layers.append(nn.Linear(hidden_dims[-1], 2))

        self.network = nn.Sequential(*layers)

    def forward(self, categorical_features, numeric_features):
        embeddings = [
            emb(categorical_features[:, i])
            for i, emb in enumerate(self.embeddings)
        ]

        x = torch.cat(embeddings + [numeric_features], dim=1)

        return self.network(x)


def load_and_encode_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)

    df["dateArgumentYear"] = pd.to_datetime(df["dateArgument"], errors="coerce").dt.year
    df = df.dropna(subset=[TARGET, "dateArgumentYear"])

    df["term"] = df["term"].fillna(df["dateArgumentYear"])

    df = add_ideology_features(df)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("MISSING")

    for col in CATEGORICAL_FEATURES:
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < 5].index
        df[col] = df[col].replace(rare_categories, "RARE_CATEGORY")

    label_encoders = {}
    encoded_features = []
    vocab_sizes = []

    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].astype(str))
        encoded_features.append(encoded)
        vocab_sizes.append(len(le.classes_))
        label_encoders[col] = le

    categorical_data = np.column_stack(encoded_features)

    numeric_data = df[NUMERIC_FEATURES].values

    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(numeric_data)

    y = (df[TARGET].values == 2).astype(int)

    return categorical_data, numeric_data, y, vocab_sizes, label_encoders, scaler, df["dateArgumentYear"].values


def calculate_embedding_dims(vocab_sizes, max_dim=20):
    return [min(max_dim, (v + 1) // 2) for v in vocab_sizes]


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for cat_features, num_features, labels in loader:
        cat_features = cat_features.to(device)
        num_features = num_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(cat_features, num_features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for cat_features, num_features, labels in loader:
            cat_features = cat_features.to(device)
            num_features = num_features.to(device)
            labels = labels.to(device)

            outputs = model(cat_features, num_features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels, np.array(all_probs)


def train_and_evaluate(
    hidden_dims=[256, 128, 64],
    dropout=0.3,
    learning_rate=0.001,
    weight_decay=1e-4,
    batch_size=64,
    n_epochs=100,
    patience=15,
    random_state=42
):
    """Trains and evaluates the improved neural network model."""

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Loading and encoding data...")
    categorical_data, numeric_data, y, vocab_sizes, label_encoders, scaler, years = load_and_encode_data()

    print(f"Total samples: {len(y)}")
    print(f"Features: {len(CATEGORICAL_FEATURES)} categorical + {len(NUMERIC_FEATURES)} numeric")
    print(f"Numeric features: {NUMERIC_FEATURES}")
    print(f"Vocabulary sizes (after grouping rare): {vocab_sizes}")
    print(f"Target distribution: {np.bincount(y)}\n")

    train_mask = years <= 2019
    recent_mask = years >= 2020

    cat_train, cat_recent = categorical_data[train_mask], categorical_data[recent_mask]
    num_train, num_recent = numeric_data[train_mask], numeric_data[recent_mask]
    y_train, y_recent = y[train_mask], y[recent_mask]

    cat_val, cat_test, num_val, num_test, y_val, y_test = train_test_split(
        cat_recent, num_recent, y_recent,
        test_size=0.5,
        random_state=random_state,
        stratify=y_recent
    )

    print(f"Train samples: {len(y_train)} (<=2019)")
    print(f"Validation samples: {len(y_val)} (>=2020)")
    print(f"Test samples: {len(y_test)} (>=2020)\n")

    train_dataset = SCOTUSDataset(cat_train, num_train, y_train)
    val_dataset = SCOTUSDataset(cat_val, num_val, y_val)
    test_dataset = SCOTUSDataset(cat_test, num_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    embedding_dims = calculate_embedding_dims(vocab_sizes, max_dim=20)
    print(f"Embedding dimensions (capped at 20): {embedding_dims}\n")

    model = ImprovedSCOTUSModel(
        vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        n_numeric=len(NUMERIC_FEATURES),
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print("Training...")
    print("=" * 80)

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab_sizes': vocab_sizes,
                'embedding_dims': embedding_dims,
                'label_encoders': label_encoders,
                'scaler': scaler,
            }, 'best_model_improved.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print("=" * 80)

    checkpoint = torch.load('best_model_improved.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\n" + "=" * 80)
    print("Validation Set Results")
    print("=" * 80)
    _, val_acc, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        val_labels, val_preds,
        target_names=["Conservative", "Liberal"],
        digits=4
    ))

    print("\n" + "=" * 80)
    print("Test Set Results")
    print("=" * 80)
    _, test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=["Conservative", "Liberal"],
        digits=4
    ))

    return model, label_encoders, scaler, test_probs, test_labels


if __name__ == "__main__":
    model, encoders, scaler, test_probs, test_labels = train_and_evaluate(
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        weight_decay=1e-4,
        patience=15
    )
