import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import json
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "SCDB_2025_01_caseCentered_Citation.csv"
TRANSCRIPT_PATH = BASE_DIR / "dataset" / "oyez_transcripts.json"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CATEGORICAL_FEATURES = [
    'jurisdiction', 'adminAction', 'caseOrigin', 'caseSource',
    'lcDisposition', 'lcDispositionDirection', 'certReason', 'lcDisagreement',
    'issue', 'issueArea', 'petitioner', 'petitionerState',
    'respondent', 'respondentState', 'authorityDecision1',
    'authorityDecision2', 'lawType', 'lawSupp', 'lawMinor',
    'naturalCourt', 'chief'
]

NUMERIC_FEATURES = ['term', 'dateArgumentYear']


def add_ideology_features(df):
    df = df.copy()

    def compute_median_ideology(term):
        if term >= 2020: return 0.50
        elif term >= 2018: return 0.40
        elif term >= 2017: return 0.35
        elif term >= 2006: return 0.20
        elif term >= 1986: return 0.60
        else: return 0.0

    df['median_ideology'] = df['term'].apply(compute_median_ideology)
    df['conservative_era'] = (df['term'] >= 2017).astype(int)
    df['liberal_era'] = (df['term'] < 1986).astype(int)

    return df


class MetadataEncoder(nn.Module):

    def __init__(self, metadata_dim, hidden_dims=[256, 128]):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.encoder(x)


class TextEncoder(nn.Module):

    def __init__(self, model_name='nlpaueb/legal-bert-base-uncased', freeze_bert=True):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.dropout(pooled_output)


class MultiModalSCOTUS(nn.Module):

    def __init__(self, metadata_dim, num_classes=2, freeze_bert=True):
        super().__init__()

        self.metadata_encoder = MetadataEncoder(metadata_dim, hidden_dims=[256, 128])
        self.text_encoder = TextEncoder(freeze_bert=freeze_bert)

        fusion_input_dim = 128 + 768

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, metadata, input_ids, attention_mask):
        metadata_features = self.metadata_encoder(metadata)
        text_features = self.text_encoder(input_ids, attention_mask)

        combined = torch.cat([metadata_features, text_features], dim=1)
        logits = self.fusion(combined)

        return logits


class SCOTUSMultiModalDataset(Dataset):

    def __init__(self, metadata_features, labels, transcript_df, tokenizer, max_length=512):
        self.metadata_features = metadata_features
        self.labels = labels
        self.transcript_df = transcript_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        metadata = torch.tensor(self.metadata_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        transcript = self.transcript_df.iloc[idx]['transcript']

        encoding = self.tokenizer(
            transcript,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'metadata': metadata,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }


def load_and_prepare_data():

    print("Loading SCDB data...")
    df = pd.read_csv(DATA_PATH, encoding='latin1')

    df = df[df['decisionDirection'].isin([1, 2])].copy()

    df['dateArgumentYear'] = pd.to_datetime(df['dateArgument'], errors='coerce').dt.year
    df = df.dropna(subset=['dateArgumentYear'])

    df = add_ideology_features(df)

    print(f"Total SCDB cases with decisionDirection: {len(df)}")

    print("Loading oral argument transcripts...")
    with open(TRANSCRIPT_PATH, 'r') as f:
        transcripts = json.load(f)

    transcript_df = pd.DataFrame(transcripts)
    print(f"Total transcripts available: {len(transcript_df)}")
    print(f"Transcript years: {transcript_df['term'].min()}-{transcript_df['term'].max()}")

    df['docket'] = df['docket'].astype(str).str.strip()
    transcript_df['docket'] = transcript_df['docket'].astype(str).str.strip()

    merged = df.merge(
        transcript_df[['term', 'docket', 'transcript']],
        on=['term', 'docket'],
        how='inner'
    )

    print(f"\nCases with both metadata + transcripts: {len(merged)}")
    print(f"Year range: {merged['term'].min()}-{merged['term'].max()}")

    print("\nPreparing features...")

    for col in CATEGORICAL_FEATURES:
        merged[col] = merged[col].fillna('MISSING').astype(str)

    for col in NUMERIC_FEATURES:
        merged[col] = merged[col].fillna(merged[col].median())

    X_cat_list = []
    for col in CATEGORICAL_FEATURES:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(merged[[col]])
        X_cat_list.append(encoded)

    X_cat = np.hstack(X_cat_list)

    X_num = merged[NUMERIC_FEATURES + ['median_ideology', 'conservative_era', 'liberal_era']].values

    X = np.hstack([X_cat, X_num])

    y = (merged['decisionDirection'].values == 2).astype(int)

    print(f"Feature dimension: {X.shape[1]}")
    print(f"Label distribution: Conservative={np.sum(y==0)}, Liberal={np.sum(y==1)}")

    return X, y, merged[['term', 'docket', 'transcript']].reset_index(drop=True)


def temporal_split(X, y, transcript_df, train_cutoff=2019):

    terms = transcript_df['term'].values

    train_mask = terms <= train_cutoff
    test_mask = terms > train_cutoff

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    transcript_train = transcript_df[train_mask].reset_index(drop=True)
    transcript_test = transcript_df[test_mask].reset_index(drop=True)

    n_test = len(X_test)
    val_indices = np.random.RandomState(42).permutation(n_test)[:n_test//2]
    test_indices = np.setdiff1d(np.arange(n_test), val_indices)

    X_val, X_test = X_test[val_indices], X_test[test_indices]
    y_val, y_test = y_test[val_indices], y_test[test_indices]
    transcript_val = transcript_test.iloc[val_indices].reset_index(drop=True)
    transcript_test = transcript_test.iloc[test_indices].reset_index(drop=True)

    print(f"\nTemporal Split:")
    print(f"  Train: {len(X_train)} cases (≤{train_cutoff})")
    print(f"  Val:   {len(X_val)} cases (>{train_cutoff}, 50%)")
    print(f"  Test:  {len(X_test)} cases (>{train_cutoff}, 50%)")

    return (X_train, y_train, transcript_train), (X_val, y_val, transcript_val), (X_test, y_test, transcript_test)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        metadata = batch['metadata'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(metadata, input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            metadata = batch['metadata'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(metadata, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    print("="*80)
    print("v5: Multi-Modal Neural Network (Metadata + Oral Arguments)")
    print("="*80)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    X, y, transcript_df = load_and_prepare_data()

    (X_train, y_train, transcript_train), \
    (X_val, y_val, transcript_val), \
    (X_test, y_test, transcript_test) = temporal_split(X, y, transcript_df)

    print("\nLoading Legal-BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    print("✅ Tokenizer loaded")

    print("\nCreating datasets...")
    train_dataset = SCOTUSMultiModalDataset(X_train, y_train, transcript_train, tokenizer, max_length=512)
    val_dataset = SCOTUSMultiModalDataset(X_val, y_val, transcript_val, tokenizer, max_length=512)
    test_dataset = SCOTUSMultiModalDataset(X_test, y_test, transcript_test, tokenizer, max_length=512)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    print(f"✅ Datasets created")

    print("\nCreating multi-modal model...")
    model = MultiModalSCOTUS(
        metadata_dim=X_train.shape[1],
        num_classes=2,
        freeze_bert=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    print(f"\n{'='*80}")
    print("Training")
    print(f"{'='*80}\n")

    num_epochs = 10
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f" New best validation accuracy: {best_val_acc:.4f}")

    print(f"\n{'='*80}")
    print("Final Test Evaluation")
    print(f"{'='*80}\n")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    print(f"\n{'='*80}")
    print("SUMMARY - v5 Multi-Modal NN")
    print(f"{'='*80}")
    print(f"Architecture: Metadata Encoder + Legal-BERT + Fusion")
    print(f"Training data: {len(X_train)} cases (≤2019)")
    print(f"Validation data: {len(X_val)} cases (≥2020)")
    print(f"Test data: {len(X_test)} cases (≥2020)")
    print(f"\nBest Val Accuracy: {best_val_acc:.1%}")
    print(f"Final Test Accuracy: {test_acc:.1%}")
    print(f"\nBaseline (v2 metadata-only): 71.1%")
    print(f"Improvement: {(test_acc - 0.711):.1%}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
