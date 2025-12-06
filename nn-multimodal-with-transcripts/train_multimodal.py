"""
Training Pipeline for Multi-Modal SCOTUS Model
Includes data loading, training, and evaluation for all three model variants
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model_multimodal import MultiModalSCOTUS, MetadataOnlyModel, TextOnlyModel


class SCOTUSMultiModalDataset(Dataset):
    """Dataset for multi-modal SCOTUS prediction"""

    def __init__(
        self,
        transcript_data: pd.DataFrame,
        metadata_features: np.ndarray,
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        mode: str = 'multimodal'  # 'multimodal', 'metadata_only', 'text_only'
    ):
        self.transcript_data = transcript_data.reset_index(drop=True)
        self.metadata_features = metadata_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get label
        label = self.labels[idx]

        output = {'label': torch.tensor(label, dtype=torch.long)}

        # Add metadata if needed
        if self.mode in ['multimodal', 'metadata_only']:
            metadata = torch.tensor(
                self.metadata_features[idx],
                dtype=torch.float32
            )
            output['metadata'] = metadata

        # Add text if needed
        if self.mode in ['multimodal', 'text_only']:
            # Get transcript
            transcript = self.transcript_data.iloc[idx]['transcript']

            # Tokenize
            encoding = self.tokenizer(
                transcript,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            output['input_ids'] = encoding['input_ids'].squeeze(0)
            output['attention_mask'] = encoding['attention_mask'].squeeze(0)

        return output


def load_multimodal_data(
    transcript_path: str,
    scdb_path: str,
    features_path: str
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load and merge transcript data with SCDB metadata

    Returns:
        transcript_df: DataFrame with transcripts
        features: Numpy array of metadata features
        labels: Numpy array of labels
    """
    print("Loading data...")

    # Load transcripts
    with open(transcript_path, 'r') as f:
        transcripts = json.load(f)
    transcript_df = pd.DataFrame(transcripts)

    # Load SCDB data
    scdb = pd.read_csv(scdb_path, encoding='latin1')

    # Load preprocessed features
    data = np.load(features_path)
    all_features = data['X']
    all_labels = data['y']
    case_ids = data['case_ids']

    # Merge on caseId
    transcript_df = transcript_df.merge(
        pd.DataFrame({'caseId': case_ids}),
        on='caseId',
        how='inner'
    )

    # Get corresponding features and labels
    indices = [np.where(case_ids == cid)[0][0] for cid in transcript_df['caseId']]
    features = all_features[indices]
    labels = all_labels[indices]

    print(f"Loaded {len(transcript_df)} cases with transcripts and metadata")
    print(f"Feature shape: {features.shape}")
    print(f"Label distribution: {np.bincount(labels)}")

    return transcript_df, features, labels


def temporal_split_multimodal(
    transcript_df: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    train_cutoff: int = 2016,
    val_cutoff: int = 2020
) -> Dict[str, Tuple]:
    """
    Temporal split for multi-modal data

    Returns:
        Dictionary with train/val/test splits
    """
    # Get terms
    terms = transcript_df['term'].values

    # Create splits
    train_mask = terms < train_cutoff
    val_mask = (terms >= train_cutoff) & (terms < val_cutoff)
    test_mask = terms >= val_cutoff

    splits = {
        'train': (
            transcript_df[train_mask].reset_index(drop=True),
            features[train_mask],
            labels[train_mask]
        ),
        'val': (
            transcript_df[val_mask].reset_index(drop=True),
            features[val_mask],
            labels[val_mask]
        ),
        'test': (
            transcript_df[test_mask].reset_index(drop=True),
            features[test_mask],
            labels[test_mask]
        )
    }

    print(f"\nTemporal split:")
    print(f"Train: {len(splits['train'][0])} cases (< {train_cutoff})")
    print(f"Val:   {len(splits['val'][0])} cases ({train_cutoff}-{val_cutoff})")
    print(f"Test:  {len(splits['test'][0])} cases (>= {val_cutoff})")

    return splits


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str = 'multimodal'
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        labels = batch['label'].to(device)

        # Forward pass
        if mode == 'multimodal':
            metadata = batch['metadata'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(metadata, input_ids, attention_mask)

        elif mode == 'metadata_only':
            metadata = batch['metadata'].to(device)
            outputs = model(metadata)

        elif mode == 'text_only':
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: str = 'multimodal'
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            labels = batch['label'].to(device)

            # Forward pass
            if mode == 'multimodal':
                metadata = batch['metadata'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(metadata, input_ids, attention_mask)

            elif mode == 'metadata_only':
                metadata = batch['metadata'].to(device)
                outputs = model(metadata)

            elif mode == 'text_only':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    mode: str,
    save_path: Optional[str] = None
) -> Dict[str, List]:
    """Complete training loop"""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }

    best_val_acc = 0

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, mode)
        history['train_loss'].append(train_loss)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, mode)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f}")
        print(f"Val Acc:    {val_metrics['accuracy']:.4f}")
        print(f"Val F1:     {val_metrics['f1']:.4f}")

        # Learning rate scheduling
        scheduler.step(val_metrics['accuracy'])

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"✅ Saved best model (acc={best_val_acc:.4f})")

    return history


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[1].plot(history['val_f1'], label='Val F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None
):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Liberal', 'Conservative'],
        yticklabels=['Liberal', 'Conservative']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Configuration
    TRANSCRIPT_PATH = "data/oyez_transcripts.json"
    SCDB_PATH = "data/SCDB_2025_01_caseCentered_Citation.csv"
    FEATURES_PATH = "data/features_engineered.npz"

    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load tokenizer
    print("Loading Legal-BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

    # Load data
    transcript_df, features, labels = load_multimodal_data(
        TRANSCRIPT_PATH, SCDB_PATH, FEATURES_PATH
    )

    # Temporal split
    splits = temporal_split_multimodal(transcript_df, features, labels)

    print("\n✅ Data loaded successfully!")
    print(f"Ready to train multi-modal models")
