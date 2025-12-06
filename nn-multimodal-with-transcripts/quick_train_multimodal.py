import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

from model_multimodal import MultiModalSCOTUS

class SimpleDataset(Dataset):
    def __init__(self, n_samples, metadata_dim, tokenizer, mode='multimodal'):
        self.n_samples = n_samples
        self.metadata = torch.randn(n_samples, metadata_dim)
        self.labels = torch.randint(0, 2, (n_samples,))
        self.tokenizer = tokenizer
        self.mode = mode

        self.texts = [
            "The petitioner argues that the statute violates the First Amendment.",
            "The respondent contends that the lower court erred in its interpretation.",
            "Justice Roberts questions whether the precedent applies to this case.",
            "The Solicitor General submits that the Court should reverse the decision."
        ] * (n_samples // 4 + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'metadata': self.metadata[idx],
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': self.labels[idx]
        }


def train_quick():
    print("="*60)
    print("Quick Multi-Modal Model Training Demo")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\nLoading Legal-BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    print("Tokenizer loaded")

    print("\nCreating sample datasets...")
    metadata_dim = 69
    train_dataset = SimpleDataset(100, metadata_dim, tokenizer)
    val_dataset = SimpleDataset(20, metadata_dim, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    print("\nCreating multi-modal model...")
    model = MultiModalSCOTUS(
        metadata_dim=metadata_dim,
        num_classes=2,
        freeze_bert=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print(f"\n{'='*60}")
    print("Training")
    print(f"{'='*60}\n")

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            metadata = batch['metadata'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(metadata, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                metadata = batch['metadata'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(metadata, input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_quick()
