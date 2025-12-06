import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, Optional


class MetadataEncoder(nn.Module):
    def __init__(self, metadata_dim: int, hidden_dims: list = [256, 128]):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = 'nlpaueb/legal-bert-base-uncased',
        freeze_bert: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output

        return self.dropout(pooled_output)

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class MultiModalSCOTUS(nn.Module):
    def __init__(
        self,
        metadata_dim: int,
        num_classes: int = 2,
        metadata_hidden: list = [256, 128],
        bert_model: str = 'nlpaueb/legal-bert-base-uncased',
        freeze_bert: bool = True,
        fusion_hidden: list = [512, 256],
        dropout: float = 0.3
    ):
        super().__init__()

        self.metadata_encoder = MetadataEncoder(
            metadata_dim=metadata_dim,
            hidden_dims=metadata_hidden
        )

        self.text_encoder = TextEncoder(
            model_name=bert_model,
            freeze_bert=freeze_bert,
            dropout=dropout
        )

        fusion_input_dim = metadata_hidden[-1] + 768

        fusion_layers = []
        in_dim = fusion_input_dim

        for hidden_dim in fusion_hidden:
            fusion_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        fusion_layers.append(nn.Linear(in_dim, num_classes))

        self.fusion = nn.Sequential(*fusion_layers)

    def forward(
        self,
        metadata: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        metadata_features = self.metadata_encoder(metadata)
        text_features = self.text_encoder(input_ids, attention_mask)
        combined = torch.cat([metadata_features, text_features], dim=1)
        logits = self.fusion(combined)
        return logits

    def unfreeze_bert(self):
        self.text_encoder.unfreeze_bert()


class MetadataOnlyModel(nn.Module):
    def __init__(
        self,
        metadata_dim: int,
        num_classes: int = 2,
        hidden_dims: list = [256, 128, 64]
    ):
        super().__init__()

        layers = []
        in_dim = metadata_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.network(metadata)


class TextOnlyModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        bert_model: str = 'nlpaueb/legal-bert-base-uncased',
        freeze_bert: bool = True,
        hidden_dims: list = [256, 128]
    ):
        super().__init__()

        self.text_encoder = TextEncoder(
            model_name=bert_model,
            freeze_bert=freeze_bert
        )

        layers = []
        in_dim = 768

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        text_features = self.text_encoder(input_ids, attention_mask)
        return self.classifier(text_features)

    def unfreeze_bert(self):
        self.text_encoder.unfreeze_bert()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


if __name__ == "__main__":
    print("Testing Multi-Modal Models\n")

    batch_size = 4
    metadata_dim = 69
    seq_len = 512
    num_classes = 2

    metadata = torch.randn(batch_size, metadata_dim)
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    print("1. Multi-Modal Model (Metadata + Text)")
    print("-" * 50)
    model = MultiModalSCOTUS(metadata_dim=metadata_dim)
    output = model(metadata, input_ids, attention_mask)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters: {params['frozen']:,}")
    print()

    print("2. Metadata-Only Model (Baseline)")
    print("-" * 50)
    model_meta = MetadataOnlyModel(metadata_dim=metadata_dim)
    output_meta = model_meta(metadata)
    params_meta = count_parameters(model_meta)
    print(f"Output shape: {output_meta.shape}")
    print(f"Total parameters: {params_meta['total']:,}")
    print(f"Trainable parameters: {params_meta['trainable']:,}")
    print()

    print("3. Text-Only Model (BERT)")
    print("-" * 50)
    model_text = TextOnlyModel()
    output_text = model_text(input_ids, attention_mask)
    params_text = count_parameters(model_text)
    print(f"Output shape: {output_text.shape}")
    print(f"Total parameters: {params_text['total']:,}")
    print(f"Trainable parameters: {params_text['trainable']:,}")
    print(f"Frozen parameters: {params_text['frozen']:,}")
    print()

    print("All models working correctly!")
