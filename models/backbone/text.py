import torch
from torch import nn


class TextEncoder(nn.Module):
    """UTF-8 byte Transformer reference, or an explicitly supplied HF model path."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.text_model:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.text_model, local_files_only=True
            )
            self.encoder = AutoModel.from_pretrained(
                cfg.text_model, local_files_only=True
            )
            self.project = nn.Linear(self.encoder.config.hidden_size, cfg.dim)
            if cfg.freeze_text:
                self.encoder.requires_grad_(False)
        else:
            self.embed = nn.Embedding(257, cfg.dim, padding_idx=0)
            self.position = nn.Embedding(cfg.max_tokens, cfg.dim)
            layer = nn.TransformerEncoderLayer(
                cfg.dim, 4, 4 * cfg.dim, dropout=0.0, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, texts):
        device = next(self.parameters()).device
        if self.cfg.text_model:
            if self.cfg.freeze_text:
                self.encoder.eval()
            token = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_tokens,
            ).to(device)
            return (
                self.project(self.encoder(**token).last_hidden_state),
                token["attention_mask"].bool(),
            )
        ids = [[b + 1 for b in s.encode("utf-8")[: self.cfg.max_tokens]] for s in texts]
        if any(not x for x in ids):
            raise ValueError("Query text cannot be empty")
        token = torch.zeros(
            len(ids), max(map(len, ids)), dtype=torch.long, device=device
        )
        for i, row in enumerate(ids):
            token[i, : len(row)] = torch.tensor(row, device=device)
        valid = token != 0
        x = self.embed(token) + self.position(
            torch.arange(token.shape[1], device=device)
        )
        return self.encoder(x, src_key_padding_mask=~valid), valid
