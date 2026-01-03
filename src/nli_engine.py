# src/nli_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class NLIResult:
    label: str
    confidence: float
    probs: Dict[str, float]

class NLIEdge:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

        # ✅ Quan trọng: tránh lỗi "position_embeddings index out of range" của RoBERTa/PhoBERT
        # Thực tế hay phải dùng max_position_embeddings - 2 vì offset padding_idx/position_ids.
        max_pos = int(getattr(self.model.config, "max_position_embeddings", 512))
        self.safe_max_len = max(8, min(512, max_pos - 2))

    @torch.inference_mode()
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        premise = (premise or "").strip()
        hypothesis = (hypothesis or "").strip()

        if not premise or not hypothesis:
            return NLIResult(label="neutral", confidence=0.0, probs={"neutral": 1.0})

        enc = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.safe_max_len,
            padding=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits.squeeze(0)  # (num_labels,)
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        label_probs = {self.id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
        best_label = max(label_probs, key=label_probs.get)

        return NLIResult(label=best_label, confidence=label_probs[best_label], probs=label_probs)
