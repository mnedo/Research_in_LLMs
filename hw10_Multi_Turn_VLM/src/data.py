from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class ActionOnlyDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x = torch.tensor(r.image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        y = torch.tensor(r.action, dtype=torch.long)
        return {"image": x, "action": y}


class ActionReasonDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tok = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x = torch.tensor(r.image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        action = torch.tensor(r.action, dtype=torch.long)
        prompt = f"reason: {r.reason} action:"
        ids = self.tok(prompt)["input_ids"]
        ids = torch.tensor(ids, dtype=torch.long)
        return {"image": x, "action": action, "reason_ids": ids}


class TinyTokenizer:
    def __init__(self):
        vocab = ["<pad>", "reason:", "action:", "иду", "влево", "вниз", "вправо", "вверх", "к", "безопасной", "клетке", "цели", "обхожу", "яму", "не", "рискую"]
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        self.pad_id = 0

    def __len__(self):
        return len(self.stoi)

    def __call__(self, text: str) -> Dict[str, List[int]]:
        out = []
        for w in text.split():
            out.append(self.stoi.get(w, self.pad_id))
        return {"input_ids": out}


def split_records(records, val_split: float = 0.15, seed: int = 42):
    n = len(records)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(n * (1.0 - val_split))
    tr = [records[i] for i in idx[:cut]]
    va = [records[i] for i in idx[cut:]]
    return tr, va


def collate_reason(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    actions = torch.stack([x["action"] for x in batch], dim=0)
    lens = [len(x["reason_ids"]) for x in batch]
    max_len = max(lens)
    ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    for i, x in enumerate(batch):
        cur = x["reason_ids"]
        ids[i, : len(cur)] = cur
    return {"image": images, "action": actions, "reason_ids": ids}

