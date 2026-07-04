import torch
import torch.nn as nn


class ImageBackbone(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.proj(z)


class NanoVLMAction(nn.Module):
    def __init__(self, n_actions: int = 4, hidden: int = 128):
        super().__init__()
        self.backbone = ImageBackbone(out_dim=hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, image):
        h = self.backbone(image)
        return self.head(h)


class NanoVLMReason(nn.Module):
    def __init__(self, vocab_size: int, n_actions: int = 4, hidden: int = 128):
        super().__init__()
        self.backbone = ImageBackbone(out_dim=hidden)
        self.action_head = nn.Linear(hidden, n_actions)
        self.emb = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(self, image, reason_ids):
        h = self.backbone(image)
        action_logits = self.action_head(h)
        tok = self.emb(reason_ids)
        init = h.unsqueeze(0)
        out, _ = self.gru(tok, init)
        reason_logits = self.lm_head(out)
        return action_logits, reason_logits

