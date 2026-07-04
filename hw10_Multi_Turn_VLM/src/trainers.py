import torch
import torch.nn.functional as F


def run_sft_action(model, loader, val_loader, cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best = 0.0
    hist = []
    for _ in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for b in loader:
            x = b["image"].to(device)
            y = b["action"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        acc = eval_action(model, val_loader, device)
        best = max(best, acc)
        hist.append({"train_loss": tr_loss / max(1, len(loader)), "val_acc": acc})
    return hist, best


def run_sft_reason(model, loader, val_loader, cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    hist = []
    for _ in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for b in loader:
            x = b["image"].to(device)
            y = b["action"].to(device)
            ids = b["reason_ids"].to(device)
            action_logits, reason_logits = model(x, ids[:, :-1])
            action_loss = F.cross_entropy(action_logits, y)
            reason_loss = F.cross_entropy(
                reason_logits.reshape(-1, reason_logits.size(-1)),
                ids[:, 1:].reshape(-1),
            )
            loss = action_loss + 0.5 * reason_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        val_acc = eval_action_reason(model, val_loader, device)
        hist.append({"train_loss": tr_loss / max(1, len(loader)), "val_acc": val_acc})
    return hist


@torch.no_grad()
def eval_action(model, loader, device):
    model.eval()
    ok, total = 0, 0
    for b in loader:
        x = b["image"].to(device)
        y = b["action"].to(device)
        p = model(x).argmax(dim=-1)
        ok += (p == y).sum().item()
        total += y.numel()
    return ok / max(1, total)


@torch.no_grad()
def eval_action_reason(model, loader, device):
    model.eval()
    ok, total = 0, 0
    for b in loader:
        x = b["image"].to(device)
        y = b["action"].to(device)
        ids = b["reason_ids"].to(device)
        p, _ = model(x, ids[:, :-1])
        p = p.argmax(dim=-1)
        ok += (p == y).sum().item()
        total += y.numel()
    return ok / max(1, total)


def ppo_update_step(model, batch, old_logp, adv, cfg, opt):
    logits = model(batch["image"])
    dist = torch.distributions.Categorical(logits=logits)
    new_logp = dist.log_prob(batch["action"])
    ratio = (new_logp - old_logp).exp()
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    loss = -torch.min(unclipped, clipped).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())

