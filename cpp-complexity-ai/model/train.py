"""
Training loop with AdamW + cosine LR schedule + early stopping + checkpointing.
"""
import os
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from model.transformer import ComplexityTransformer


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="  train", leave=False):
        ids  = batch["token_ids"].to(device)
        mask = batch["padding_mask"].to(device)
        feat = batch["ast_features"].to(device)
        lbl  = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(ids, feat, mask)
        loss = criterion(logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(lbl)
        correct += (logits.argmax(-1) == lbl).sum().item()
        total += len(lbl)
    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="  eval ", leave=False):
        ids  = batch["token_ids"].to(device)
        mask = batch["padding_mask"].to(device)
        feat = batch["ast_features"].to(device)
        lbl  = batch["label"].to(device)
        logits = model(ids, feat, mask)
        loss = criterion(logits, lbl)
        total_loss += loss.item() * len(lbl)
        preds = logits.argmax(-1)
        correct += (preds == lbl).sum().item()
        total += len(lbl)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(lbl.cpu().tolist())
    return {"loss": total_loss / total, "accuracy": correct / total,
            "preds": all_preds, "labels": all_labels}


def train(model, train_loader, val_loader, n_epochs=40, lr=3e-4,
          weight_decay=1e-2, patience=8,
          checkpoint_dir="checkpoints", log_dir="logs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Model parameters: {model.count_parameters():,}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    Path(checkpoint_dir).mkdir(exist_ok=True)
    Path(log_dir).mkdir(exist_ok=True)
    best_val_acc, patience_counter, history = 0.0, 0, []
    
    # Extract model config from model for saving
    def get_model_config(m):
        return {
            "d_model": m.d_model,
            "n_heads": m.layers[0].attn.n_heads,
            "n_layers": len(m.layers),
            "d_ff": m.layers[0].ff.net[0].out_features,
            "max_seq_len": 128,  # Default for demo
        }
    
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}  lr={scheduler.get_last_lr()[0]:.2e}")
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  train loss={tr['loss']:.4f} acc={tr['accuracy']:.3f}")
        print(f"  val   loss={vl['loss']:.4f} acc={vl['accuracy']:.3f}  ({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch, "train_loss": tr["loss"],
                         "train_acc": tr["accuracy"],
                         "val_loss": vl["loss"], "val_acc": vl["accuracy"]})
        if vl["accuracy"] > best_val_acc:
            best_val_acc = vl["accuracy"]
            patience_counter = 0
            # Include config in checkpoint
            ckpt = {
                "epoch": epoch, 
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc,
                **get_model_config(model)
            }
            torch.save(ckpt, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  ✓ Saved (val_acc={best_val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    with open(os.path.join(log_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    return history
