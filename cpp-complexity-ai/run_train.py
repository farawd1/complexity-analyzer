"""
Entry point — generates synthetic data + loads custom JSONL, trains model, evaluates.
"""
import os
import torch
import torch.nn as nn
from model.transformer import ComplexityTransformer
from model.tokenizer import CppTokenizer
from parser.ast_parser import CppASTParser
from data.generator import build_dataset
from data.dataset import get_dataloaders, load_custom_jsonl
from model.train import train, evaluate
from eval.metrics import classification_report, print_report, confusion_matrix
import numpy as np

CUSTOM_JSONL = "data/custom_dataset.jsonl"


def main():
    print("=" * 60)
    print("C++ Complexity Analyzer — Training Pipeline")
    print("=" * 60)

    print("\n[1/5] Generating synthetic dataset...")
    data = build_dataset(seed=42)

    # Load user-provided examples if file exists
    if os.path.exists(CUSTOM_JSONL):
        print(f"\n[+] Loading custom examples from {CUSTOM_JSONL}...")
        custom = load_custom_jsonl(CUSTOM_JSONL)
        data = data + custom
        print(f"[+] Total after merge: {len(data)} examples")
    else:
        print(f"\n[!] No custom dataset found at {CUSTOM_JSONL} — using synthetic only")
        print(f"    Create this file to add your own examples (see README for format)")

    print("\n[2/5] Building tokenizer...")
    tokenizer = CppTokenizer(max_vocab=8000, max_seq_len=512)
    tokenizer.build_vocab([c for c, _ in data])
    tokenizer.save("checkpoints/tokenizer.json")
    print(f"Vocab size: {tokenizer.vocab_size}")

    print("\n[3/5] Creating dataloaders...")
    parser = CppASTParser()
    train_loader, val_loader, test_loader = get_dataloaders(
        data, tokenizer, parser, batch_size=32)

    print("\n[4/5] Building model...")
    model = ComplexityTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256, n_heads=8,
        n_layers=4, d_ff=512,
        max_seq_len=512, dropout=0.1)
    print(f"Parameters: {model.count_parameters():,}")

    print("\n[5/5] Training...")
    train(model, train_loader, val_loader, n_epochs=40, lr=3e-4, patience=8)

    print("\n--- Test Evaluation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)
    report = classification_report(test_metrics["preds"], test_metrics["labels"])
    print_report(report)
    np.save("logs/confusion_matrix.npy",
            confusion_matrix(test_metrics["preds"], test_metrics["labels"]))
    print("\nDone! Start API: uvicorn api.server:app --reload --port 8000")


if __name__ == "__main__":
    main()
