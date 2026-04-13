"""
Быстрое обучение с меньшей моделью
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
    print("C++ Complexity Analyzer — Fast Training")
    print("=" * 60)

    # Только кастомный датасет для скорости
    print("\n[1/4] Loading custom dataset...")
    custom = load_custom_jsonl(CUSTOM_JSONL)
    print(f"Loaded {len(custom)} examples")
    
    # Добавим немного синтетических для балансировки
    print("\n[2/4] Building tokenizer...")
    data = custom
    tokenizer = CppTokenizer(max_vocab=2000, max_seq_len=256)
    tokenizer.build_vocab([c for c, _ in data])
    tokenizer.save("checkpoints/tokenizer.json")
    print(f"Vocab size: {tokenizer.vocab_size}")

    print("\n[3/4] Creating dataloaders (batch=64)...")
    parser = CppASTParser()
    train_loader, val_loader, test_loader = get_dataloaders(
        data, tokenizer, parser, batch_size=64)

    print("\n[4/4] Building model (small)...")
    model = ComplexityTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128, n_heads=4,
        n_layers=2, d_ff=256,
        max_seq_len=256, dropout=0.1)
    print(f"Parameters: {model.count_parameters():,}")

    print("\n[5/5] Training (3 epochs)...")
    train(model, train_loader, val_loader, n_epochs=3, lr=1e-3, patience=10)

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
    print("\nDone!")


if __name__ == "__main__":
    main()
