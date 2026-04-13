"""
Ultra-fast demo trainer - works in ~2-3 minutes on CPU.
Reduces dataset and training significantly.
"""
import torch
import torch.nn as nn
import random
from model.transformer import ComplexityTransformer
from model.tokenizer import CppTokenizer
from parser.ast_parser import CppASTParser
from data.generator import build_dataset
from data.dataset import get_dataloaders
from model.train import train, evaluate
from eval.metrics import classification_report, print_report, confusion_matrix
import numpy as np


def main():
    print("=" * 60)
    print("C++ Complexity Analyzer — Ultra-Fast Demo")
    print("=" * 60)

    # Use smaller dataset (just 1500 samples)
    print("\n[1/5] Generating small dataset (1500 samples)...")
    original_generators = {
        "O(1)":       (lambda: __import__('data.generator', fromlist=['']).gen_o1(), 500),
        "O(log n)":   (lambda: __import__('data.generator', fromlist=['']).gen_ologn(), 200),
        "O(n)":       (lambda: __import__('data.generator', fromlist=['']).gen_on(), 300),
        "O(n log n)": (lambda: __import__('data.generator', fromlist=['']).gen_onlogn(), 200),
        "O(n^2)":     (lambda: __import__('data.generator', fromlist=['']).gen_on2(), 250),
        "O(2^n)":     (lambda: __import__('data.generator', fromlist=['']).gen_o2n(), 30),
        "O(n!)":      (lambda: __import__('data.generator', fromlist=['']).gen_onfactorial(), 20),
    }
    from data import generator
    dataset = []
    random.seed(42)
    for label, (gen_fn, count) in {
        "O(1)": (generator.gen_o1, 500),
        "O(log n)": (generator.gen_ologn, 200),
        "O(n)": (generator.gen_on, 300),
        "O(n log n)": (generator.gen_onlogn, 200),
        "O(n^2)": (generator.gen_on2, 250),
        "O(2^n)": (generator.gen_o2n, 30),
        "O(n!)": (generator.gen_onfactorial, 20),
    }.items():
        for _ in range(count):
            code, lbl = gen_fn()
            dataset.append((code.strip(), lbl))
    random.shuffle(dataset)
    print(f"Dataset size: {len(dataset)}")

    print("\n[2/5] Building tokenizer...")
    tokenizer = CppTokenizer(max_vocab=2000, max_seq_len=128)
    tokenizer.build_vocab([c for c, _ in dataset])
    tokenizer.save("checkpoints/tokenizer.json")
    print(f"Vocab size: {tokenizer.vocab_size}")

    print("\n[3/5] Creating dataloaders (batch_size=64)...")
    parser = CppASTParser()
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, tokenizer, parser, batch_size=64)

    print("\n[4/5] Building model (tiny)...")
    model = ComplexityTransformer(
        vocab_size=tokenizer.vocab_size, d_model=48, n_heads=4,
        n_layers=1, d_ff=96, max_seq_len=128, dropout=0.1)
    print(f"Parameters: {model.count_parameters():,}")

    print("\n[5/5] Training (2 epochs for demo)...")
    train(model, train_loader, val_loader, n_epochs=2, lr=1e-3, patience=10)

    print("\n--- Test Evaluation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        test_metrics = evaluate(model, test_loader, criterion, device)
        report = classification_report(test_metrics["preds"], test_metrics["labels"])
        print_report(report)
        np.save("logs/confusion_matrix.npy",
                confusion_matrix(test_metrics["preds"], test_metrics["labels"]))
    except Exception as e:
        print(f"Could not evaluate: {e}")

    print("\n" + "=" * 60)
    print("Demo complete! For full training run:")
    print("  python run_train.py  # Full model (~2 hours on CPU)")
    print("  or use GPU for faster training")
    print("=" * 60)


if __name__ == "__main__":
    main()