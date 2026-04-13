"""
Per-class precision, recall, F1 and confusion matrix.
"""
from collections import defaultdict
from typing import List
import numpy as np
from parser.ast_parser import IDX_TO_LABEL, COMPLEXITY_CLASSES


def classification_report(preds: List[int], labels: List[int]) -> dict:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for p, l in zip(preds, labels):
        if p == l: tp[l] += 1
        else: fp[p] += 1; fn[l] += 1
    report = {}
    for idx, cls in IDX_TO_LABEL.items():
        prec = tp[idx] / (tp[idx] + fp[idx] + 1e-9)
        rec  = tp[idx] / (tp[idx] + fn[idx] + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        report[cls] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": tp[idx] + fn[idx]}
    report["overall_accuracy"] = sum(p == l for p, l in zip(preds, labels)) / len(preds)
    return report


def confusion_matrix(preds, labels):
    n = len(COMPLEXITY_CLASSES)
    m = np.zeros((n, n), dtype=int)
    for p, l in zip(preds, labels): m[l][p] += 1
    return m


def print_report(report: dict):
    print(f"\n{'Class':<16} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 52)
    for cls in COMPLEXITY_CLASSES:
        if cls in report:
            r = report[cls]
            print(f"{cls:<16} {r['precision']:>8.3f} {r['recall']:>8.3f} "
                  f"{r['f1']:>8.3f} {r['support']:>8}")
    print("-" * 52)
    print(f"{'Overall accuracy':<16} {report['overall_accuracy']:>8.3f}")