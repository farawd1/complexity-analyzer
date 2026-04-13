"""
PyTorch Dataset wrapping (code, label) pairs.
Uses pre-computed features for speed, falls back if AST parsing fails.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Optional
from model.tokenizer import CppTokenizer
from parser.ast_parser import CppASTParser, ASTFeatures, LABEL_TO_IDX


# Global cache for AST features (computed once during dataset init)
_ast_feature_cache: List[Optional[torch.Tensor]] = []


def features_to_tensor(f: ASTFeatures) -> torch.Tensor:
    return torch.tensor([
        float(f.max_loop_depth),
        float(f.total_loops),
        float(f.nested_loop_pairs),
        float(f.has_recursion),
        float(f.recursive_calls),
        float(f.branch_count),
        float(f.array_accesses),
        float(f.map_accesses),
        float(f.total_calls),
        float(f.stdlib_sort_calls),
        float(f.stdlib_search_calls),
        float(f.node_count) / 100.0,
        float(getattr(f, 'has_sqrt_call', False)),  # 13th feature
    ], dtype=torch.float32)


def compute_all_features(data: List[Tuple[str, str]], parser: CppASTParser) -> List[torch.Tensor]:
    """Pre-compute AST features for all samples."""
    global _ast_feature_cache
    _ast_feature_cache = []
    print("  Computing AST features...")
    for i, (code, label) in enumerate(data):
        if i % 1000 == 0:
            print(f"    {i}/{len(data)}")
        try:
            ast_feats = parser.parse_code(code, label)
            _ast_feature_cache.append(features_to_tensor(ast_feats))
        except Exception:
            _ast_feature_cache.append(torch.zeros(13, dtype=torch.float32))
    return _ast_feature_cache


class CppComplexityDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]],
                 tokenizer: CppTokenizer):
        self.tokenizer = tokenizer
        self.samples = data

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        code, label = self.samples[idx]
        token_ids = torch.tensor(self.tokenizer.encode(code), dtype=torch.long)
        padding_mask = (token_ids == self.tokenizer.pad_id)
        ast_features = _ast_feature_cache[idx]
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "ast_features": ast_features,
            "label": torch.tensor(LABEL_TO_IDX[label], dtype=torch.long),
        }


def get_dataloaders(data, tokenizer, parser, batch_size=32,
                    train_ratio=0.8, val_ratio=0.1, num_workers=0):
    # Pre-compute AST features once
    compute_all_features(data, parser)
    
    dataset = CppComplexityDataset(data, tokenizer)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42))
    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=torch.cuda.is_available())
    return make_loader(train_set, True), make_loader(val_set, False), make_loader(test_set, False)


def load_custom_jsonl(path: str):
    """
    Load user-provided examples from a JSONL file.
    Each line must be: {"code": "...", "complexity": "O(n log n)", "notes": "..."}
    Only examples with known labels are loaded. Unknown labels are skipped with a warning.
    """
    import json
    from parser.ast_parser import LABEL_TO_IDX
    data = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [JSONL] Line {lineno}: JSON parse error — {e}")
                skipped += 1
                continue
            code = obj.get("code", "").strip()
            label = obj.get("complexity", "").strip()
            if not code:
                print(f"  [JSONL] Line {lineno}: empty code — skipping")
                skipped += 1
                continue
            if label not in LABEL_TO_IDX:
                print(f"  [JSONL] Line {lineno}: unknown label {label!r} — skipping")
                skipped += 1
                continue
            data.append((code, label))
    print(f"[JSONL] Loaded {len(data)} examples, skipped {skipped} from {path}")
    return data
