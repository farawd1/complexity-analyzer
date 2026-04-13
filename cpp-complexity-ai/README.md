# C++ Complexity Analyzer AI

Predicts Big-O complexity of C++ functions using a Transformer built from scratch.

## Complexity classes
`O(1)` · `O(log n)` · `O(sqrt n)` · `O(n)` · `O(n log n)` · `O(n sqrt n)` · `O(n^2)` · `O(n^2 log n)` · `O(n^3)` · `O(n + m)` · `O(n * m)` · `O((n + q) * sqrt(n))` · `O(2^n)` · `O(n!)`

## Architecture
- **Parser**: libclang AST traversal → 13 hand-crafted features (including sqrt detection)
- **Tokenizer**: custom vocabulary for C++ source (no HuggingFace)
- **Model**: Transformer encoder from scratch (no nn.MultiheadAttention wrapper)
  - Multi-head self-attention, sinusoidal positional encoding, pre-norm layers
  - Feature fusion: token path + AST feature path → classifier (14 classes)
- **Dataset**: 15,400+ synthetic C++ examples auto-generated from templates
- **API**: FastAPI REST endpoint

## Quickstart
```bash
pip install -r requirements.txt
python run_train.py
uvicorn api.server:app --reload --port 8000
```

## API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code": "int sum(vector<int>& a) { int s=0; for(int x:a) s+=x; return s; }"}'
```

## Adding your own examples

Create `data/custom_dataset.jsonl` — one JSON object per line:

```json
{"code": "void f(...) { ... }", "complexity": "O(n log n)", "notes": "binary search in loop"}
{"code": "...", "complexity": "O((n + q) * sqrt(n))", "notes": "sqrt decomp"}
```

Supported complexity labels:
`O(1)` · `O(log n)` · `O(sqrt n)` · `O(n)` · `O(n log n)` · `O(n sqrt n)` ·
`O(n^2)` · `O(n^2 log n)` · `O(n^3)` · `O(n + m)` · `O(n * m)` ·
`O((n + q) * sqrt(n))` · `O(2^n)` · `O(n!)`

Then retrain: `python run_train.py`

## Project layout
cpp-complexity-ai/
├── parser/      ast_parser.py            (libclang AST traversal)
├── model/       transformer.py           (from-scratch Transformer)
│                tokenizer.py             (custom C++ tokenizer)
│                train.py                 (training loop)
├── data/        generator.py             (synthetic dataset)
│                dataset.py               (PyTorch Dataset)
│                custom_dataset.jsonl    (user-provided examples)
├── eval/        metrics.py               (precision, recall, F1)
├── api/         server.py                (FastAPI)
├── tests/       test_parser.py
├── checkpoints/ best_model.pt, tokenizer.json
└── logs/        history.json, confusion_matrix.npy
