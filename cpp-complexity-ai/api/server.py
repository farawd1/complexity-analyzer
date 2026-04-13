"""
FastAPI inference server.
"""
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from model.transformer import ComplexityTransformer
from model.tokenizer import CppTokenizer
from parser.ast_parser import CppASTParser, ASTFeatures, IDX_TO_LABEL
from data.dataset import features_to_tensor

app = FastAPI(title="C++ Complexity Analyzer", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer: CppTokenizer = None
model: ComplexityTransformer = None
parser = CppASTParser()

EXPLANATIONS = {
    "O(1)":                  "Constant time — direct operations only, no loops.",
    "O(log n)":              "Logarithmic — binary search or halving recursion.",
    "O(sqrt n)":             "Square root — loop runs up to √n (e.g. divisor check, primality test).",
    "O(n)":                  "Linear — single pass over input of size n.",
    "O(n log n)":            "Linearithmic — e.g. sorting, binary search inside linear loop.",
    "O(n sqrt n)":           "n·√n — sqrt decomposition or sieve of Eratosthenes.",
    "O(n^2)":                "Quadratic — two nested loops over n elements.",
    "O(n^2 log n)":          "n²·log n — two nested loops with sorting inside.",
    "O(n^3)":                "Cubic — three nested loops (e.g. Floyd-Warshall, naive 3Sum).",
    "O(n + m)":              "Linear in graph size — BFS, DFS, topological sort.",
    "O(n * m)":              "Product of two dimensions — DP on grid or two sequences.",
    "O((n + q) * sqrt(n))":  "Sqrt decomposition with q queries, each O(√n).",
    "O(2^n)":                "Exponential — recursive branching (subsets, naive Fibonacci).",
    "O(n!)":                 "Factorial — all permutations (TSP brute force).",
}


@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = CppTokenizer.load("checkpoints/tokenizer.json")
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
    
    # Recreate model with same config as trained
    model = ComplexityTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=ckpt.get('d_model', 128),
        n_heads=ckpt.get('n_heads', 4),
        n_layers=ckpt.get('n_layers', 2),
        d_ff=ckpt.get('d_ff', 256),
        max_seq_len=256,  # Must be 256 to match checkpoint
        dropout=0.1
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    print(f"Model ready (val_acc={ckpt.get('val_acc', 'N/A'):.3f})")


class PredictRequest(BaseModel):
    code: str

class PredictResponse(BaseModel):
    complexity: str
    confidence: float
    probabilities: Dict[str, float]
    ast_features: Dict[str, float]
    explanation: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.code.strip():
        raise HTTPException(400, "Code must not be empty")
    ids = torch.tensor([tokenizer.encode(req.code)], dtype=torch.long, device=device)
    pmask = (ids == tokenizer.pad_id)
    try:
        feats = parser.parse_code(req.code)
    except Exception:
        feats = ASTFeatures()
    ftensor = features_to_tensor(feats).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(ids, ftensor, pmask), dim=-1)[0]
    pred_idx = probs.argmax().item()
    pred_label = IDX_TO_LABEL[pred_idx]
    from parser.ast_parser import COMPLEXITY_CLASSES as _C
    return PredictResponse(
        complexity=pred_label,
        confidence=round(probs[pred_idx].item(), 4),
        probabilities={IDX_TO_LABEL[i]: round(probs[i].item(), 4) for i in range(len(_C))},
        ast_features={
            "max_loop_depth": float(feats.max_loop_depth),
            "total_loops": float(feats.total_loops),
            "nested_loop_pairs": float(feats.nested_loop_pairs),
            "has_recursion": float(feats.has_recursion),
            "branch_count": float(feats.branch_count),
        },
        explanation=EXPLANATIONS.get(pred_label, ""),
    )

@app.get("/health")
def health(): return {"status": "ok", "model_loaded": model is not None}
