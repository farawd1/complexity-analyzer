"""
Custom BPE-style tokenizer for C++ source code.
Maps tokens to integer ids. Unknown tokens go to <UNK>.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict

CPP_KEYWORDS = [
    "int", "float", "double", "bool", "char", "void", "auto", "long", "short",
    "unsigned", "signed", "const", "static", "inline", "virtual", "override",
    "return", "if", "else", "while", "for", "do", "switch", "case", "break",
    "continue", "class", "struct", "public", "private", "protected", "new",
    "delete", "nullptr", "true", "false", "template", "typename", "namespace",
    "using", "include", "vector", "string", "map", "set", "unordered_map",
    "pair", "sort", "find", "size", "push_back", "pop_back", "begin", "end",
    "swap", "min", "max", "abs", "cout", "cin",
]
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]


class CppTokenizer:
    def __init__(self, max_vocab: int = 8000, max_seq_len: int = 512):
        self.max_vocab = max_vocab
        self.max_seq_len = max_seq_len
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for tok in SPECIAL_TOKENS:
            idx = len(self.token2id)
            self.token2id[tok] = idx
            self.id2token[idx] = tok
        for kw in CPP_KEYWORDS:
            if kw not in self.token2id:
                idx = len(self.token2id)
                self.token2id[kw] = idx
                self.id2token[idx] = kw

    @property
    def pad_id(self): return self.token2id["<PAD>"]
    @property
    def unk_id(self): return self.token2id["<UNK>"]
    @property
    def bos_id(self): return self.token2id["<BOS>"]
    @property
    def eos_id(self): return self.token2id["<EOS>"]
    @property
    def vocab_size(self): return len(self.token2id)

    def _split(self, code: str) -> List[str]:
        code = re.sub(r"//[^\n]*", " ", code)
        code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
        return re.findall(r"[a-zA-Z_]\w*|\d+|[^\s\w]", code)

    def tokenize(self, code: str) -> List[str]:
        return self._split(code)

    def encode(self, code: str, add_special: bool = True) -> List[int]:
        tokens = self.tokenize(code)
        ids = [self.token2id.get(t, self.unk_id) for t in tokens]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        ids = ids[:self.max_seq_len]
        ids += [self.pad_id] * (self.max_seq_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id2token.get(i, "<UNK>") for i in ids
                  if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return " ".join(tokens)

    def build_vocab(self, corpus: List[str]):
        counter: Counter = Counter()
        for code in corpus:
            for tok in self._split(code):
                counter[tok] += 1
        for tok, _ in counter.most_common(self.max_vocab - len(self.token2id)):
            if tok not in self.token2id:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id,
                       "max_seq_len": self.max_seq_len}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CppTokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(max_seq_len=data["max_seq_len"])
        tok.token2id = data["token2id"]
        tok.id2token = {int(v): k for k, v in data["token2id"].items()}
        return tok