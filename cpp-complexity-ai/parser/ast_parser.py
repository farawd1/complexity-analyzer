"""
AST Parser for C++ code using libclang.
Extracts structural features used as input to the model.
"""
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional
import clang.cindex as clang


@dataclass
class ASTFeatures:
    max_loop_depth: int = 0
    total_loops: int = 0
    nested_loop_pairs: int = 0
    has_recursion: bool = False
    recursive_calls: int = 0
    branch_count: int = 0
    ternary_count: int = 0
    array_accesses: int = 0
    map_accesses: int = 0
    total_calls: int = 0
    stdlib_sort_calls: int = 0
    stdlib_search_calls: int = 0
    node_count: int = 0
    max_depth: int = 0
    has_sqrt_call: bool = False
    node_type_sequence: list = field(default_factory=list)
    complexity_label: Optional[str] = None


COMPLEXITY_CLASSES = [
    "O(1)",
    "O(log n)",
    "O(sqrt n)",
    "O(n)",
    "O(n log n)",
    "O(n sqrt n)",
    "O(n^2)",
    "O(n^2 log n)",
    "O(n^3)",
    "O(n + m)",
    "O(n * m)",
    "O((n + q) * sqrt(n))",
    "O(2^n)",
    "O(n!)",
]
LABEL_TO_IDX = {c: i for i, c in enumerate(COMPLEXITY_CLASSES)}
IDX_TO_LABEL = {i: c for c, i in LABEL_TO_IDX.items()}

# Include range-based for loops (CXX_FOR_RANGE_STMT)
LOOP_KINDS = {
    clang.CursorKind.FOR_STMT,
    clang.CursorKind.WHILE_STMT,
    clang.CursorKind.DO_STMT,
    clang.CursorKind.CXX_FOR_RANGE_STMT,
}
BRANCH_KINDS = {
    clang.CursorKind.IF_STMT,
    clang.CursorKind.SWITCH_STMT,
}


class CppASTParser:
    def __init__(self):
        self.index = clang.Index.create()

    def parse_code(self, code: str, label: Optional[str] = None) -> ASTFeatures:
        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            args = ["-std=c++17", "-x", "c++"]
            tu = self.index.parse(tmp_path, args=args)
            features = ASTFeatures(complexity_label=label)
            func_cursor = self._find_first_function(tu.cursor)
            if func_cursor:
                func_name = func_cursor.spelling
                self._visit(func_cursor, features, loop_depth=0,
                            current_depth=0, func_name=func_name)
        finally:
            os.unlink(tmp_path)
        return features

    def _find_first_function(self, cursor):
        for child in cursor.get_children():
            if child.kind in (
                clang.CursorKind.FUNCTION_DECL,
                clang.CursorKind.CXX_METHOD,
                clang.CursorKind.FUNCTION_TEMPLATE,
            ) and child.is_definition():
                return child
        return None

    def _visit(self, cursor, features: ASTFeatures,
               loop_depth: int, current_depth: int, func_name: str):
        features.node_count += 1
        features.max_depth = max(features.max_depth, current_depth)
        features.node_type_sequence.append(cursor.kind.value)

        if cursor.kind in LOOP_KINDS:
            features.total_loops += 1
            new_loop_depth = loop_depth + 1
            features.max_loop_depth = max(features.max_loop_depth, new_loop_depth)
            if loop_depth >= 1:
                features.nested_loop_pairs += 1
        else:
            new_loop_depth = loop_depth

        if cursor.kind in BRANCH_KINDS:
            features.branch_count += 1
        if cursor.kind == clang.CursorKind.CONDITIONAL_OPERATOR:
            features.ternary_count += 1

        if cursor.kind == clang.CursorKind.CALL_EXPR:
            features.total_calls += 1
            callee = cursor.spelling
            if callee == func_name:
                features.has_recursion = True
                features.recursive_calls += 1
            if callee in ("sort", "stable_sort", "partial_sort"):
                features.stdlib_sort_calls += 1
            if callee in ("find", "lower_bound", "upper_bound", "binary_search"):
                features.stdlib_search_calls += 1
            # Detect sqrt() calls — key signal for sqrt decomposition
            if callee in ("sqrt", "sqrtl", "sqrtf"):
                features.has_sqrt_call = True

        if cursor.kind == clang.CursorKind.ARRAY_SUBSCRIPT_EXPR:
            features.array_accesses += 1

        # Skip CXX_MEMBER_CALL_EXPR - doesn't exist in some libclang versions

        for child in cursor.get_children():
            self._visit(child, features, new_loop_depth,
                        current_depth + 1, func_name)
