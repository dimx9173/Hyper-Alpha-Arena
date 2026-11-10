"""
Code validator for Program Trader.
Validates strategy code for syntax, security, and template compliance.
"""

import ast
from collections import OrderedDict
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


# Forbidden imports and functions
FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "requests", "urllib", "http",
    "pickle", "marshal", "shelve",
    "ctypes", "multiprocessing", "threading",
    "importlib", "builtins", "__builtins__",
}

FORBIDDEN_FUNCTIONS = {
    "eval", "exec", "compile", "open", "input",
    "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr",
    "breakpoint", "exit", "quit",
}

ALLOWED_BUILTINS = {
    "abs", "min", "max", "sum", "len", "round",
    "int", "float", "str", "bool", "list", "dict", "tuple", "set",
    "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "any", "all", "isinstance", "type",
    "True", "False", "None",
}

PREINJECTED_MODULE_GUIDANCE = {
    "math": "Do not use import math. Use injected math.sqrt()/math.log()/math.exp() directly.",
    "time": "Do not use import time. Use injected time.time() directly.",
}


class CodeValidator:
    """Validates strategy code for safety and correctness."""

    def validate(self, code: str) -> ValidationResult:
        """Run all validation checks on code."""
        errors = []
        warnings = []

        # 1. Syntax check
        syntax_result = self._check_syntax(code)
        if syntax_result:
            errors.append(syntax_result)
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # 2. Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # 3. Security check
        security_errors = self._check_security(tree)
        errors.extend(security_errors)

        # 4. Template compliance check
        template_errors, template_warnings = self._check_template(tree)
        errors.extend(template_errors)
        warnings.extend(template_warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_syntax(self, code: str) -> Optional[str]:
        """Check Python syntax."""
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"Line {e.lineno}: {e.msg}"

    def _check_security(self, tree: ast.AST) -> List[str]:
        """Check for forbidden imports and function calls."""
        errors = []

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in PREINJECTED_MODULE_GUIDANCE:
                        errors.append(PREINJECTED_MODULE_GUIDANCE[module])
                        continue
                    if module in FORBIDDEN_IMPORTS:
                        errors.append(f"Forbidden import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in PREINJECTED_MODULE_GUIDANCE:
                        errors.append(PREINJECTED_MODULE_GUIDANCE[module])
                        continue
                    if module in FORBIDDEN_IMPORTS:
                        errors.append(f"Forbidden import: {node.module}")

            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_FUNCTIONS:
                        errors.append(f"Forbidden function: {node.func.id}()")

        return errors

    def _check_template(self, tree: ast.AST) -> Tuple[List[str], List[str]]:
        """Check strategy template compliance."""
        errors = []
        warnings = []

        # Find class definitions
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        if not classes:
            errors.append("No class definition found. Strategy must define a class.")
            return errors, warnings

        # Check for Strategy-like class
        strategy_class = None
        for cls in classes:
            # Check if inherits from Strategy or has should_trade method
            methods = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
            if "should_trade" in methods:
                strategy_class = cls
                break

        if not strategy_class:
            errors.append("Strategy class must have 'should_trade' method.")
            return errors, warnings

        # Check should_trade signature
        for node in strategy_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == "should_trade":
                args = node.args
                # Should have self and data parameters
                if len(args.args) < 2:
                    errors.append("should_trade must accept 'data' parameter.")

        # Check for init method (warning if missing)
        methods = [n.name for n in strategy_class.body if isinstance(n, ast.FunctionDef)]
        if "init" not in methods and "__init__" not in methods:
            warnings.append("Consider adding 'init' method for parameter initialization.")

        return errors, warnings


# validate() 是純函式：輸出只由 code 決定，沒有其他輸入也沒有外部狀態。
# 但回測會對同一份程式碼呼叫它幾萬次（每個 trigger 一次），每次都重跑
# 完整的 ast.parse + 兩次 ast.walk。實測 8000 個 trigger 裡這佔 61% 的時間。
# 依程式碼內容快取：同一份字串直接重用結果，不同字串自然不會撞。
#
# 只快取有界的數量，避免長期服務累積無限記憶體。
_VALIDATE_CACHE: "OrderedDict[str, ValidationResult]" = OrderedDict()
_VALIDATE_CACHE_MAX = 256


def validate_strategy_code(code: str) -> ValidationResult:
    """Convenience function to validate strategy code.

    Results are memoized by code content. `validate()` is a pure function of
    `code`, so this only skips recomputation -- it cannot change the outcome.
    """
    cached = _VALIDATE_CACHE.get(code)
    if cached is not None:
        _VALIDATE_CACHE.move_to_end(code)
        return cached

    validator = CodeValidator()
    result = validator.validate(code)

    _VALIDATE_CACHE[code] = result
    if len(_VALIDATE_CACHE) > _VALIDATE_CACHE_MAX:
        _VALIDATE_CACHE.popitem(last=False)
    return result
