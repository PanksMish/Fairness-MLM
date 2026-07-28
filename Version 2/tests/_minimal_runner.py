"""
Zero-dependency test runner used ONLY in environments without pytest
(e.g. this sandbox, which has no network access to pip install).

In any real environment, run the actual test suite with:
    pytest tests/ -v

This file monkeypatches a tiny `pytest`-like shim (approx / raises) so the
real test files in this directory can be imported and executed unmodified,
proving the underlying implementation logic is correct even without the
pytest package installed.
"""
import importlib.util
import sys
import traceback
import types
import math


class _Approx:
    def __init__(self, expected, rel=1e-6, abs_=1e-9):
        self.expected = expected
        self.rel = rel
        self.abs = abs_

    def __eq__(self, other):
        return math.isclose(other, self.expected, rel_tol=self.rel, abs_tol=self.abs)

    def __repr__(self):
        return f"approx({self.expected})"


class _RaisesContext:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} but no exception was raised")
        if not issubclass(exc_type, self.exc_type):
            return False
        return True


fake_pytest = types.ModuleType("pytest")
fake_pytest.approx = _Approx
fake_pytest.raises = _RaisesContext
sys.modules["pytest"] = fake_pytest


def run_test_file(path):
    spec = importlib.util.spec_from_file_location("test_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    test_fns = [getattr(mod, name) for name in dir(mod) if name.startswith("test_")]
    passed, failed = 0, 0
    for fn in test_fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc(limit=2)
            failed += 1
    return passed, failed


if __name__ == "__main__":
    import os
    test_dir = os.path.dirname(__file__)
    files = sorted(
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if f.startswith("test_") and f.endswith(".py")
    )
    total_pass, total_fail = 0, 0
    for f in files:
        print(f"\n=== {os.path.basename(f)} ===")
        p, fl = run_test_file(f)
        total_pass += p
        total_fail += fl
    print(f"\n{'='*40}\nTOTAL: {total_pass} passed, {total_fail} failed\n{'='*40}")
    sys.exit(1 if total_fail else 0)
