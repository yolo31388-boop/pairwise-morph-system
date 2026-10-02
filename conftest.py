"""Pytest fixtures for the skeleton test suite.

The provided test classes do not inherit from unittest.TestCase, so the
unittest-style assertion helpers they call are injected here at collection
time (only when missing) without modifying the test files.
"""


def _assert_is_not_none(self, obj, msg=None):
    assert obj is not None, msg or f"{obj!r} is None"


def _assert_is_none(self, obj, msg=None):
    assert obj is None, msg or f"{obj!r} is not None"


def _assert_true(self, expr, msg=None):
    assert expr, msg or f"{expr!r} is not true"


def _assert_equal(self, first, second, msg=None):
    assert first == second, msg or f"{first!r} != {second!r}"


_INJECTED = {
    "assertIsNotNone": _assert_is_not_none,
    "assertIsNone": _assert_is_none,
    "assertTrue": _assert_true,
    "assertEqual": _assert_equal,
}


def pytest_collection_modifyitems(items):
    for item in items:
        cls = getattr(item, "cls", None)
        if cls is None:
            continue
        for name, func in _INJECTED.items():
            if not hasattr(cls, name):
                setattr(cls, name, func)
