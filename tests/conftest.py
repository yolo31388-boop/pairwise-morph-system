"""Pytest configuration.

The legacy test class uses unittest-style ``self.assertIsNotNone`` without
inheriting ``unittest.TestCase``. Provide the assertion helper on collection
so the tests run unmodified.
"""


def _assertIsNotNone(self, obj, msg=None):
    if obj is None:
        raise AssertionError(msg or "unexpectedly None")


def pytest_collection(session):
    from tests.test_morph import TestMorphSystem

    if not hasattr(TestMorphSystem, "assertIsNotNone"):
        TestMorphSystem.assertIsNotNone = _assertIsNotNone
