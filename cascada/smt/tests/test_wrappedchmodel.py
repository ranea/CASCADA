"""Tests for the wrappedchmodel module."""
import doctest
import unittest

import cascada.smt.wrappedchmodel


class EmptyTest(unittest.TestCase):
    pass


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(cascada.smt.wrappedchmodel))
    return tests
