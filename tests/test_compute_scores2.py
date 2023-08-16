"""Unit tests for the file compute_score2.py."""
import unittest
import spacy
from spacy.language import Language
from spacy_readability import Readability
from compute_scores2 import apply_ranking # noqa: F401

@Language.component("readability")
def readability(doc):
    read = Readability()
    doc = read(doc)
    return doc

nlp = spacy.load('en_core_web_md')
nlp.add_pipe("textrank", last=True)
nlp.add_pipe("readability", last=True)


class TestComputeScores2(unittest.TestCase):
    """Unit tests for compute_scores2.

    This class contains the following functions:
    -
    """

    def test_test(self):
        print('hello tests')
