import pytest
import torch

from lyricaligner.alignment import Aligner


@pytest.fixture
def aligner():
    return Aligner()


def test_alignment(aligner):
    logits = torch.randn(100, 100)
    tokens = torch.randint(0, 100, (100,))
    blank_id = 0
    alignment = aligner.align(logits, tokens, blank_id)
    assert len(alignment) == 100
