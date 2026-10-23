"""EmbeddingService.profile resolves the Hub commit under transformers 4 and 5."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from victor.storage.embeddings.service import EmbeddingService

COMMIT = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


class _FakeModel:
    def __init__(self, *, commit=None, card_revision=None, has_commit_attr=True):
        config = SimpleNamespace()
        if has_commit_attr:
            config._commit_hash = commit
        self._modules = [SimpleNamespace(auto_model=SimpleNamespace(config=config))]
        self.model_card_data = SimpleNamespace(base_model_revision=card_revision)
        self.max_seq_length = 512

    def __getitem__(self, index):
        return self._modules[index]

    def get_sentence_embedding_dimension(self):
        return 384


def _profile(model):
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    service._model = model
    with mock.patch.object(EmbeddingService, "_ensure_model_loaded", lambda self: None):
        return service.profile


def test_transformers4_commit_hash_is_used():
    assert _profile(_FakeModel(commit=COMMIT)).revision == COMMIT


def test_transformers5_falls_back_to_the_model_card_revision():
    """transformers 5 has no config._commit_hash; the card records the loaded commit."""
    profile = _profile(_FakeModel(card_revision=COMMIT, has_commit_attr=False))
    assert (profile.revision, profile.dimension, profile.max_tokens) == (COMMIT, 384, 512)


def test_commit_hash_wins_over_the_card_when_both_exist():
    other = "0" * 40
    assert _profile(_FakeModel(commit=COMMIT, card_revision=other)).revision == COMMIT


@pytest.mark.parametrize(
    "model",
    [
        _FakeModel(has_commit_attr=False),  # local path: no Hub provenance at all
        _FakeModel(commit="", card_revision=None),
        _FakeModel(commit=None, card_revision="main"),  # a branch is not immutable
    ],
)
def test_unversioned_models_still_fail_closed(model):
    with pytest.raises(ValueError, match="immutable 40-hex commit"):
        _profile(model)


def test_revision_helper_tolerates_unexpected_model_shapes():
    assert EmbeddingService._loaded_model_revision(object()) is None
    assert EmbeddingService._loaded_model_revision([]) is None
