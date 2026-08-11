"""Guard tests for perks retrieval quality.

Runs the labeled benchmark (`mcp_servers.perks_eval`) against an ephemeral store
so a regression in the embeddings, seed, or ranking blend fails CI rather than
silently degrading recommendations.
"""

import chromadb
import pytest

from table_for_four.mcp_servers.perks.eval import EVAL_CASES, run_eval
from table_for_four.mcp_servers.perks.server import build_collection


@pytest.fixture(scope="module")
def collection():
    return build_collection(chromadb.EphemeralClient())


def test_every_intent_surfaces_its_restaurant(collection):
    # At the default blend, every labeled intent must surface a relevant
    # restaurant, and the first hit should rank at (or very near) the top.
    result = run_eval(collection, k=3)
    agg = result["aggregate"]
    assert agg["hit@3"] == 1.0, result["cases"]
    assert agg["mrr"] >= 0.9, result["cases"]


def test_semantic_weight_improves_precision(collection):
    # The vector half of the hybrid should earn its keep: leaning semantic yields
    # at least as precise a top-k as leaning on metadata fit alone.
    pure_semantic = run_eval(collection, k=3, semantic_weight=1.0)["aggregate"]
    pure_metadata = run_eval(collection, k=3, semantic_weight=0.0)["aggregate"]
    assert pure_semantic["prec@3"] >= pure_metadata["prec@3"]


def test_benchmark_labels_reference_real_places(collection):
    # Every relevance label must be a place_id that actually exists in the store,
    # so the benchmark can't silently pass on typos.
    stored = set(collection.get()["metadatas"] and
                 [m["place_id"] for m in collection.get()["metadatas"]])
    for case in EVAL_CASES:
        assert case.relevant, case.query
        assert case.relevant <= stored, (case.query, case.relevant - stored)
