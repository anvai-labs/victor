"""Ranking metrics cannot become arbitrary SQL expressions."""

from unittest.mock import Mock

import pytest

from victor.framework.rl.credit_persistence import CreditAssignmentDB


def test_unrecognized_ranking_metric_is_rejected_before_database_access(monkeypatch):
    database = object.__new__(CreditAssignmentDB)
    connection = Mock(side_effect=AssertionError("database must not be touched"))
    monkeypatch.setattr(database, "_get_connection", connection)
    with pytest.raises(ValueError, match="ranking metric"):
        database.get_top_agents(
            "total_credit) FROM agent_attribution UNION SELECT * FROM secrets --"
        )
    connection.assert_not_called()


@pytest.mark.parametrize("metric", ["total_credit", "direct_credit", "received_credit"])
def test_documented_ranking_metrics_are_accepted(monkeypatch, metric):
    from unittest.mock import MagicMock

    database = object.__new__(CreditAssignmentDB)
    connection = MagicMock()
    connection.__enter__.return_value.execute.return_value.fetchall.return_value = []
    monkeypatch.setattr(database, "_get_connection", lambda: connection)
    assert database.get_top_agents(metric, 3) == []
    args = connection.__enter__.return_value.execute.call_args.args
    assert args[1] == (3,)
