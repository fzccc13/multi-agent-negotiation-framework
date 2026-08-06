import contextlib
import io

import pytest

from framework import MultiAgentNegotiationFramework, VoteResult


@pytest.mark.parametrize("n,k", [(1, 1), (5, 0), (5, 6)])
def test_rejects_invalid_topology(n, k):
    with pytest.raises(ValueError):
        MultiAgentNegotiationFramework(n, k)


def test_weighted_scores_respect_voter_weight():
    framework = MultiAgentNegotiationFramework(3, 2)
    framework.agents[0].weight = 0.6
    framework.agents[1].weight = 0.3
    framework.agents[2].weight = 0.1
    scores = framework.calculate_weighted_scores([
        VoteResult(0, [1, 2]),
        VoteResult(1, [0, 2]),
        VoteResult(2, [0, 1]),
    ])
    assert scores[1] == pytest.approx(0.65)
    assert scores[0] == pytest.approx(0.4)
    assert scores[2] == pytest.approx(0.45)


@pytest.mark.parametrize("k", [1, 2, 5])
def test_protocol_terminates_with_one_winner(k):
    framework = MultiAgentNegotiationFramework(5, k)

    def initial(agent_id, _phase):
        return f"solution-{agent_id}"

    def refine(agent, _alive, _weights):
        return agent.solution + "-reviewed"

    def vote(agent, alive, top_k, weight_distribution):
        candidates = [other for other in alive if other.agent_id != agent.agent_id]
        candidates.sort(key=lambda other: (-weight_distribution[other.agent_id]["weight"], other.agent_id))
        return [other.agent_id for other in candidates[:top_k]]

    with contextlib.redirect_stdout(io.StringIO()):
        winner = framework.run(initial, refine, vote)
    assert winner in framework.agents
    assert framework.round < framework.N
    assert framework.history[-1].get("winner") == winner.agent_id
