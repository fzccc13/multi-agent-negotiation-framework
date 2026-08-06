import pytest

from experiment_ascendc import AgentLLMInterface, CallBudgetExceeded
from negotiation.evaluation.reporting import classify_failure, summarize


def test_llm_call_budget_is_enforced_before_extra_call():
    client = AgentLLMInterface(use_mock=True)
    client.call_budget = 1
    client.call_llm(0, "first")
    with pytest.raises(CallBudgetExceeded):
        client.call_llm(0, "second")
    assert client.call_count == 1


def test_summary_reports_resources_and_failure_types():
    records = [
        {"mode": "baseline", "passed": True, "llm_calls": 3, "npu_executions": 2,
         "failure_type": None},
        {"mode": "baseline", "passed": False, "llm_calls": 5, "npu_executions": 4,
         "failure_type": "compile_failure"},
    ]
    report = summarize(records)["baseline"]
    assert report["pass_rate"] == 0.5
    assert report["avg_llm_calls"] == 4
    assert report["avg_npu_executions"] == 3
    assert report["failure_types"] == {"compile_failure": 1}
    assert classify_failure("precision mismatch") == "precision_failure"
