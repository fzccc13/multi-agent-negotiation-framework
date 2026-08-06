from simulated_executor import DeterministicDemoExecutor


def test_demo_executor_is_mode_neutral():
    executor = DeterministicDemoExecutor()
    code = 'extern "C" __global__ __aicore__ void add() {}'
    outcomes = []
    for mode in ("baseline", "K=N", "K=1", "K=2"):
        executor.current_mode = mode
        outcomes.append(executor.execute_test("ascend_add", code, {})[0])
    assert outcomes == [True, True, True, True]


def test_demo_executor_rejects_non_kernel_text():
    executor = DeterministicDemoExecutor()
    passed, output = executor.execute_test("ascend_add", "plain text", {})
    assert not passed
    assert "DEMO_ONLY" in output
