"""
确定性演示执行器（Deterministic Demo Executor）

用途：
    在无 310B4 真机 / 无 API Key 的环境下跑通协议与执行器接口，生成协商
    过程回放。它只验证工程流程，不评估代码质量，也不量化协商增益。

设计原则：
    - 不依赖 paramiko / SSH / 真实 NPU，纯 Python + 标准库 + numpy。
    - 确定性：相同任务与代码永远得到相同结果。
    - 模式中立：判定不读取 baseline/K=1/K=2/K=N 模式。

重要说明（诚实标注）：
    本执行器产出的布尔值没有性能含义，不得用于计算或宣传协商增益。
    真实通过率必须由真实模型和 310B4 执行器的原始记录计算。
"""

import re
import hashlib
from typing import Dict, Tuple


class DeterministicDemoExecutor:
    """
    AscendC 执行器接口的确定性演示实现。

    与 experiment_ascendc.py 中的 AscendCTestExecutor 保持相同接口
    （execute_test），便于在模拟 / 真机两种后端间无缝切换。
    """

    def __init__(self):
        self.execution_count = 0
        # 兼容旧调用方；execute_test 有意忽略该字段。
        self.current_mode = 'baseline'

    # ------------------------------------------------------------------
    # 公共工具
    # ------------------------------------------------------------------
    @staticmethod
    def extract_ascendc_code(text: str) -> str:
        """从 LLM 输出中提取 AscendC kernel 代码（与 AscendCTestExecutor 保持一致）。"""
        for pat in (r'```(?:cpp|c\+\+)\s*\n(.*?)\n```', r'```\s*\n(.*?)\n```'):
            matches = re.findall(pat, text, re.DOTALL)
            if matches:
                return matches[0].strip()
        return text.strip()

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------
    def execute_test(self, op_name: str, kernel_code: str, problem: Dict,
                     agent_id: int = None, suffix: str = None) -> Tuple[bool, str]:
        """
        检查是否收到非空 AscendC 风格代码并返回稳定结果。

        判定不读取模式、Agent 或轮次，避免将预设差异误报为算法增益。

        返回: (是否通过, 输出信息)
        """
        self.execution_count += 1
        extracted = self.extract_ascendc_code(kernel_code)
        passed = bool(extracted.strip()) and (
            "__global__" in extracted or "__aicore__" in extracted
        )
        digest = hashlib.sha256(f"{op_name}|{extracted}".encode("utf-8")).hexdigest()[:12]

        if passed:
            out = f"[DEMO_ONLY] interface accepted code digest={digest}"
        else:
            out = f"[DEMO_ONLY] missing AscendC kernel marker digest={digest}"
        return passed, out


# Compatibility alias for the original public API.
SimulatedExecutor = DeterministicDemoExecutor
