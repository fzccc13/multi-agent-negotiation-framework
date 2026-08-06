"""
多Agent协商框架 - AscendC算子实验运行脚本
适配310B4真机环境，通过SSH远程执行build/run验证
对比四种模式：Baseline / K=N / K=1 / K=2
"""
import os
import sys
import io
import json
import re
import time
import tempfile
import textwrap
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# 强制 stdout/stderr 使用 UTF-8，避免 Windows 默认 GBK 编码在打印含 Unicode
# (如编译错误中的 ❌ U+2717) 时抛 UnicodeEncodeError 导致 Agent 运行异常
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# 导入框架（通用协商逻辑不变）
from framework import MultiAgentNegotiationFramework, Agent
from config_ascendc import EXPERIMENT_CONFIG, AGENT_MODELS, DATASET_CONFIG, NPU_CONFIG, EXECUTION_CONFIG
from simulated_executor import SimulatedExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


# ==================== SSH远程执行器 ====================
class SSHExecutor:
    """SSH连接管理器，负责与310B4真机通信"""

    def __init__(self):
        self.client = None
        self._connected = False

    def connect(self) -> bool:
        """建立SSH连接"""
        try:
            import paramiko
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(
                hostname=NPU_CONFIG['host'],
                port=NPU_CONFIG['port'],
                username=NPU_CONFIG['username'],
                password=NPU_CONFIG['password'],
                timeout=30,
                look_for_keys=False,
                allow_agent=False,
            )
            self._connected = True
            print(f"[SSH] 连接成功: {NPU_CONFIG['username']}@{NPU_CONFIG['host']}:{NPU_CONFIG['port']}")

            # 初始化环境
            for cmd in NPU_CONFIG['env_setup_commands']:
                self.run_command(cmd, timeout=10)

            return True
        except ImportError:
            print("[ERROR] 未安装paramiko库，请执行: pip install paramiko")
            return False
        except Exception as e:
            print(f"[ERROR] SSH连接失败: {e}")
            return False

    def _is_alive(self) -> bool:
        """检测SSH连接是否仍然存活"""
        if not self._connected or not self.client:
            return False
        try:
            stdin, stdout, stderr = self.client.exec_command("echo ok", timeout=5)
            exit_code = stdout.channel.recv_exit_status()
            return exit_code == 0
        except Exception:
            return False

    def ensure_connected(self) -> bool:
        """确保SSH连接存活，断开则自动重连（3次重试，递增等待）"""
        if self._is_alive():
            return True
        # 多次重试，递增等待：10s, 20s, 30s
        for attempt in range(3):
            wait = (attempt + 1) * 10
            if attempt == 0:
                print("[SSH] 连接已断开，尝试重连...")
            else:
                print(f"[SSH] 第{attempt + 1}次重连（等待{wait}秒）...")
            try:
                self.disconnect()
            except Exception:
                pass
            time.sleep(wait)
            if self.connect():
                return True
            print(f"[SSH] 重连失败，{wait}秒后重试...")
        print("[ERROR] SSH重连3次均失败")
        return False

    def disconnect(self):
        """关闭SSH连接"""
        if self.client:
            self.client.close()
            self._connected = False
            print("[SSH] 连接已关闭")

    def run_command(self, command: str, timeout: int = None) -> Tuple[int, str, str]:
        """
        执行远程命令（激活conda base + 登录shell环境）

        关键修复：bash --login -c 不会加载 .bashrc（因为 .bashrc 有
        "if not interactive, return" guard），导致 conda base 未激活，
        opc编译kernel时找不到numpy。

        解决方案：显式从常见路径 source conda.sh 并 activate base，
        然后再执行实际命令。

        返回: (returncode, stdout, stderr)
        """
        if not self._connected:
            # 尝试自动重连，而不是直接报错
            if not self.ensure_connected():
                raise RuntimeError("SSH未连接且重连失败")

        timeout = timeout or EXECUTION_CONFIG['ssh_timeout']

        for _ssh_attempt in range(3):  # 最多3次：初始 + 2次重连重试
            if _ssh_attempt > 0:
                if not self.ensure_connected():
                    return -1, "", "SSH重连失败"
                print(f"[SSH] 重连成功，重试命令")

            try:
                import base64

                # conda激活脚本：尝试常见安装路径
                conda_init = (
                    'for p in /opt/miniconda3 "$HOME/miniconda3" "$HOME/anaconda3" '
                    '/opt/anaconda3 /usr/local/miniconda3 /usr/local/anaconda3; do '
                    'if [ -f "$p/etc/profile.d/conda.sh" ]; then '
                    '. "$p/etc/profile.d/conda.sh" 2>/dev/null; '
                    'conda activate base 2>/dev/null; '
                    'break; '
                    'fi; '
                    'done 2>/dev/null'
                )

                # 完整脚本：conda激活 + 原始命令
                full_script = f"{conda_init}; {command}"
                encoded = base64.b64encode(full_script.encode('utf-8')).decode('ascii')
                full_cmd = f"bash --login -c 'eval \"$(echo {encoded} | base64 -d)\"'"

                stdin, stdout, stderr = self.client.exec_command(full_cmd, timeout=timeout)
                exit_code = stdout.channel.recv_exit_status()
                out = stdout.read().decode('utf-8', errors='replace')
                err = stderr.read().decode('utf-8', errors='replace')
                return exit_code, out, err
            except Exception as e:
                if _ssh_attempt == 0:
                    print(f"[SSH] 命令执行异常({e})，将尝试重连...")
                    continue
                else:
                    print(f"[ERROR] SSH命令执行失败(重连后仍失败): {e}")
                    return -1, "", str(e)

    def run_command_with_env(self, command: str, timeout: int = None) -> Tuple[int, str, str]:
        """
        执行远程命令（带环境初始化）
        先source环境变量，再执行命令
        """
        env_prefix = " && ".join(NPU_CONFIG['env_setup_commands'])
        full_command = f"{env_prefix} && {command}"
        return self.run_command(full_command, timeout)

    def upload_file_content(self, remote_path: str, content: str) -> bool:
        """通过SFTP上传文件内容到远程"""
        for _sftp_attempt in range(2):
            if _sftp_attempt > 0:
                if not self.ensure_connected():
                    return False
            if not self._connected:
                if not self.ensure_connected():
                    return False

            try:
                sftp = self.client.open_sftp()
                with sftp.file(remote_path, 'w') as f:
                    f.write(content)
                sftp.close()
                return True
            except Exception as e:
                if _sftp_attempt < 1:
                    print(f"[SSH] SFTP上传异常({e})，将尝试重连...")
                    continue
                else:
                    print(f"[ERROR] SFTP上传失败 {remote_path}: {e}")
                    return False

    def download_file_content(self, remote_path: str) -> Optional[str]:
        """通过SFTP下载远程文件内容"""
        for _sftp_attempt in range(2):
            if _sftp_attempt > 0:
                if not self.ensure_connected():
                    return None
            if not self._connected:
                if not self.ensure_connected():
                    return None

            try:
                sftp = self.client.open_sftp()
                with sftp.file(remote_path, 'r') as f:
                    content = f.read().decode('utf-8', errors='replace')
                sftp.close()
                return content
            except Exception as e:
                if _sftp_attempt < 1:
                    print(f"[SSH] SFTP下载异常({e})，将尝试重连...")
                    continue
                else:
                    print(f"[ERROR] SFTP下载失败 {remote_path}: {e}")
                    return None


# ==================== AscendC测试执行器 ====================
class AscendCTestExecutor:
    """
    AscendC算子测试执行器
    通过SSH到310B4真机执行build/run验证

    核心策略：基于310B4上已验证通过的参考算子目录做模板复制，
    只替换kernel代码文件（op_kernel/*.cpp），其余构建/运行基础设施不变。

    流程：
    1. 将参考算子目录（如 <remote_work_dir>/AddCustom）整体复制为测试目录
    2. 仅替换 op_kernel/add.cpp（即LLM生成的kernel代码）
    3. 远程执行 bash build.sh 编译（在 op_kernel 上层目录）
    4. 安装 .run 包到配置的 remote_opp_packages
    5. 修改 AclNNInvocation/src/CMakeLists.txt 的 CUST_PKG_PATH
    6. 远程执行 bash run.sh 运行验证
    7. 解析运行输出，判断是否通过 Precision!
    """

    # 算子名 → 参考算子目录映射（310B4上已验证通过的）
    # 参考算子需预先部署到配置的远程工作目录，作为测试模板。
    REFERENCE_OP_MAP = {
        'ascend_add': 'AddCustom',
        'ascend_relu': 'ReluCustom',
        'ascend_sigmoid': 'SigmoidCustom',
        'ascend_fmod': 'FmodCustom',
        'ascend_asinh': 'AsinhCustom',
    }

    # 算子名 → kernel入口名（op_kernel目录下的cpp文件名）
    KERNEL_FILE_MAP = {
        'ascend_add': 'add.cpp',
        'ascend_relu': 'relu.cpp',
        'ascend_sigmoid': 'sigmoid.cpp',
        'ascend_fmod': 'fmod.cpp',
        'ascend_asinh': 'asinh.cpp',
    }

    # 算子名 → op_kernel上层目录名（如 Add/、Relu/）
    OP_DIR_MAP = {
        'ascend_add': 'Add',
        'ascend_relu': 'Relu',
        'ascend_sigmoid': 'Sigmoid',
        'ascend_fmod': 'Fmod',
        'ascend_asinh': 'Asinh',
    }

    def __init__(self, ssh: SSHExecutor):
        self.ssh = ssh
        self.execution_count = 0
        self.remote_work_dir = NPU_CONFIG['remote_work_dir']
        self.opp_packages_dir = NPU_CONFIG['remote_opp_packages']

    @staticmethod
    def extract_ascendc_code(text: str) -> str:
        """
        从LLM输出中提取AscendC kernel代码

        支持格式:
        - ```cpp ... ``` / ```c++ ... ```
        - ``` ... ```
        - 纯文本（直接返回）
        """
        # 尝试提取 ```cpp 或 ```c++ 代码块
        pattern = r'```(?:cpp|c\+\+)\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # 尝试提取 ``` 代码块
        pattern = r'```\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # 返回原文（可能是纯代码）
        return text.strip()

    def _get_test_dir(self, op_name: str, agent_id: int = None, suffix: str = None) -> str:
        """
        获取测试算子在远程的工作目录路径

        格式：<remote_work_dir>/{OpName}_test_{agent_id} 或 ..._test_{suffix}
        这样多个Agent的测试目录互不冲突
        """
        ref_name = self.REFERENCE_OP_MAP.get(op_name, op_name.replace('ascend_', '') + 'Custom')
        if agent_id is not None:
            return f"{self.remote_work_dir}/{ref_name}_test_{agent_id}"
        elif suffix:
            return f"{self.remote_work_dir}/{ref_name}_test_{suffix}"
        else:
            return f"{self.remote_work_dir}/{ref_name}_test"

    def _get_ref_dir(self, op_name: str) -> str:
        """获取参考算子在远程的目录路径"""
        ref_name = self.REFERENCE_OP_MAP.get(op_name, '')
        return f"{self.remote_work_dir}/{ref_name}"

    def _get_kernel_path(self, op_name: str, test_dir: str) -> str:
        """获取kernel代码文件在远程的路径"""
        op_subdir = self.OP_DIR_MAP.get(op_name, op_name.replace('ascend_', ''))
        kernel_file = self.KERNEL_FILE_MAP.get(op_name, op_name.replace('ascend_', '') + '.cpp')
        return f"{test_dir}/FrameworkLaunch/{op_subdir}/op_kernel/{kernel_file}"

    def _prepare_remote_dir(self, op_name: str, agent_id: int = None, suffix: str = None) -> bool:
        """
        准备远程测试目录：基于参考算子目录整体复制

        不生成任何模板文件，完全复用已验证通过的参考算子基础设施，
        只在后续步骤替换 kernel 代码。

        参数:
            op_name: 算子名称（如 ascend_add）
            agent_id: Agent编号（用于区分不同Agent的测试目录）
            suffix: 目录后缀（如 'neg_K2_round3'）
        """
        test_dir = self._get_test_dir(op_name, agent_id=agent_id, suffix=suffix)
        ref_dir = self._get_ref_dir(op_name)

        # 1. 清理旧测试目录（不影响参考算子）
        exit_code, out, err = self.ssh.run_command(
            f"rm -rf {test_dir} && cp -r {ref_dir} {test_dir}",
            timeout=30
        )
        if exit_code != 0:
            print(f"[ERROR] 复制参考算子目录失败: {err}")
            return False

        # 2. 清理旧编译产物（确保全新编译）
        op_subdir = self.OP_DIR_MAP.get(op_name, op_name.replace('ascend_', ''))
        self.ssh.run_command(
            f"rm -rf {test_dir}/FrameworkLaunch/{op_subdir}/build_out",
            timeout=10
        )
        self.ssh.run_command(
            f"rm -rf {test_dir}/FrameworkLaunch/AclNNInvocation/build",
            timeout=10
        )
        self.ssh.run_command(
            f"rm -rf {test_dir}/FrameworkLaunch/AclNNInvocation/input/*.bin",
            timeout=10
        )
        # 只删output中的bin，保留golden.bin（gen_data.py会重新生成）
        self.ssh.run_command(
            f"find {test_dir}/FrameworkLaunch/AclNNInvocation/output -name '*.bin' ! -name 'golden.bin' -delete",
            timeout=10
        )

        print(f"[PREP] 测试目录已准备: {test_dir} (基于 {ref_dir})")
        return True

    def upload_kernel_code(self, op_name: str, kernel_code: str,
                           agent_id: int = None, suffix: str = None) -> bool:
        """
        上传LLM生成的kernel代码到远程测试目录

        只替换 op_kernel/*.cpp，其余文件不动

        参数:
            op_name: 算子名称
            kernel_code: LLM生成的kernel代码（已提取）
            agent_id: Agent编号（用于定位测试目录）
            suffix: 目录后缀
        """
        test_dir = self._get_test_dir(op_name, agent_id=agent_id, suffix=suffix)
        kernel_path = self._get_kernel_path(op_name, test_dir)

        success = self.ssh.upload_file_content(kernel_path, kernel_code)

        if not success:
            print(f"[ERROR] kernel代码上传失败: {kernel_path}")
            return False

        print(f"[UPLOAD] kernel代码已上传: {kernel_path}")
        return True

    def _fix_cmakelists_cust_path(self, op_name: str, test_dir: str) -> bool:
        """
        修改 AclNNInvocation/src/CMakeLists.txt 中的 CUST_PKG_PATH

        将其改为指向配置的 remote_opp_packages，避免系统目录权限问题
        这与之前AddCustom手动修改的方式一致
        """
        aclnn_dir = f"{test_dir}/FrameworkLaunch/AclNNInvocation/src"
        cust_path = f"{self.opp_packages_dir}/vendors/customize/op_api"

        # 使用sed替换CUST_PKG_PATH（与手动验证时用的一致）
        exit_code, out, err = self.ssh.run_command(
            f"sed -i 's|^set(CUST_PKG_PATH.*|set(CUST_PKG_PATH \"{cust_path}\")|' {aclnn_dir}/CMakeLists.txt",
            timeout=10
        )
        if exit_code != 0:
            print(f"[ERROR] 修改CMakeLists.txt失败: {err}")
            return False

        print(f"[FIX] CMakeLists.txt CUST_PKG_PATH已改为: {cust_path}")
        return True

    def build_operator(self, op_name: str, agent_id: int = None, suffix: str = None) -> Tuple[bool, str]:
        """
        在310B4上编译算子

        两步流程（与AddCustom验证通过的流程一致）：
        1. 在 op_kernel 上层目录执行 bash build.sh → 生成 .run 包
        2. 安装 .run 包到配置的 remote_opp_packages

        返回: (是否成功, 编译输出信息)
        """
        test_dir = self._get_test_dir(op_name, agent_id=agent_id, suffix=suffix)
        op_subdir = self.OP_DIR_MAP.get(op_name, op_name.replace('ascend_', ''))
        build_dir = f"{test_dir}/FrameworkLaunch/{op_subdir}"

        print(f"[BUILD] 开始编译: {op_name}")
        # 将完整编译输出保存到日志文件，避免截断丢失关键错误信息
        build_log = f"{build_dir}/build_full.log"
        exit_code, out, err = self.ssh.run_command_with_env(
            f"cd {build_dir} && bash build.sh 2>&1 | tee {build_log}",
            timeout=EXECUTION_CONFIG['build_timeout']
        )

        # SSH断连保护：如果编译命令因SSH失败返回-1，抛出异常让上层处理
        if exit_code == -1 and not self.ssh._connected:
            raise RuntimeError("SSH连接断开且重连失败，无法继续编译")

        build_output = out + "\n" + err

        # 检查是否编译成功：查找.run包文件（必须用-type f排除CPack目录）
        check_code, check_out, _ = self.ssh.run_command_with_env(
            f"find {build_dir}/build_out -type f -name 'custom_opp_*.run' | head -1",
            timeout=10
        )

        run_file = check_out.strip()
        if not run_file:
            # build.sh没生成.run包，有两种可能：
            # 1. kernel编译失败（LLM代码问题）→ 报告实际C++错误
            # 2. kernel编译成功但CANN 8.0.RC3的binary/config目录bug → 修复后重新打包

            # 检查kernel binary是否已生成（只检查binary/目录，避免op_host产物误判）
            binary_check_code, binary_check_out, _ = self.ssh.run_command(
                f"find {build_dir}/build_out/op_kernel/binary -name '*.json' 2>/dev/null | head -5",
                timeout=10
            )

            if binary_check_out.strip():
                # kernel编译成功了，但binary/config目录缺失（CANN 8.0.RC3已知bug）
                print(f"[FIX] 检测到kernel已编译但binary/config缺失，尝试修复...")
                self.ssh.run_command(
                    f"mkdir -p {build_dir}/build_out/op_kernel/binary/config",
                    timeout=10
                )
                # 重新执行打包
                repack_code, repack_out, repack_err = self.ssh.run_command_with_env(
                    f"cd {build_dir}/build_out && cmake --build . --target package 2>&1 | tee -a {build_log}",
                    timeout=120
                )
                build_output += "\n" + repack_out + "\n" + repack_err

                # 重新检查.run包
                check_code, check_out, _ = self.ssh.run_command_with_env(
                    f"find {build_dir}/build_out -type f -name 'custom_opp_*.run' | head -1",
                    timeout=10
                )
                run_file = check_out.strip()

            if not run_file:
                # 确实编译失败：提取实际的C++编译错误（grep error行 + 末尾日志）
                err_code, err_out, _ = self.ssh.run_command(
                    f"grep -i 'error:' {build_log} 2>/dev/null | head -20; echo '---'; tail -c 2000 {build_log} 2>/dev/null",
                    timeout=10
                )
                full_error = err_out if err_out.strip() else build_output[:3000]
                print(f"[BUILD] 编译失败（kernel代码有误），错误信息:\n{full_error}")
                return False, full_error

        # 安装 .run 包
        # 注意：.run包(makeself格式)不支持--install参数，只支持--install-path=
        # --quiet避免非交互式SSH下的交互提示
        install_code, install_out, install_err = self.ssh.run_command(
            f"chmod +x {run_file} && {run_file} --quiet --install-path={self.opp_packages_dir}",
            timeout=60
        )
        if install_code != 0:
            install_msg = install_out + "\n" + install_err
            print(f"[ERROR] .run包安装失败: {install_msg[:500]}")
            return False, build_output + f"\n[INSTALL ERROR] {install_msg}"

        # 验证安装是否成功：检查set_env.bash是否存在
        verify_code, verify_out, _ = self.ssh.run_command(
            f"ls -la {self.opp_packages_dir}/vendors/customize/bin/set_env.bash 2>/dev/null && echo 'INSTALLED_OK'",
            timeout=10
        )
        if 'INSTALLED_OK' not in verify_out:
            print(f"[ERROR] 算子包安装验证失败：set_env.bash不存在")
            return False, build_output + "\n[INSTALL VERIFY FAILED]"

        print(f"[BUILD] 编译+安装成功，算子包: {run_file}")
        return True, build_output

    def run_operator(self, op_name: str, agent_id: int = None, suffix: str = None) -> Tuple[bool, str]:
        """
        在310B4上运行算子并验证精度

        使用参考算子目录中已验证通过的 run.sh，无需修改

        返回: (是否通过Precision!, 运行输出信息)
        """
        test_dir = self._get_test_dir(op_name, agent_id=agent_id, suffix=suffix)
        aclnn_dir = f"{test_dir}/FrameworkLaunch/AclNNInvocation"

        # 修改CMakeLists.txt的CUST_PKG_PATH（确保指向用户目录）
        self._fix_cmakelists_cust_path(op_name, test_dir)

        print(f"[RUN] 开始运行验证: {op_name}")
        # unset旧的ASCEND_CUSTOM_OPP_PATH + source两个环境 + bash run.sh
        env_cmds = " && ".join(NPU_CONFIG['env_setup_commands'])
        exit_code, out, err = self.ssh.run_command(
            f"unset ASCEND_CUSTOM_OPP_PATH; {env_cmds} && cd {aclnn_dir} && bash run.sh",
            timeout=EXECUTION_CONFIG['run_timeout']
        )

        run_output = out + "\n" + err

        # 判断是否通过：检查输出中是否包含"passed Precision"或"test pass"
        passed = False
        if 'passed Precision' in run_output or 'test pass' in run_output.lower():
            passed = True
            print(f"[RUN] ✓ 精度验证通过!")
        else:
            print(f"[RUN] ✗ 精度验证失败")
            print(f"  输出片段: {run_output[:800]}")

        return passed, run_output

    def execute_test(self, op_name: str, kernel_code: str, problem: Dict,
                     agent_id: int = None, suffix: str = None) -> Tuple[bool, str]:
        """
        完整的测试执行流程：
        准备目录(复制参考算子) → 上传kernel代码 → 编译 → 运行 → 验证

        参数:
            op_name: 算子名称
            kernel_code: LLM生成的kernel代码（原始LLM输出，会自动提取）
            problem: 题目信息
            agent_id: Agent编号（区分不同测试目录）
            suffix: 目录后缀（协商场景中可能需要）

        返回: (是否通过, 输出信息)
        """
        self.execution_count += 1
        # 提取纯代码
        clean_code = self.extract_ascendc_code(kernel_code)

        # 1. 准备远程目录（复制参考算子）
        if not self._prepare_remote_dir(op_name, agent_id=agent_id, suffix=suffix):
            return False, "远程目录准备失败"

        # 2. 上传kernel代码（只替换op_kernel/*.cpp）
        if not self.upload_kernel_code(op_name, clean_code, agent_id=agent_id, suffix=suffix):
            return False, "kernel代码上传失败"

        # 3. 编译
        build_ok, build_output = self.build_operator(op_name, agent_id=agent_id, suffix=suffix)
        if not build_ok:
            return False, f"编译失败:\n{build_output}"

        # 4. 运行验证
        run_ok, run_output = self.run_operator(op_name, agent_id=agent_id, suffix=suffix)
        if run_ok:
            return True, f"精度验证通过:\n{run_output}"
        else:
            return False, f"精度验证失败:\n{run_output}"


# ==================== LLM接口 ====================
class CallBudgetExceeded(RuntimeError):
    """Raised before an LLM call would exceed the experiment cap."""


class AgentLLMInterface:
    """
    多Agent LLM调用接口
    每个Agent使用不同的模型，生成AscendC kernel代码
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.call_count = 0
        self.call_budget = None
        self.force_agent_id = None
        self.clients = {}

        if not use_mock:
            try:
                from openai import OpenAI

                for agent_config in AGENT_MODELS:
                    agent_id = agent_config['agent_id']
                    self.clients[agent_id] = OpenAI(
                        api_key=agent_config['api_key'],
                        base_url=agent_config['base_url'],
                        max_retries=5,       # 客户端内置重试（瞬断）
                        timeout=300.0,       # 单次请求超时放宽到5分钟
                    )
                    print(f"Agent {agent_id} ({agent_config['name']}) 客户端创建成功")

            except ImportError:
                print("警告: 未安装openai库，使用模拟模式")
                print("安装命令: pip install openai")
                self.use_mock = True
            except Exception as e:
                print(f"警告: LLM客户端创建失败: {e}")
                self.use_mock = True

    def call_llm(self, agent_id: int, prompt: str, system_prompt: str = "") -> str:
        """
        调用指定Agent的LLM生成AscendC代码
        """
        if self.call_budget is not None and self.call_count >= self.call_budget:
            raise CallBudgetExceeded(
                f"LLM call budget exhausted ({self.call_count}/{self.call_budget})"
            )
        self.call_count += 1
        effective_agent_id = self.force_agent_id
        if effective_agent_id is None:
            effective_agent_id = agent_id

        if self.use_mock:
            return self._mock_generate(prompt)
        else:
            return self._real_generate(effective_agent_id, prompt, system_prompt)

    def _real_generate(self, agent_id: int, prompt: str, system_prompt: str) -> str:
        """真实LLM调用"""
        if agent_id not in self.clients:
            raise ValueError(f"Agent {agent_id} 客户端不存在")

        agent_config = next(c for c in AGENT_MODELS if c['agent_id'] == agent_id)

        extra = {}
        # 每个agent可在config中设置 enable_thinking（默认按模型名推断）
        et = agent_config.get('enable_thinking')
        if et is None:
            ml = agent_config['model'].lower()
            # qwen3.7-max 系列强制 enable_thinking=True；flash 系列可设 False
            et = False if 'flash' in ml else True
        extra['extra_body'] = {"enable_thinking": et}

        # 全部走流式：代理掉流时流读取立刻报错->触发重试，避免缓冲模式下长响应被代理吞掉后无限挂起（进程空闲被环境杀）
        stream_models = ['qwen3.6-flash', 'qwen3.5-27b', 'glm-4.5', 'glm-4.5-air', 'glm-5', 'minimax-m2.1', 'glm-4.6', 'qwq-plus']
        use_stream = True

        # 心跳线程：调用期间每5秒打印一次，防止进程因长时间无stdout被环境杀死
        # （尤其首token等待期 / 重试退避 sleep / 流式消费前的静默窗口）
        import threading
        _hb_stop = threading.Event()
        def _heartbeat():
            while not _hb_stop.wait(5):
                print(f"  [心跳] Agent {agent_id} 调用中...", flush=True)
        _hb = threading.Thread(target=_heartbeat, daemon=True)
        _hb.start()
        try:
            # 重试：覆盖 create() + 流式消费全过程
            # 旧版只重试 create()，流式 for chunk in response 中途断开（httpx.RemoteProtocolError）会直接崩溃
            # 新版把流式消费也放进重试循环，任何阶段断开都从头重试
            from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
            import httpx
            last_err = None
            import concurrent.futures
            WALL_TIMEOUT = 300  # 墙钟硬超时(秒)：根治代理半开连接导致流式读超时失效、主线程永久挂起的问题
            for _attempt in range(1, 7):  # 1次初始 + 最多5次重试
                def _worker():
                    response = self.clients[agent_id].chat.completions.create(
                        model=agent_config['model'],
                        messages=[
                            {"role": "system", "content": system_prompt or "You are an AscendC programming expert for Huawei NPU. Write correct, efficient AscendC kernel code."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=agent_config['temperature'],
                        max_tokens=agent_config['max_tokens'],
                        timeout=180,  # AscendC代码生成可能较长（流式读超时，双保险之一）
                        stream=use_stream,
                        **extra
                    )
                    # 流式消费也在 worker 内——中途断开则整个 create+stream 从头重试
                    if use_stream:
                        content_chunks = []
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content:
                                content_chunks.append(chunk.choices[0].delta.content)
                        return "".join(content_chunks)
                    else:
                        return response.choices[0].message.content
                _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    _fut = _ex.submit(_worker)
                    try:
                        return _fut.result(timeout=WALL_TIMEOUT)  # 墙钟兜底：任何挂起都在此被打破
                    except concurrent.futures.TimeoutError:
                        # 墙钟超时：放弃本轮，后台 worker 线程随进程退出回收（不 join 避免阻塞）
                        _wait = min(30, 5 * (2 ** (_attempt - 1)))
                        print(f"  [重试] Agent {agent_id} 调用墙钟超时({WALL_TIMEOUT}s, 第{_attempt}次); {_wait}s后重试", flush=True)
                        time.sleep(_wait)
                        continue
                except (APIConnectionError, APITimeoutError, RateLimitError) as _e:
                    last_err = _e
                    _wait = min(60, 5 * (2 ** (_attempt - 1)))  # 5,10,20,40,60s 指数退避
                    print(f"  [重试] Agent {agent_id} API连接错误(第{_attempt}次): {_e}; {_wait}s后重试", flush=True)
                    time.sleep(_wait)
                except APIStatusError as _e:
                    _code = getattr(_e, 'status_code', 0)
                    if _code and _code >= 500:
                        last_err = _e
                        _wait = min(60, 5 * (2 ** (_attempt - 1)))
                        print(f"  [重试] Agent {agent_id} 服务端5xx(第{_attempt}次): {_e}; {_wait}s后重试", flush=True)
                        time.sleep(_wait)
                    else:
                        raise  # 4xx 鉴权/参数错误不重试
                except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                        httpx.ReadTimeout, httpx.ConnectTimeout) as _e:
                    # 流式读取中途连接断开——旧版未捕获导致整个实验崩溃
                    last_err = _e
                    _wait = min(60, 5 * (2 ** (_attempt - 1)))
                    print(f"  [重试] Agent {agent_id} 流式读取异常(第{_attempt}次): {type(_e).__name__}: {_e}; {_wait}s后重试", flush=True)
                    time.sleep(_wait)
                except Exception as _e:
                    # 兜底：其他未知异常也重试（如 SSE 解析错误），避免单次抖动崩溃整个实验
                    last_err = _e
                    _wait = min(60, 5 * (2 ** (_attempt - 1)))
                    print(f"  [重试] Agent {agent_id} 未知异常(第{_attempt}次): {type(_e).__name__}: {_e}; {_wait}s后重试", flush=True)
                    time.sleep(_wait)
                finally:
                    # 正常返回/异常/墙钟超时分支都释放线程池（worker 已完成或已抛异常，不阻塞进程退出）
                    _ex.shutdown(wait=False)
            # 所有重试耗尽，抛出最后一次错误
            raise last_err
        finally:
            _hb_stop.set()
            _hb.join(timeout=1)

    def _mock_generate(self, prompt: str) -> str:
        """模拟LLM生成AscendC代码"""
        # 返回一个简单的mock kernel（仅用于测试流程）
        return """#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X> class KernelMock {
    public:
        __aicore__ inline KernelMock() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR y,
            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain,
            TPipe * pipeIn) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
            this->tileLength = block_size;
            this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
            auto startPointer = core_size * GetBlockIdx();
            auto bufferlength = this->blockLength;
            Gm_x1.SetGlobalBuffer((__gm__ TYPE_X*)x1 + startPointer, bufferlength);
            Gm_y.SetGlobalBuffer((__gm__ TYPE_X*)y + startPointer, bufferlength);
            this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
            pipe = pipeIn;
            pipe->InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
            pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        }
        __aicore__ inline void Process() {
            int32_t loopCount = this->tileNum;
            for (int32_t i = 0; i < loopCount - 1; i++) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t loopIdx) {}
        __aicore__ inline void Compute(int32_t loopIdx) {}
        __aicore__ inline void CopyOut(int32_t loopIdx) {}
        TPipe* pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1;
        TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
        GlobalTensor<TYPE_X> Gm_x1, Gm_y;
        LocalTensor<TYPE_X> x1Local, yLocal;
        uint32_t blockLength, tileLength, tileNum, ALIGN_NUM;
};

extern "C" __global__ __aicore__ void mock(GM_ADDR x1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    // Mock implementation
}
"""


# ==================== 实验结果 ====================
@dataclass
class ExperimentResult:
    """实验结果"""
    mode: str
    N: int
    K: int
    problem_name: str
    passed: bool
    winner_agent_id: int
    total_rounds: int
    total_llm_calls: int
    execution_time: float
    generated_code: str = ""
    test_output: str = ""
    negotiation_history: List[Dict] = field(default_factory=list)
    speech_log: List[Dict] = field(default_factory=list)
    # 新增：逐Agent完整记录（代码+编译/运行错误），方便分析失败原因
    agent_details: List[Dict] = field(default_factory=list)


# ==================== 实验运行器 ====================
class ExperimentRunner:
    """AscendC算子实验运行器"""

    def __init__(self, use_mock_llm: bool = True, simulate: bool = False):
        self.use_mock_llm = use_mock_llm
        self.simulate = simulate
        self.llm = AgentLLMInterface(use_mock=use_mock_llm)
        if simulate:
            # 模拟后端：无需 SSH / 真机，结果可复现（见 simulated_executor.py）
            self.ssh = None
            self.executor = SimulatedExecutor()
        else:
            self.ssh = SSHExecutor()
            self.executor = AscendCTestExecutor(self.ssh)
        self.dataset = self._load_dataset()
        self.results: List[ExperimentResult] = []
        self.max_fix_rounds = EXECUTION_CONFIG['max_fix_rounds']

    def _load_dataset(self) -> List[Dict]:
        """加载AscendC算子数据集"""
        dataset_path = DATASET_CONFIG['dataset_path']
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(HERE, dataset_path)
        if not os.path.exists(dataset_path):
            print(f"错误: 数据集文件不存在: {dataset_path}")
            sys.exit(1)

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 筛选指定来源的算子
        filter_source = DATASET_CONFIG.get('filter_source', '')
        if filter_source:
            data = [d for d in data if d.get('original_source') == filter_source]
            print(f"筛选来源 '{filter_source}': {len(data)} 个算子")

        # 限制样本数量
        num_samples = DATASET_CONFIG['num_samples']
        return data[:num_samples]

    def _build_prompt(self, problem: Dict, context: str = "") -> str:
        """构建LLM提示词"""
        # AscendC算子的prompt字段已包含完整描述和要求
        prompt = problem.get('prompt', '')

        # 如果有参考kernel作为上下文（用于协商改进）
        if context:
            prompt = f"{context}\n\n---\n\n{prompt}"

        return prompt

    def _build_system_prompt(self, agent_name: str, problem: Dict) -> str:
        """构建系统提示词"""
        category = problem.get('category', 'unknown')
        description = problem.get('description', '')

        return f"""You are {agent_name}, an AscendC programming expert for Huawei NPU (Ascend 310B4).
Your task is to write correct AscendC operator kernel code.

Guidelines:
1. Use the AscendC programming model (Kernel class with Init() and Process())
2. Implement extern "C" __global__ __aicore__ void entry function
3. Use TPipe, TQue, LocalTensor, GlobalTensor APIs correctly
4. Handle data alignment (32B/ALIGN_NUM) properly
5. Support float16/float/int32 data types with proper template specialization
6. Use double buffering (BUFFER_NUM = 2) for performance
7. Ensure proper memory management (pipe->InitBuffer for all queues and temp buffers)

Operator: {description}
Category: {category}

Write ONLY the kernel code (header file content), no explanation needed.
"""

    def connect_npu(self) -> bool:
        """连接310B4真机"""
        print("\n[INIT] 正在连接310B4真机...")
        connected = self.ssh.connect()

        if connected:
            # 验证环境
            exit_code, out, err = self.ssh.run_command_with_env(
                "npu-smi info | head -5",
                timeout=10
            )
            if exit_code == 0:
                print(f"[INIT] NPU环境验证成功:\n{out[:200]}")
            else:
                print(f"[WARN] NPU环境验证异常: {err}")

        return connected

    def disconnect_npu(self):
        """断开310B4连接"""
        if self.simulate or self.ssh is None:
            return
        self.ssh.disconnect()

    # ==================== 编译反馈循环 ====================
    def _extract_core_error(self, output: str, max_lines: int = 50) -> str:
        """
        提取核心编译/运行错误，限制在max_lines行内，避免冗余日志淹没真错误。
        优先保留含 'error' 的行，再补末尾上下文。
        """
        if not output or not output.strip():
            return "(无错误输出)"

        lines = output.split('\n')
        error_lines = [l for l in lines if 'error' in l.lower()]
        other_lines = [l for l in lines if 'error' not in l.lower()]

        selected = error_lines[:30]
        tail = other_lines[-15:] if len(other_lines) > 15 else other_lines
        result = selected
        if tail and tail != selected:
            result = result + ['', '--- 末尾上下文 ---'] + tail
        result = result[:max_lines]
        return '\n'.join(result)

    def _build_fix_system_prompt(self, agent_name: str, problem: Dict) -> str:
        """专业修复引导：要求逐条修复编译错误，不重写整个代码"""
        category = problem.get('category', 'unknown')
        description = problem.get('description', '')
        return f"""You are {agent_name}, an AscendC programming expert for Huawei NPU (Ascend 310B4).
Your previous AscendC kernel code failed to compile/run. Fix the compilation errors line by line.
Do NOT rewrite the entire code - only modify the lines causing errors.

Common fixes:
- Define missing constants: `constexpr int32_t BUFFER_NUM = 2;` and `constexpr int32_t ALIGN_NUM = 32;`
- Use correct header: `#include "kernel_operator.h"`
- Use GET_TILING_DATA macro to get tiling data, and access fields defined in op_host (e.g. ALIGN_NUM, block_size, core_size, core_remain)
- Do NOT cast GM_ADDR directly to uint32_t* in aicore functions
- Use TPipe/TQue/LocalTensor APIs correctly

Write ONLY the kernel code (header file content), no explanation needed.

Operator: {description}
Category: {category}
"""

    def _build_fix_prompt(self, problem: Dict, current_code: str, core_error: str) -> str:
        """构建修复提示词：包含当前代码 + 编译器报错"""
        return f"""你的AscendC算子代码在310B4真机上编译/运行失败。

【算子描述】
{problem.get('description', '')}

【编译器/运行报错】
{core_error}

【你当前的代码】
{current_code}

请逐条修复上述错误，只修改导致错误的那几行，不要重写整个代码。
修复后只输出完整的kernel代码（header file content），不需要解释。"""

    def compile_fix_loop(self, op_name: str, agent_id: int, problem: Dict,
                         initial_code: str, suffix_base: str,
                         max_rounds: int = None) -> Tuple[bool, str, int, str, List[Dict]]:
        """
        编译反馈循环：agent生成代码 → 编译 → 若失败把错误返回给LLM修正 → 再编译 → ... (最多max_rounds次)

        参数:
            op_name: 算子名称
            agent_id: Agent编号（用于调用LLM修复代码）
            problem: 题目信息
            initial_code: 初始生成的代码（原始LLM输出）
            suffix_base: 远程目录后缀前缀（每轮追加 _r{轮次} 保证独立隔离）
            max_rounds: 最大编译修正轮数（默认取 self.max_fix_rounds=10）

        返回: (是否最终通过, 最终代码, 实际用了几轮, 最终测试输出, 每轮编译历史)
            compile_history: [{{round, code, error, passed}}]
        """
        if max_rounds is None:
            max_rounds = self.max_fix_rounds

        compile_history = []
        current_code = initial_code
        final_code = initial_code
        final_output = ""
        passed = False

        agent_name = next((c['name'] for c in AGENT_MODELS if c['agent_id'] == agent_id), f"Agent-{agent_id}")

        for r in range(1, max_rounds + 1):
            # 每轮使用独立远程目录，避免编译产物互相覆盖
            suffix = f"{suffix_base}_r{r}"
            round_passed, round_output = self.executor.execute_test(
                op_name=op_name,
                kernel_code=current_code,
                problem=problem,
                agent_id=None,  # 用suffix区分，不用agent_id
                suffix=suffix
            )

            compile_history.append({
                'round': r,
                'code': current_code,
                'error': '' if round_passed else round_output,
                'passed': round_passed
            })
            final_code = current_code
            final_output = round_output

            if round_passed:
                passed = True
                print(f"    [FIX] Agent {agent_id} ({agent_name}) 第{r}轮编译+验证通过 ✓")
                break

            # 最后一轮失败后不再调用 LLM；生成但不执行的修复既浪费预算，也无法评测。
            if r == max_rounds:
                break

            # 编译/运行失败 → 提取核心错误反馈给LLM修正
            core_error = self._extract_core_error(round_output)
            print(f"    [FIX] Agent {agent_id} ({agent_name}) 第{r}轮失败，提取错误({len(core_error)}字符)用于修正...")

            fix_prompt = self._build_fix_prompt(problem, current_code, core_error)
            fix_system = self._build_fix_system_prompt(agent_name, problem)
            current_code = self.llm.call_llm(agent_id, fix_prompt, fix_system)

        if not passed:
            print(f"    [FIX] Agent {agent_id} ({agent_name}) 达到上限{max_rounds}轮仍未通过")

        return passed, final_code, len(compile_history), final_output, compile_history


    def run_baseline(self, problem: Dict, single_agent: int = None) -> ExperimentResult:
        """
        模式1: Baseline - 单轮取最优Agent
        每个Agent独立生成kernel，在310B4上分别build/run验证

        参数:
            single_agent: 如果指定，只测试该Agent（用于快速验证流程）
        """
        start_time = time.time()
        N = EXPERIMENT_CONFIG['N']

        # 确定要测试的Agent列表
        if single_agent is not None:
            agent_ids = [single_agent]
            test_label = f"单Agent验证 (Agent {single_agent})"
        else:
            agent_ids = list(range(N))
            test_label = f"N={N} 全部Agent"

        print(f"\n{'='*60}")
        print(f"[Baseline] {test_label}, 无协商")
        print(f"  算子: {problem['name']} ({problem.get('description', '')})")
        print(f"{'='*60}")

        agent_results = []
        baseline_speech_log = []

        for i in agent_ids:
            agent_name = next((c['name'] for c in AGENT_MODELS if c['agent_id'] == i), f"Agent-{i}")
            print(f"\n  测试 {agent_name} (Agent {i})...")

            # ---- 断点续跑：该Agent已完成则直接恢复，避免重跑 ----
            ckpt_path = os.path.join(EXECUTION_CONFIG['output_dir'],
                                     f"ckpt_baseline_{problem['name']}_agent{i}.json")
            if os.path.exists(ckpt_path):
                try:
                    with open(ckpt_path, encoding='utf-8') as _cf:
                        done = json.load(_cf)
                    agent_results.append(done)
                    baseline_speech_log.append({
                        'round': 0, 'phase': 'baseline', 'agent_id': i,
                        'agent_name': agent_name, 'type': 'initial',
                        'content': done.get('initial_solution', done.get('code', '')),
                        'weight': 1.0 / N, 'test_passed': done['passed'],
                    })
                    print(f"  [跳过] {agent_name} 已完成 (passed={done['passed']})，从检查点恢复")
                    continue
                except Exception as _ce:
                    print(f"  [警告] 检查点读取失败，重跑 {agent_name}: {_ce}")

            try:
                # SSH若已断开则重连（防止长任务中连接被回收）；模拟模式跳过
                if not self.simulate and not self.ssh._connected:
                    print(f"  [重连] Agent {i} 前重新连接NPU...")
                    self.connect_npu()

                prompt = self._build_prompt(problem)
                system_prompt = self._build_system_prompt(agent_name, problem)
                solution = self.llm.call_llm(i, prompt, system_prompt)

                # 编译反馈循环：生成后编译，失败则把错误返回给LLM修正，最多10轮
                passed, final_code, fix_rounds, test_output, compile_history = self.compile_fix_loop(
                    op_name=problem['name'],
                    agent_id=i,
                    problem=problem,
                    initial_code=solution,
                    suffix_base=f"bl_a{i}"
                )

                agent_result = {
                    'agent_id': i,
                    'agent_name': agent_name,
                    'passed': passed,
                    'code': final_code,         # 最终修复后的完整LLM输出
                    'extracted_code': self.executor.extract_ascendc_code(final_code),  # 提取的纯代码
                    'test_output': test_output,  # 最终编译/运行输出（不截断）
                    'code_length': len(final_code),
                    'fix_rounds': fix_rounds,   # 编译修正轮数
                    'compile_history': compile_history,  # 每轮编译历史
                    'initial_solution': solution,  # 供检查点恢复speech_log
                }
                agent_results.append(agent_result)

                baseline_speech_log.append({
                    'round': 0,
                    'phase': 'baseline',
                    'agent_id': i,
                    'agent_name': agent_name,
                    'type': 'initial',
                    'content': solution,
                    'weight': 1.0 / N,
                    'test_passed': passed,
                })

                status = "[OK] 通过" if passed else "[FAIL] 失败"
                print(f"    {agent_name}: {status}")

                # 写检查点（正常运行完成，含编译失败但未崩溃的情况）
                try:
                    with open(ckpt_path, 'w', encoding='utf-8') as _cf:
                        json.dump(agent_result, _cf, ensure_ascii=False, indent=2)
                except Exception as _we:
                    print(f"  [警告] 检查点写入失败: {_we}")

            except Exception as _ae:
                # 运行期意外异常（SSH/网络）：写检查点标记该Agent已完成(passed=False)
                # 避免wrapper因无检查点而无限重试同一Agent；如需重跑可手动删除该ckpt
                print(f"  [错误] {agent_name} 运行异常: {_ae}（写检查点，标记失败）")
                fail_rec = {
                    'agent_id': i,
                    'agent_name': agent_name,
                    'passed': False,
                    'code': '',
                    'extracted_code': '',
                    'test_output': f'EXCEPTION: {_ae}',
                    'code_length': 0,
                    'fix_rounds': 0,
                    'compile_history': [],
                }
                agent_results.append(fail_rec)
                try:
                    with open(ckpt_path, 'w', encoding='utf-8') as _cf:
                        json.dump(fail_rec, _cf, ensure_ascii=False, indent=2)
                    print(f"  [检查点] {agent_name} 失败记录已写入，避免重复重试")
                except Exception as _we:
                    print(f"  [警告] 检查点写入失败: {_we}")

        # 选最优Agent
        winner = next((r for r in agent_results if r['passed']), agent_results[0])
        passed = winner['passed']

        execution_time = time.time() - start_time

        baseline_history = [{
            'round': 1,
            'phase': 'Baseline独立测试',
            'agent_results': [
                {
                    'agent_id': r['agent_id'],
                    'agent_name': r['agent_name'],
                    'passed': r['passed'],
                    'test_output': r['test_output'][:200]
                }
                for r in agent_results
            ],
            'action': f"选出最优Agent: {winner['agent_name']}",
            'winner': winner['agent_id']
        }]

        return ExperimentResult(
            mode='baseline',
            N=N,
            K=N,
            problem_name=problem['name'],
            passed=passed,
            winner_agent_id=winner['agent_id'],
            total_rounds=1,
            total_llm_calls=self.llm.call_count,
            execution_time=execution_time,
            generated_code=winner['code'],
            test_output=winner['test_output'],
            negotiation_history=baseline_history,
            speech_log=baseline_speech_log,
            agent_details=agent_results  # 完整逐Agent记录
        )

    def run_negotiation(self, problem: Dict, K: int) -> ExperimentResult:
        """
        运行协商框架
        协商逻辑沿用framework.py（通用），测试执行改为AscendCTestExecutor
        支持中间检查点：每个Agent完成init/refine/vote后保存进度，崩溃可恢复
        """
        start_time = time.time()
        N = EXPERIMENT_CONFIG['N']
        alpha = EXPERIMENT_CONFIG['alpha']
        gamma = EXPERIMENT_CONFIG['gamma']

        # === 中间检查点支持 ===
        neg_ckpt_intermediate = os.path.join(
            EXECUTION_CONFIG['output_dir'],
            f"ckpt_neg_intermediate_{problem['name']}_K{K}.json"
        )

        # 尝试加载中间检查点
        restored_speech_log = []
        if os.path.exists(neg_ckpt_intermediate):
            try:
                with open(neg_ckpt_intermediate, encoding='utf-8') as _nf:
                    _ck = json.load(_nf)
                restored_speech_log = _ck.get('speech_log', [])
                print(f"  [恢复] 发现中间检查点，已恢复 {len(restored_speech_log)} 条记录，从崩溃点继续")
            except Exception as _e:
                print(f"  [警告] 读取中间检查点失败: {_e}，从头开始")
                restored_speech_log = []

        # 从speech_log构建缓存（key = type_round_agentId）
        init_cache = {}    # {agent_id: solution}
        refine_cache = {}  # {"r{round}_a{agent_id}": solution}
        vote_cache = {}    # {"r{round}_a{agent_id}": [voted_ids]}
        for _entry in restored_speech_log:
            if _entry['type'] == 'initial':
                init_cache[_entry['agent_id']] = _entry['content']
            elif _entry['type'] == 'refine':
                refine_cache[f"r{_entry['round']}_a{_entry['agent_id']}"] = _entry['content']
            elif _entry['type'] == 'vote':
                vote_cache[f"r{_entry['round']}_a{_entry['agent_id']}"] = _entry.get('voted_for', [])

        speech_log = list(restored_speech_log)  # 继续在恢复的基础上追加

        framework = MultiAgentNegotiationFramework(
            N=N, K=K, alpha=alpha, gamma=gamma
        )

        def _save_intermediate_ckpt(phase_label):
            """保存中间检查点（每个agent完成后调用）"""
            try:
                with open(neg_ckpt_intermediate, 'w', encoding='utf-8') as _nf:
                    json.dump({
                        'problem': problem['name'],
                        'K': K,
                        'last_update': f"{phase_label} @ {time.strftime('%H:%M:%S')}",
                        'speech_log_entries': len(speech_log),
                        'speech_log': speech_log,
                    }, _nf, ensure_ascii=False, indent=2)
            except Exception as _e:
                print(f"  [警告] 保存中间检查点失败: {_e}")

        def get_agent_name(agent_id):
            return next((c['name'] for c in AGENT_MODELS if c['agent_id'] == agent_id), f"Agent-{agent_id}")

        def llm_initial(agent_id, phase):
            """初始代码生成 + 编译反馈循环"""
            # 检查缓存：如果该agent的init已完成，直接返回
            if agent_id in init_cache:
                _cached = init_cache[agent_id]
                print(f"  [INIT-CACHE] Agent {agent_id} 从缓存恢复初始方案 ({len(_cached)}字符)")
                return _cached

            agent_name = get_agent_name(agent_id)
            print(f"  [INIT] {agent_name} (Agent {agent_id}) 正在生成初始方案...")

            prompt = self._build_prompt(problem)
            system_prompt = self._build_system_prompt(agent_name, problem)

            # SSH断连保护：如果编译过程中SSH断开，等待后重试整个生成+编译流程
            for _ssh_retry in range(3):
                try:
                    solution = self.llm.call_llm(agent_id, prompt, system_prompt)

                    # 编译反馈循环：初始生成后编译，失败则修正，最多10轮
                    passed, final_code, fix_rounds, test_output, compile_history = self.compile_fix_loop(
                        op_name=problem['name'],
                        agent_id=agent_id,
                        problem=problem,
                        initial_code=solution,
                        suffix_base=f"negK{K}_init_a{agent_id}"
                    )
                    break  # 成功，跳出重试循环
                except RuntimeError as _ssh_err:
                    if "SSH" in str(_ssh_err) and _ssh_retry < 2:
                        _wait = (_ssh_retry + 1) * 60
                        print(f"  [SSH-RETRY] SSH断连，等待{_wait}秒后重试 (第{_ssh_retry + 1}/3次)...")
                        time.sleep(_wait)
                        # 强制重连
                        if not self.executor.ssh.ensure_connected():
                            print(f"  [SSH-RETRY] 重连仍失败，继续等待...")
                            time.sleep(60)
                    else:
                        raise

            speech_log.append({
                'round': 0,
                'phase': '初始方案',
                'agent_id': agent_id,
                'agent_name': agent_name,
                'type': 'initial',
                'content': final_code,
                'fix_rounds': fix_rounds,
                'compile_history': compile_history,
                'weight': 1.0 / N,
            })

            # 保存中间检查点
            _save_intermediate_ckpt(f"init_a{agent_id}")

            print(f"    [OK] {agent_name} 初始+编译反馈完成 (修正{fix_rounds}轮, 通过={passed}, {len(final_code)}字符)")
            return final_code

        def llm_refine(agent, alive_agents, weight_distribution):
            """方案反思与改进"""
            _cache_key = f"r{framework.round + 1}_a{agent.agent_id}"
            if _cache_key in refine_cache:
                _cached = refine_cache[_cache_key]
                print(f"  [REFINE-CACHE] Agent {agent.agent_id} 从缓存恢复改进方案 (轮次{framework.round + 1})")
                return _cached

            agent_name = get_agent_name(agent.agent_id)
            print(f"  [REFINE] {agent_name} 正在反思改进方案...")

            # 构建权重分布信息
            weight_info = "\n当前权重分布:\n"
            for a in alive_agents:
                a_name = get_agent_name(a.agent_id)
                weight_info += f"  - {a_name}: W={a.weight:.4f} (Γ={a.history_consistency:.4f})\n"

            # 收集其他Agent的方案作为参考
            context = "其他Agent的方案参考:\n"
            for other in alive_agents:
                if other.agent_id != agent.agent_id and other.solution:
                    other_name = get_agent_name(other.agent_id)
                    context += f"\n【{other_name}】(权重={other.weight:.4f}):\n{other.solution[:500]}...\n"

            # 提供参考kernel（从数据集）作为补充
            reference_kernel = problem.get('reference_kernel', '')
            if reference_kernel:
                context += f"\n参考实现（正确版本）:\n{reference_kernel[:800]}...\n"

            prompt = weight_info + "\n" + context + "\n" + self._build_prompt(problem)

            system_prompt = f"""You are {agent_name}. You have weight {agent.weight:.4f}.
Refine your AscendC kernel solution based on others' solutions and the weight distribution.
Focus on fixing compilation or runtime errors, improving data alignment, and ensuring correct buffer management.
"""

            solution = self.llm.call_llm(agent.agent_id, prompt, system_prompt)

            # SSH断连保护：refine的编译反馈循环也可能遇到SSH断连
            for _ssh_retry in range(3):
                try:
                    # 编译反馈循环：refine后编译，失败则修正，最多10轮
                    passed, final_code, fix_rounds, test_output, compile_history = self.compile_fix_loop(
                        op_name=problem['name'],
                        agent_id=agent.agent_id,
                        problem=problem,
                        initial_code=solution,
                        suffix_base=f"negK{K}_r{framework.round}_a{agent.agent_id}"
                    )
                    break  # 成功，跳出重试循环
                except RuntimeError as _ssh_err:
                    if "SSH" in str(_ssh_err) and _ssh_retry < 2:
                        _wait = (_ssh_retry + 1) * 60
                        print(f"  [SSH-RETRY] SSH断连，等待{_wait}秒后重试 (第{_ssh_retry + 1}/3次)...")
                        time.sleep(_wait)
                        if not self.executor.ssh.ensure_connected():
                            print(f"  [SSH-RETRY] 重连仍失败，继续等待...")
                            time.sleep(60)
                    else:
                        raise

            speech_log.append({
                'round': framework.round + 1,
                'phase': '淘汰期' if framework.is_elimination_phase else '终局期',
                'agent_id': agent.agent_id,
                'agent_name': agent_name,
                'type': 'refine',
                'content': final_code,
                'fix_rounds': fix_rounds,
                'compile_history': compile_history,
                'weight': agent.weight,
            })

            # 保存中间检查点
            _save_intermediate_ckpt(f"refine_r{framework.round + 1}_a{agent.agent_id}")

            print(f"    [OK] {agent_name} 改进+编译反馈完成 (修正{fix_rounds}轮, 通过={passed}, {len(final_code)}字符)")
            return final_code

        def llm_vote(agent, alive_agents, top_k, weight_distribution):
            """交叉投票"""
            _cache_key = f"r{framework.round + 1}_a{agent.agent_id}"
            if _cache_key in vote_cache:
                _cached = vote_cache[_cache_key]
                print(f"  [VOTE-CACHE] Agent {agent.agent_id} 从缓存恢复投票 (轮次{framework.round + 1})")
                return _cached

            agent_name = get_agent_name(agent.agent_id)
            print(f"  [VOTE] {agent_name} (Agent {agent.agent_id}) 正在投票 (Top-{top_k})...")

            weight_info = "当前权重分布:\n"
            for a in alive_agents:
                a_name = get_agent_name(a.agent_id)
                weight_info += f"  - {a_name}: W={a.weight:.4f}\n"

            vote_prompt = f"""You are {agent_name}. Your current weight is {agent.weight:.4f}.

{weight_info}

Evaluate the following AscendC kernel solutions and vote for the best {top_k} (excluding yourself).
Consider: code correctness, proper AscendC API usage, buffer management, and data alignment.

Your own solution (weight={agent.weight:.4f}):
{agent.solution[:500] if agent.solution else "Not generated yet"}

Other solutions:
"""
            for other in alive_agents:
                if other.agent_id != agent.agent_id:
                    other_name = get_agent_name(other.agent_id)
                    vote_prompt += f"\n【{other_name}】(weight={other.weight:.4f}):\n{other.solution[:500] if other.solution else 'Not generated yet'}\n"

            vote_prompt += f"\nReply with ONLY the agent IDs (comma-separated) of the best {top_k} solutions."
            vote_prompt += f"\nIMPORTANT: You MUST ONLY use these valid agent IDs: {', '.join([str(a.agent_id) for a in alive_agents if a.agent_id != agent.agent_id])}."

            response = self.llm.call_llm(agent.agent_id, vote_prompt, "You are a code reviewer. Reply with ONLY agent IDs.")

            print(f"    [DEBUG] {agent_name} 原始响应: {response[:150]}...")

            # 解析投票结果
            try:
                valid_ids = [a.agent_id for a in alive_agents if a.agent_id != agent.agent_id]
                voted_ids = []

                numbers = re.findall(r'\b(\d+)\b', response)
                for n in numbers:
                    num = int(n)
                    if num in valid_ids:
                        voted_ids.append(num)

                if not voted_ids:
                    for other in alive_agents:
                        if other.agent_id != agent.agent_id:
                            other_name = get_agent_name(other.agent_id)
                            if other_name in response:
                                voted_ids.append(other.agent_id)

                voted_ids = list(dict.fromkeys(voted_ids))[:top_k]

                if voted_ids:
                    result = voted_ids
                    print(f"    [OK] {agent_name} 投票给: {[get_agent_name(v) for v in result]}")
                else:
                    result = sorted(valid_ids)[:min(top_k, len(valid_ids))]
                    print(f"    [WARN] {agent_name} 未提取到有效ID，使用确定性回退: {[get_agent_name(v) for v in result]}")
            except Exception as e:
                valid_ids = [a.agent_id for a in alive_agents if a.agent_id != agent.agent_id]
                result = sorted(valid_ids)[:min(top_k, len(valid_ids))]
                print(f"    [ERROR] {agent_name} 投票解析异常({e})，使用确定性回退")

            speech_log.append({
                'round': framework.round + 1,
                'phase': '淘汰期' if framework.is_elimination_phase else '终局期',
                'agent_id': agent.agent_id,
                'agent_name': agent_name,
                'type': 'vote',
                'content': '',
                'voted_for': result,
                'vote_type': f'Top-{top_k}' if top_k > 1 else 'Best-1',
                'weight': agent.weight,
                'raw_response': response[:300],
            })

            # 保存中间检查点
            _save_intermediate_ckpt(f"vote_r{framework.round + 1}_a{agent.agent_id}")

            return result

        # 运行协商
        winner = framework.run(llm_initial, llm_refine, llm_vote)

        # 胜出方案进入与 baseline 相同的编译反馈闭环。
        passed, final_code, fix_rounds, test_output, compile_history = self.compile_fix_loop(
            op_name=problem['name'],
            agent_id=winner.agent_id,
            problem=problem,
            initial_code=winner.solution,
            suffix_base=f'neg_K{K}_a{winner.agent_id}',
        )

        execution_time = time.time() - start_time

        # 协商完成，清理中间检查点
        if os.path.exists(neg_ckpt_intermediate):
            try:
                os.remove(neg_ckpt_intermediate)
                print(f"  [清理] 中间检查点已删除（协商完成）")
            except Exception:
                pass

        return ExperimentResult(
            mode='K=N' if K == N else f'K={K}',
            N=N,
            K=K,
            problem_name=problem['name'],
            passed=passed,
            winner_agent_id=winner.agent_id,
            total_rounds=framework.round,
            total_llm_calls=self.llm.call_count,
            execution_time=execution_time,
            generated_code=final_code,
            test_output=test_output[:500],
            negotiation_history=framework.history,
            speech_log=speech_log,
            agent_details=[{
                'agent_id': winner.agent_id,
                'agent_name': get_agent_name(winner.agent_id),
                'passed': passed,
                'code': final_code,
                'extracted_code': self.executor.extract_ascendc_code(final_code),
                'test_output': test_output,
                'fix_rounds': fix_rounds,
                'compile_history': compile_history,
            }],
        )

    def run_single_problem(self, problem: Dict):
        """运行单个算子的四种模式对比"""
        print(f"\n{'#'*80}")
        print(f"# 算子: {problem['name']} ({problem.get('description', '')})")
        print(f"{'#'*80}")

        # 模式1: Baseline
        self.llm.call_count = 0
        result_baseline = self.run_baseline(problem)
        self.results.append(result_baseline)
        print(f"\n[Baseline] 通过: {result_baseline.passed}, 耗时: {result_baseline.execution_time:.2f}s")

        # 模式2: K=N
        self.llm.call_count = 0
        result_kn = self.run_negotiation(problem, K=EXPERIMENT_CONFIG['N'])
        self.results.append(result_kn)
        print(f"\n[K=N] 通过: {result_kn.passed}, 轮数: {result_kn.total_rounds}, 耗时: {result_kn.execution_time:.2f}s")

        # 模式3: K=1
        self.llm.call_count = 0
        result_k1 = self.run_negotiation(problem, K=1)
        self.results.append(result_k1)
        print(f"\n[K=1] 通过: {result_k1.passed}, 轮数: {result_k1.total_rounds}, 耗时: {result_k1.execution_time:.2f}s")

        # 模式4: K=2
        self.llm.call_count = 0
        result_k2 = self.run_negotiation(problem, K=2)
        self.results.append(result_k2)
        print(f"\n[K=2] 通过: {result_k2.passed}, 轮数: {result_k2.total_rounds}, 耗时: {result_k2.execution_time:.2f}s")

    def run_all_experiments(self):
        """运行所有实验"""
        print(f"\n{'#'*80}")
        print(f"# AscendC算子 - 多Agent协商框架对比实验")
        print(f"# N={EXPERIMENT_CONFIG['N']}, 模式: Baseline, K=N, K=1, K=2")
        print(f"# 数据集: {DATASET_CONFIG['dataset_path']}")
        print(f"# 算子数: {len(self.dataset)}")
        print(f"# LLM模式: {'模拟' if self.llm.use_mock else '真实API'}")
        print(f"# NPU: {NPU_CONFIG['host']}:{NPU_CONFIG['port']} ({NPU_CONFIG['soc_version']})")
        print(f"{'#'*80}")

        # 连接310B4（模拟模式跳过）
        if not self.simulate and not self.connect_npu():
            print("[ERROR] 无法连接310B4真机，实验终止")
            return

        try:
            for i, problem in enumerate(self.dataset):
                print(f"\n进度: {i+1}/{len(self.dataset)}")
                self.run_single_problem(problem)

                # 每题保存一次进度（因为310B4执行较慢）
                self.save_results()
        finally:
            # 断开连接
            self.disconnect_npu()

        # 最终统计
        self.print_statistics()

    def save_results(self):
        """保存实验结果（完整记录，不截断）"""
        output_dir = EXECUTION_CONFIG['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存完整JSON结果（含每个Agent的完整代码和编译错误）
        results_file = os.path.join(output_dir, 'results_ascendc.json')
        results_data = []
        for r in self.results:
            entry = {
                'mode': r.mode,
                'N': r.N,
                'K': r.K,
                'problem_name': r.problem_name,
                'passed': r.passed,
                'winner_agent_id': r.winner_agent_id,
                'total_rounds': r.total_rounds,
                'total_llm_calls': r.total_llm_calls,
                'execution_time': r.execution_time,
                'generated_code': r.generated_code,  # 完整代码（不截断）
                'test_output': r.test_output,        # 完整编译/运行输出
                'negotiation_history': r.negotiation_history,
                'speech_log': r.speech_log,
                'agent_details': r.agent_details,    # 逐Agent完整记录
            }
            results_data.append(entry)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {results_file}")

        # 2. 额外保存每个Agent的代码和编译错误到独立文件，方便逐个分析
        for r in self.results:
            if not r.agent_details:
                continue
            op_name = r.problem_name
            mode = r.mode
            for detail in r.agent_details:
                agent_id = detail['agent_id']
                agent_name = detail['agent_name']
                # 保存LLM生成的代码（最终修复后）
                code_file = os.path.join(output_dir, f'{op_name}_{mode}_agent{agent_id}_{agent_name}_code.cpp')
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(detail.get('extracted_code', detail.get('code', '')))
                # 保存最终编译/运行错误日志
                if detail.get('test_output'):
                    log_file = os.path.join(output_dir, f'{op_name}_{mode}_agent{agent_id}_{agent_name}_error.log')
                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.write(detail['test_output'])
                # 保存每轮编译反馈历史（代码 + 错误），便于分析收敛过程
                compile_history = detail.get('compile_history', [])
                if compile_history:
                    for ch in compile_history:
                        cr = ch.get('round', 0)
                        status = 'pass' if ch.get('passed') else 'fail'
                        # 每轮的代码
                        cr_code_file = os.path.join(
                            output_dir,
                            f'{op_name}_{mode}_agent{agent_id}_{agent_name}_r{cr}_{status}_code.cpp'
                        )
                        with open(cr_code_file, 'w', encoding='utf-8') as f:
                            f.write(self.executor.extract_ascendc_code(ch.get('code', '')))
                        # 每轮的错误（仅失败轮有）
                        if ch.get('error'):
                            cr_err_file = os.path.join(
                                output_dir,
                                f'{op_name}_{mode}_agent{agent_id}_{agent_name}_r{cr}_{status}_error.log'
                            )
                            with open(cr_err_file, 'w', encoding='utf-8') as f:
                                f.write(ch['error'])

        print(f"逐Agent代码和错误日志已保存到: {output_dir}/")

    def print_statistics(self):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"AscendC算子实验统计")
        print(f"{'='*80}")

        modes = ['baseline', 'K=N', 'K=1', 'K=2']

        for mode in modes:
            mode_results = [r for r in self.results if r.mode == mode]
            if not mode_results:
                continue

            total = len(mode_results)
            passed = sum(1 for r in mode_results if r.passed)
            pass_rate = passed / total if total > 0 else 0
            avg_time = sum(r.execution_time for r in mode_results) / total
            avg_rounds = sum(r.total_rounds for r in mode_results) / total
            avg_llm_calls = sum(r.total_llm_calls for r in mode_results) / total

            print(f"\n{mode}:")
            print(f"  通过率: {passed}/{total} ({pass_rate:.2%})")
            print(f"  平均耗时: {avg_time:.2f}s")
            print(f"  平均轮数: {avg_rounds:.2f}")
            print(f"  平均LLM调用: {avg_llm_calls:.2f}")

            # 打印通过的算子列表
            passed_ops = [r.problem_name for r in mode_results if r.passed]
            if passed_ops:
                print(f"  通过的算子: {passed_ops}")
            failed_ops = [r.problem_name for r in mode_results if not r.passed]
            if failed_ops:
                print(f"  失败的算子: {failed_ops}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='AscendC算子多Agent协商实验')
    parser.add_argument('--real', action='store_true', help='使用真实LLM API + 310B4 真机（默认模拟执行器，无需硬件）')
    parser.add_argument('--sim', action='store_true', help='强制使用模拟执行器（默认 mock 模式即开启）')
    parser.add_argument('--mode', choices=['baseline', 'all', 'single-baseline', 'negotiation'], default='all',
                        help='实验模式: baseline(只跑baseline), all(跑全部4种), single-baseline(只跑1题baseline验证流程), negotiation(只跑协商K=N/K=1/K=2,不重跑baseline)')
    parser.add_argument('--num', type=int, default=None, help='限制算子数量（默认用config中的num_samples）')
    parser.add_argument('--problem', type=str, default=None, help='只跑指定题目(按problem name,如 ascend_relu)，用于单题协商实验')
    parser.add_argument('--agent', type=int, default=None,
                        help='只测试指定Agent（0-4），用于快速验证流程。如 --agent 0 只跑DeepSeek-V3.1')
    parser.add_argument('--max-fix-rounds', type=int, default=None,
                        help='编译反馈循环上限（默认10）。agent根据编译错误最多修正次数')
    parser.add_argument('--K', type=int, default=None,
                        help='只跑指定K值的协商模式（如 --K 5 只跑K=N，--K 1 只跑K=1，--K 2 只跑K=2）。不指定则跑全部K=N/K=1/K=2')
    parser.add_argument('--ssh-port', type=int, default=None,
                        help='覆盖config中的SSH端口（用于多机并行：进程1用9001，进程2用9002）')
    args = parser.parse_args()

    if not args.real:
        print("Legacy mock performance evaluation is disabled because it cannot measure model quality.")
        print("Use `python run.py demo` for protocol evidence or add `--real` for hardware evaluation.")
        return

    # 覆盖SSH端口（在创建任何SSHExecutor之前生效）
    if args.ssh_port is not None:
        NPU_CONFIG['port'] = args.ssh_port
        print(f"[CONFIG] SSH端口覆盖为: {args.ssh_port}")

    use_mock = not args.real
    simulate = use_mock or args.sim
    if use_mock:
        print("使用模拟LLM（测试模式）")
        print("如需使用真实API，请运行: python experiment_ascendc.py --real")
    else:
        print("使用真实LLM API")

    # 检查paramiko依赖（仅真机模式需要）
    if args.real:
        try:
            import paramiko
            print("paramiko已安装")
        except ImportError:
            print("[ERROR] 未安装paramiko库，请执行: pip install paramiko")
            print("这是SSH连接310B4真机所必需的依赖")
            sys.exit(1)
    else:
        print("使用模拟执行器（无需 310B4 真机 / API Key，结果可复现）；"
              "如需真机请加 --real")

    runner = ExperimentRunner(use_mock_llm=use_mock, simulate=simulate)
    if args.max_fix_rounds is not None:
        runner.max_fix_rounds = args.max_fix_rounds
        print(f"编译反馈循环上限设为: {runner.max_fix_rounds} 轮")

    # 限制算子数量（用于快速测试）
    if args.num:
        runner.dataset = runner.dataset[:args.num]
        print(f"限制算子数量: {args.num}")

    # 只跑指定题目（按 problem name 过滤）
    if args.problem:
        runner.dataset = [p for p in runner.dataset if p.get('name') == args.problem]
        if not runner.dataset:
            print(f"[ERROR] 数据集中未找到题目: {args.problem}")
            return
        print(f"只跑指定题目: {args.problem}")

    if args.mode == 'single-baseline':
        # 只跑1题baseline，验证流程是否通畅
        if args.agent is not None:
            agent_name = next((c['name'] for c in AGENT_MODELS if c['agent_id'] == args.agent), f"Agent-{args.agent}")
            print(f"\n[单Agent验证] 只跑第1题 + Agent {args.agent} ({agent_name})，最快验证SSH+编译+运行流程")
        else:
            print("\n[单题baseline验证] 只跑第1题的baseline模式（全部5个Agent），验证SSH+编译+运行流程")
        if not runner.simulate and not runner.connect_npu():
            print("[ERROR] 无法连接310B4真机，实验终止")
            return
        try:
            problem = runner.dataset[0]
            result = runner.run_baseline(problem, single_agent=args.agent)
            runner.results.append(result)
            print(f"\n结果: 通过={result.passed}, 耗时={result.execution_time:.2f}s")
            if args.agent is not None:
                print(f"测试Agent: {args.agent} ({agent_name})")
            else:
                print(f"胜出Agent: {result.winner_agent_id}")
            runner.save_results()
        finally:
            runner.disconnect_npu()

    elif args.mode == 'baseline':
        # 只跑baseline模式（所有算子）
        if args.agent is not None:
            agent_name = next((c['name'] for c in AGENT_MODELS if c['agent_id'] == args.agent), f"Agent-{args.agent}")
            print(f"\n[Baseline模式] 每个算子只跑Agent {args.agent} ({agent_name})")
        else:
            print("\n[Baseline模式] 每个算子只跑baseline（全部5个Agent无协商）")
        if not runner.simulate and not runner.connect_npu():
            print("[ERROR] 无法连接310B4真机，实验终止")
            return
        try:
            for i, problem in enumerate(runner.dataset):
                print(f"\n进度: {i+1}/{len(runner.dataset)}")
                result = runner.run_baseline(problem, single_agent=args.agent)
                runner.results.append(result)
                runner.save_results()
        finally:
            runner.disconnect_npu()
        runner.print_statistics()

    elif args.mode == 'negotiation':
        # 只跑协商模式(K=N/K=1/K=2)，不重跑baseline，不覆盖 ckpt_baseline_*.json
        from types import SimpleNamespace
        if not runner.simulate and not runner.connect_npu():
            print("[ERROR] 无法连接310B4真机，实验终止")
            return
        try:
            for i, problem in enumerate(runner.dataset):
                print(f"\n进度: {i+1}/{len(runner.dataset)} - {problem['name']}")
                # --K 参数支持单K模式：指定则只跑该K值，不指定则跑全部K=N/K=1/K=2
                K_list = [args.K] if args.K is not None else [EXPERIMENT_CONFIG['N'], 1, 2]
                for K in K_list:
                    neg_ckpt = os.path.join(EXECUTION_CONFIG['output_dir'],
                                            f"ckpt_negotiation_{problem['name']}_K{K}.json")
                    if os.path.exists(neg_ckpt):
                        print(f"  [跳过] {problem['name']} K={K} 已完成（协商检查点存在）")
                        try:
                            with open(neg_ckpt, encoding='utf-8') as _nf:
                                nd = json.load(_nf)
                            runner.results.append(SimpleNamespace(
                                mode=nd.get('mode'), N=nd.get('N'), K=K,
                                problem_name=nd.get('problem'), passed=nd.get('passed'),
                                winner_agent_id=nd.get('winner'), total_rounds=nd.get('total_rounds', 0),
                                total_llm_calls=nd.get('total_llm_calls', 0), execution_time=nd.get('execution_time', 0),
                                generated_code='', test_output='', negotiation_history=[], speech_log=[], agent_details=[]))
                        except Exception as _e:
                            print(f"    [警告] 读回协商检查点失败: {_e}")
                        continue
                    result = runner.run_negotiation(problem, K=K)
                    runner.results.append(result)
                    runner.save_results()
                    try:
                        with open(neg_ckpt, 'w', encoding='utf-8') as _nf:
                            json.dump({
                                'problem': problem['name'], 'mode': result.mode, 'N': result.N,
                                'K': K, 'passed': result.passed, 'winner': result.winner_agent_id,
                                'total_rounds': result.total_rounds, 'total_llm_calls': result.total_llm_calls,
                                'execution_time': result.execution_time
                            }, _nf, ensure_ascii=False, indent=2)
                        print(f"  [协商检查点] {problem['name']} K={K} 已保存")
                    except Exception as _e:
                        print(f"    [警告] 写协商检查点失败: {_e}")
            runner.print_statistics()
        finally:
            runner.disconnect_npu()

    else:
        # 跑全部4种模式对比
        runner.run_all_experiments()


if __name__ == "__main__":
    import sys as _sys, traceback as _tb
    _CRASH_LOG = './experiment_results_ascendc/crash.log'
    def _excepthook(et, ev, tb):
        try:
            with open(_CRASH_LOG, 'a', encoding='utf-8') as _f:
                _f.write('\n=== CRASH ' + ''.join(_tb.format_exception_only(et, ev)).strip() + ' ===\n')
                _f.write(''.join(_tb.format_exception(et, ev, tb)))
        except Exception:
            pass
        _sys.__excepthook__(et, ev, tb)
    _sys.excepthook = _excepthook
    try:
        main()
    except Exception as _e:
        try:
            with open(_CRASH_LOG, 'a', encoding='utf-8') as _f:
                _f.write('\n=== MAIN EXCEPTION ===\n' + ''.join(_tb.format_exception(type(_e), _e, _e.__traceback__)))
        except Exception:
            pass
        raise
