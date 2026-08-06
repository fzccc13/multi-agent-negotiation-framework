"""
多Agent协商框架 - AscendC算子实验配置
适配310B4真机环境，替代Rust HumanEval数据集
"""

# ==================== 实验配置 ====================
EXPERIMENT_CONFIG = {
    'N': 5,  # Agent总数（5个不同模型）
    'K_values': ['baseline', 'N', 1, 2],  # 四种实验模式
    'alpha': 0.1,  # 权重更新学习率
    'gamma': 0.3,  # 终局权重吸收率
    'W_init': None,  # 初始权重（自动计算为 1/N）
}

# ==================== 多Agent模型配置 ====================
# 阿里云百炼平台 - 兼容OpenAI接口
AGENT_MODELS = [
    {
        'agent_id': 0,
        'name': 'GLM-5.2',
        'model': 'glm-5.2',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': 'YOUR_DASHSCOPE_API_KEY',
        'temperature': 0.7,
        'max_tokens': 4096,  # AscendC代码较长，增加token上限
        'enable_thinking': True,  # True=开启思考模式; qwen-flash可设False
    },
    {
        'agent_id': 1,
        'name': 'Qwen3.7-Flash',
        'model': 'qwen3.7-flash-2026-07-15',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': 'YOUR_DASHSCOPE_API_KEY',
        'temperature': 0.7,
        'max_tokens': 4096,
        'enable_thinking': False,  # True=开启思考模式; qwen-flash可设False
    },
    {
        'agent_id': 2,
        'name': 'Qwen3.7-Max-0517',
        'model': 'qwen3.7-max-2026-05-17',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': 'YOUR_DASHSCOPE_API_KEY',
        'temperature': 0.7,
        'max_tokens': 4096,
        'enable_thinking': True,  # True=开启思考模式; qwen-flash可设False
    },
    {
        'agent_id': 3,
        'name': 'Qwen3.7-Max-0608',
        'model': 'qwen3.7-max-2026-06-08',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': 'YOUR_DASHSCOPE_API_KEY',
        'temperature': 0.7,
        'max_tokens': 4000,
        'enable_thinking': True,  # True=开启思考模式; qwen-flash可设False
    },
    {
        'agent_id': 4,
        'name': 'Kimi-K2.7-Code',
        'model': 'kimi-k2.7-code',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': 'YOUR_DASHSCOPE_API_KEY',
        'temperature': 0.7,
        'max_tokens': 4096,
        'enable_thinking': True,  # True=开启思考模式; qwen-flash可设False
    },
]

# ==================== AscendC数据集配置 ====================
DATASET_CONFIG = {
    'dataset_path': './ascend-ops-dataset/final/test.json',
    'language': 'ascendc',
    'num_samples': 5,  # 先跑5题baseline（desktop-5-ops）
    'filter_source': 'desktop-5-ops',  # 只筛选desktop-5-ops来源的算子
}

# ==================== 310B4真机SSH配置 ====================
NPU_CONFIG = {
    'host': 'YOUR_NPU_HOST',
    'port': 22,
    'username': 'YOUR_NPU_USERNAME',
    'password': 'YOUR_NPU_SSH_PASSWORD',
    'remote_work_dir': '/home/YOUR_NPU_USERNAME/negotiation-eval',
    'remote_opp_packages': '/home/YOUR_NPU_USERNAME/opp_packages',
    'soc_version': 'Ascend310B4',
    # 远程环境初始化命令（每次SSH连接后执行）
    'env_setup_commands': [
        'source /usr/local/Ascend/ascend-toolkit/set_env.sh',
        'source /home/YOUR_NPU_USERNAME/opp_packages/vendors/customize/bin/set_env.bash',
    ],
}

# ==================== 执行配置 ====================
EXECUTION_CONFIG = {
    'ssh_timeout': 300,  # SSH单次命令超时（秒），build/run可能较长
    'build_timeout': 180,  # 编译超时（秒）
    'run_timeout': 60,  # 运行超时（秒）
    'output_dir': './experiment_results_ascendc',  # 结果输出目录（C盘workspace，避开D盘沙箱写拦截）
    'max_retries': 2,  # SSH连接最大重试次数
    'max_fix_rounds': 10,  # 编译反馈循环上限：agent根据编译错误最多修正次数
}

# ==================== 环境变量注入（避免真实密钥落盘） ====================
# 若设置了以下环境变量，自动覆盖上面的脱敏占位（方式 B，推荐）。
import os
_ENV_KEY = os.environ.get('DASHSCOPE_API_KEY')
if _ENV_KEY:
    for _m in AGENT_MODELS:
        _m['api_key'] = _ENV_KEY
if os.environ.get('NPU_HOST'):
    NPU_CONFIG['host'] = os.environ['NPU_HOST']
if os.environ.get('NPU_USERNAME'):
    NPU_CONFIG['username'] = os.environ['NPU_USERNAME']
if os.environ.get('NPU_SSH_PASSWORD'):
    NPU_CONFIG['password'] = os.environ['NPU_SSH_PASSWORD']
if os.environ.get('NPU_PORT'):
    try:
        NPU_CONFIG['port'] = int(os.environ['NPU_PORT'])
    except ValueError:
        pass
if os.environ.get('NPU_REMOTE_WORK_DIR'):
    NPU_CONFIG['remote_work_dir'] = os.environ['NPU_REMOTE_WORK_DIR']
if os.environ.get('NPU_OPP_PACKAGES'):
    NPU_CONFIG['remote_opp_packages'] = os.environ['NPU_OPP_PACKAGES']

NPU_CONFIG['env_setup_commands'] = [
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh',
    f"source {NPU_CONFIG['remote_opp_packages']}/vendors/customize/bin/set_env.bash",
]

