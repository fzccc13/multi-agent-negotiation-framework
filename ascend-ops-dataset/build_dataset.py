"""
将三个数据来源统一转换为AscendC算子测试数据集
输出: ascend-ops-dataset/final/test.json

实际数据格式说明:
- MultiKernelBench: 每个level目录下是扁平文件 N_operator_name.json + N_operator_name.py
  - .json: JSONL格式，每行一个测试用例(input specs)
  - .py: PyTorch参考实现(Model类 + get_input_groups + get_init_inputs)
- ops-math: 每个算子是目录 ops-math/math/abs/，内含 op_kernel/, op_host/ 等
  - kernel文件可能是 {op_name}.cpp 或 {op_name}_apt.cpp
- 桌面5个算子: 每个算子是目录，内含 op_kernel/xxx.cpp, op_host/xxx.cpp, scripts/
"""
import os
import json
import re
import glob

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCES_DIR = os.path.join(DATASET_DIR, 'sources')
FINAL_DIR = os.path.join(DATASET_DIR, 'final')


def read_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    return ""


# ============================================================
# 来源1: 桌面5个算子示例
# ============================================================
def parse_desktop_ops():
    ops = []
    base = os.path.join(SOURCES_DIR, 'desktop-5-ops')
    if not os.path.exists(base):
        print(f"  [SKIP] desktop-5-ops not found")
        return ops

    operator_map = {
        'AddCustom': {'name': 'Add', 'op_dir': 'Add', 'category': 'math', 'difficulty': 0,
                      'desc': 'Element-wise addition: y = x1 + x2'},
        'ReluCustom': {'name': 'ReLU', 'op_dir': 'Relu', 'category': 'activation', 'difficulty': 0,
                       'desc': 'Element-wise ReLU: y = max(0, x)'},
        'SigmoidCustom': {'name': 'Sigmoid', 'op_dir': 'Sigmoid', 'category': 'activation', 'difficulty': 1,
                          'desc': 'Element-wise Sigmoid: y = 1/(1+e^(-x))'},
        'FmodCustom': {'name': 'Fmod', 'op_dir': 'Fmod', 'category': 'math', 'difficulty': 1,
                       'desc': 'Floating-point modulo: y = x1 - floor(x1/x2)*x2'},
        'AsinhCustom': {'name': 'Asinh', 'op_dir': 'Asinh', 'category': 'math', 'difficulty': 1,
                        'desc': 'Inverse hyperbolic sine: y = asinh(x)'},
    }

    for custom_dir, info in operator_map.items():
        framework_dir = os.path.join(base, custom_dir, 'FrameworkLaunch')
        op_dir = os.path.join(framework_dir, info['op_dir'])
        kernel_file = os.path.join(op_dir, 'op_kernel', f'{info["op_dir"].lower()}.cpp')
        host_file = os.path.join(op_dir, 'op_host', f'{info["op_dir"].lower()}.cpp')
        tiling_file = os.path.join(op_dir, 'op_host', f'{info["op_dir"].lower()}_tiling.h')
        scripts_dir = os.path.join(framework_dir, 'AclNNInvocation', 'scripts')
        gen_data_file = os.path.join(scripts_dir, 'gen_data.py')
        verify_file = os.path.join(scripts_dir, 'verify_result.py')

        if not os.path.exists(kernel_file):
            print(f"  [SKIP] {info['name']}: kernel not found")
            continue

        prompt = f"""Please implement the {info['name']} operator kernel using AscendC.

Description: {info['desc']}

Requirements:
1. Use AscendC programming model
2. Implement Kernel class with Init() and Process() methods
3. Implement extern "C" __global__ __aicore__ void entry function
4. Support float16/float/int32 data types
5. Target Ascend 310B4 platform

Entry function signature:
extern "C" __global__ __aicore__ void {info['op_dir'].lower()}(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
"""

        op_data = {
            'name': f'ascend_{info["name"].lower()}',
            'category': info['category'],
            'difficulty': info['difficulty'],
            'language': 'ascendc',
            'description': info['desc'],
            'prompt': prompt,
            'reference_kernel': read_file(kernel_file),
            'reference_host': read_file(host_file),
            'tiling_header': read_file(tiling_file),
            'test_gen_script': read_file(gen_data_file),
            'test_verify_script': read_file(verify_file),
            'original_source': 'desktop-5-ops',
            'has_test_scripts': True
        }
        ops.append(op_data)
        print(f"  [OK] {info['name']}: loaded")

    return ops


# ============================================================
# 来源2: MultiKernelBench (NPU算子)
# ============================================================
def parse_multikernelbench():
    ops = []
    base = os.path.join(SOURCES_DIR, 'MultiKernelBench', 'reference')
    if not os.path.exists(base):
        print(f"  [SKIP] MultiKernelBench not found")
        return ops

    for level in range(5):
        level_dir = os.path.join(base, f'npukernelbench_level{level}')
        if not os.path.exists(level_dir):
            continue

        level_count = 0
        # 找所有 .py 文件，对应的 .json 是测试用例
        py_files = sorted(glob.glob(os.path.join(level_dir, '*.py')))

        for py_file in py_files:
            basename = os.path.basename(py_file)  # e.g. "10_relu.py"
            stem = basename[:-3]  # "10_relu"
            json_file = os.path.join(level_dir, f'{stem}.json')

            # 提取算子名称: "10_relu" -> "relu", "1_logical_and" -> "logical_and"
            parts = stem.split('_', 1)
            if len(parts) < 2:
                op_name = stem
            else:
                op_name = parts[1]

            ref_code = read_file(py_file)

            # 读取测试用例(JSONL格式)
            test_cases = []
            json_content = read_file(json_file)
            if json_content:
                for line in json_content.strip().split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            test_cases.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

            # 从PyTorch参考代码中提取描述
            desc_match = re.search(r'"""(.*?)"""', ref_code, re.DOTALL)
            desc = desc_match.group(1).strip().split('\n')[0] if desc_match else f'Operator: {op_name}'

            # 构建prompt
            # 提取Model类的forward方法签名
            forward_match = re.search(r'def forward\((.*?)\)', ref_code)
            forward_sig = forward_match.group(1) if forward_match else ''

            prompt = f"""Please implement the {op_name} operator using AscendC for NPU.

Description: {desc}

Reference PyTorch implementation:
{ref_code[:1500]}

Test cases (input specifications):
{json.dumps(test_cases[:3], indent=2)}  # showing first 3 of {len(test_cases)} cases

Requirements:
1. Use AscendC programming model
2. Implement Kernel class with Init() and Process() methods
3. Implement extern "C" __global__ __aicore__ void entry function
4. Target Ascend NPU platform
"""

            op_data = {
                'name': f'mkb_{op_name}',
                'category': 'npu_operator',
                'difficulty': level,
                'language': 'ascendc',
                'description': desc,
                'prompt': prompt,
                'reference_kernel': ref_code,  # PyTorch reference
                'reference_host': '',
                'tiling_header': '',
                'test_cases': test_cases,
                'test_gen_script': '',
                'test_verify_script': '',
                'original_source': f'MultiKernelBench_level{level}',
                'has_test_scripts': len(test_cases) > 0
            }
            ops.append(op_data)
            level_count += 1

        print(f"  [OK] Level {level}: {level_count} operators")

    return ops


# ============================================================
# 来源3: 华为CANN ops-math仓库
# ============================================================
def parse_cann_ops():
    ops = []
    base = os.path.join(SOURCES_DIR, 'ops-math')
    if not os.path.exists(base):
        print(f"  [SKIP] ops-math not found")
        return ops

    for category in ['math', 'conversion', 'random', 'experimental']:
        cat_dir = os.path.join(base, category)
        if not os.path.exists(cat_dir):
            continue

        cat_count = 0
        for op_name in sorted(os.listdir(cat_dir)):
            op_dir = os.path.join(cat_dir, op_name)
            if not os.path.isdir(op_dir):
                continue

            kernel_dir = os.path.join(op_dir, 'op_kernel')
            host_dir = os.path.join(op_dir, 'op_host')

            if not os.path.exists(kernel_dir):
                continue

            # 尝试多种文件名模式
            kernel_code = ""
            host_code = ""
            tiling_code = ""

            # Kernel: try {op_name}.cpp, {op_name}_apt.cpp, any .cpp in op_kernel
            kernel_files = glob.glob(os.path.join(kernel_dir, '*.cpp'))
            if kernel_files:
                kernel_code = read_file(kernel_files[0])

            # Host: try {op_name}.cpp, any .cpp in op_host
            host_files = glob.glob(os.path.join(host_dir, '*.cpp'))
            if host_files:
                # Prefer the one matching op_name
                for hf in host_files:
                    if op_name.lower() in os.path.basename(hf).lower():
                        host_code = read_file(hf)
                        break
                if not host_code and host_files:
                    host_code = read_file(host_files[0])

            # Tiling: try {op_name}_tiling.h, any .h in op_host
            tiling_files = glob.glob(os.path.join(host_dir, '*_tiling.h'))
            if tiling_files:
                tiling_code = read_file(tiling_files[0])

            if not kernel_code or len(kernel_code) < 50:
                continue

            # 读取README获取描述
            readme = read_file(os.path.join(op_dir, 'README.md'))
            desc = op_name
            if readme:
                # 提取第一段非标题文本
                for line in readme.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('!'):
                        desc = line
                        break

            difficulty = 0 if category == 'math' else (1 if category == 'conversion' else 1)

            prompt = f"""Please implement the {op_name} operator kernel using AscendC.

Category: {category}
Description: {desc}

Requirements:
1. Use AscendC programming model
2. Implement Kernel class with Init() and Process() methods
3. Target Ascend 310B4 platform

Host-side registration (for reference):
{host_code[:500] if host_code else 'N/A'}
"""

            # 检查是否有测试
            test_dir = os.path.join(op_dir, 'tests')
            has_tests = os.path.exists(test_dir) and bool(glob.glob(os.path.join(test_dir, '*.py')))

            op_data = {
                'name': f'cann_{op_name.lower()}',
                'category': category,
                'difficulty': difficulty,
                'language': 'ascendc',
                'description': desc,
                'prompt': prompt,
                'reference_kernel': kernel_code,
                'reference_host': host_code,
                'tiling_header': tiling_code,
                'test_gen_script': '',
                'test_verify_script': '',
                'original_source': f'cann-ops-math/{category}',
                'has_test_scripts': has_tests
            }
            ops.append(op_data)
            cat_count += 1

        print(f"  [OK] {category}: {cat_count} operators")

    return ops


# ============================================================
# 来源3b: 华为CANN ops-nn仓库
# ============================================================
def parse_cann_nn_ops():
    ops = []
    base = os.path.join(SOURCES_DIR, 'ops-nn')
    if not os.path.exists(base):
        print(f"  [SKIP] ops-nn not found")
        return ops

    categories = ['activation', 'conv', 'loss', 'matmul', 'norm', 'pooling',
                  'optim', 'quant', 'rnn', 'index', 'control', 'foreach', 'hash']
    cat_count = 0
    for category in categories:
        cat_dir = os.path.join(base, category)
        if not os.path.exists(cat_dir):
            continue

        for op_name in sorted(os.listdir(cat_dir)):
            op_dir = os.path.join(cat_dir, op_name)
            if not os.path.isdir(op_dir) or op_name.startswith('.'):
                continue

            kernel_dir = os.path.join(op_dir, 'op_kernel')
            if not os.path.exists(kernel_dir):
                continue

            kernel_files = glob.glob(os.path.join(kernel_dir, '*.cpp'))
            if not kernel_files:
                continue

            kernel_code = read_file(kernel_files[0])

            host_dir = os.path.join(op_dir, 'op_host')
            host_files = glob.glob(os.path.join(host_dir, '*.cpp'))
            host_code = ""
            if host_files:
                host_code = read_file(host_files[0])

            tiling_files = glob.glob(os.path.join(host_dir, '*_tiling.h'))
            tiling_code = read_file(tiling_files[0]) if tiling_files else ""

            if len(kernel_code) < 50:
                continue

            readme = read_file(os.path.join(op_dir, 'README.md'))
            desc = op_name
            if readme:
                for line in readme.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('!'):
                        desc = line
                        break

            prompt = f"""Please implement the {op_name} operator kernel using AscendC.

Category: neural_network/{category}
Description: {desc}

Requirements:
1. Use AscendC programming model
2. Implement Kernel class with Init() and Process() methods
3. Target Ascend 310B4 platform
"""

            op_data = {
                'name': f'cannnn_{op_name.lower()}',
                'category': f'nn_{category}',
                'difficulty': 2,
                'language': 'ascendc',
                'description': desc,
                'prompt': prompt,
                'reference_kernel': kernel_code,
                'reference_host': host_code,
                'tiling_header': tiling_code,
                'test_gen_script': '',
                'test_verify_script': '',
                'original_source': f'cann-ops-nn/{category}',
                'has_test_scripts': False
            }
            ops.append(op_data)
            cat_count += 1

    print(f"  [OK] ops-nn: {cat_count} operators")
    return ops


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("Build AscendC Operator Test Dataset")
    print("=" * 60)

    all_ops = []

    print("\n[1/3] Desktop 5 operators...")
    all_ops.extend(parse_desktop_ops())

    # [MultiKernelBench 已按老师要求从交付数据集中剔除]
    # 该来源非老师指定，且其 reference_kernel 为 PyTorch 实现而非 AscendC。
    # 如需重新纳入，取消下一行注释即可。
    # all_ops.extend(parse_multikernelbench())

    print("\n[2/3] CANN ops-math...")
    all_ops.extend(parse_cann_ops())

    print("\n[3/3] CANN ops-nn...")
    all_ops.extend(parse_cann_nn_ops())

    print(f"\nTotal: {len(all_ops)} operators")

    from collections import Counter
    source_counts = Counter(op['original_source'].split('/')[0].split('_level')[0] for op in all_ops)
    print("\nBy source:")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count}")

    diff_counts = Counter(op['difficulty'] for op in all_ops)
    print("\nBy difficulty:")
    for d in sorted(diff_counts.keys()):
        print(f"  Level {d}: {diff_counts[d]}")

    has_test = sum(1 for op in all_ops if op.get('has_test_scripts'))
    print(f"\nWith test scripts: {has_test}/{len(all_ops)}")

    # 去重
    seen = set()
    unique_ops = []
    for op in all_ops:
        if op['name'] not in seen:
            seen.add(op['name'])
            unique_ops.append(op)
    if len(unique_ops) < len(all_ops):
        print(f"After dedup: {len(unique_ops)} operators")

    # 保存
    os.makedirs(FINAL_DIR, exist_ok=True)
    output_file = os.path.join(FINAL_DIR, 'test.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_ops, f, ensure_ascii=False, indent=2)

    print(f"\nDataset saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    # 简化版清单
    simple_ops = [{'name': op['name'], 'category': op['category'],
                   'difficulty': op['difficulty'], 'description': op['description'][:100],
                   'source': op['original_source'], 'has_test': op.get('has_test_scripts', False)}
                  for op in unique_ops]
    simple_file = os.path.join(FINAL_DIR, 'operator_list.json')
    with open(simple_file, 'w', encoding='utf-8') as f:
        json.dump(simple_ops, f, ensure_ascii=False, indent=2)
    print(f"Operator list saved to: {simple_file}")


if __name__ == '__main__':
    main()
