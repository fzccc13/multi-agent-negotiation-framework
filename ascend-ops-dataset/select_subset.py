"""
从完整数据集中筛选一个子集用于实验
策略：保留有完整参考代码的算子，按难度均衡选择
"""
import json
import os

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATASET_DIR, 'final', 'test.json')
OUTPUT_FILE = os.path.join(DATASET_DIR, 'final', 'test_selected.json')

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"原始数据集: {len(data)} 个算子")

    has_kernel = [op for op in data if len(op.get('reference_kernel', '').strip()) > 50]
    print(f"有核函数代码: {len(has_kernel)}")

    has_test = [op for op in has_kernel if op.get('test_gen_script', '').strip()]
    no_test = [op for op in has_kernel if not op.get('test_gen_script', '').strip()]
    print(f"  其中带测试脚本: {len(has_test)}")
    print(f"  无测试脚本: {len(no_test)}")

    by_level = {}
    for op in has_kernel:
        level = op['difficulty']
        by_level.setdefault(level, []).append(op)

    target = {0: 999, 1: 999, 2: 30, 3: 10, 4: 10}
    selected = []
    for level in sorted(by_level.keys()):
        pool = by_level[level]
        n = min(target.get(level, 20), len(pool))
        with_test = [op for op in pool if op.get('test_gen_script', '').strip()]
        without_test = [op for op in pool if not op.get('test_gen_script', '').strip()]
        selected.extend(with_test[:n])
        remaining = n - len(with_test)
        if remaining > 0:
            selected.extend(without_test[:remaining])
        print(f"Level {level}: {len(pool)} available, selected {min(n, len(pool))}")

    print(f"\n最终选择: {len(selected)} 个算子")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
