#!/usr/bin/env python3
"""
检查多模态特征的统计信息：均值、方差、范围
"""
import json
import numpy as np
from pathlib import Path
import sys

def compute_feature_stats(file_path, max_lines=100):
    """计算特征文件的统计信息"""
    all_values = []
    line_count = 0

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num >= max_lines:
                break

            try:
                data = json.loads(line.strip())
                if 'emb' in data:
                    emb = data['emb']
                    if isinstance(emb, list):
                        emb_array = np.array(emb, dtype=np.float32)
                        all_values.extend(emb_array)
                        line_count += 1

            except Exception:
                continue

    if not all_values:
        return None

    all_values = np.array(all_values)
    stats = {
        'mean': np.mean(all_values),
        'std': np.std(all_values),
        'min': np.min(all_values),
        'max': np.max(all_values),
        'abs_max': np.max(np.abs(all_values)),
        'q1': np.percentile(all_values, 25),
        'q2': np.percentile(all_values, 50),  # 中位数
        'q3': np.percentile(all_values, 75),
        'count': len(all_values),
        'lines': line_count
    }
    return stats

def main():
    data_dir = Path("data/TencentGR_1k/creative_emb")
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        sys.exit(1)

    # 检查所有特征目录
    feat_dirs = list(data_dir.glob("emb_*"))

    for feat_dir in feat_dirs:
        if feat_dir.is_dir():
            print(f"\n{'='*60}")
            print(f"统计目录: {feat_dir.name}")
            print(f"{'='*60}")

            part_files = list(feat_dir.glob("part-*"))
            if not part_files:
                print("  没有找到part文件")
                continue

            # 取第一个文件进行统计
            part_file = part_files[0]
            print(f"分析文件: {part_file.name} (前100行)")

            stats = compute_feature_stats(part_file, max_lines=100)
            if stats:
                print(f"  样本数量: {stats['lines']} 行, {stats['count']} 个值")
                print(f"  均值: {stats['mean']:.6f}")
                print(f"  标准差: {stats['std']:.6f}")
                print(f"  最小值: {stats['min']:.6f}")
                print(f"  最大值: {stats['max']:.6f}")
                print(f"  绝对最大值: {stats['abs_max']:.6f}")
                print(f"  四分位数: Q1={stats['q1']:.6f}, Q2={stats['q2']:.6f}, Q3={stats['q3']:.6f}")

                # 检查是否需要进行标准化
                if stats['abs_max'] > 10:
                    print(f"  ⚠️  绝对值较大 (>10)，可能需要标准化")
                if stats['std'] < 0.001:
                    print(f"  ⚠️  标准差很小 (<0.001)，特征可能已标准化")
                if abs(stats['mean']) > 1:
                    print(f"  ⚠️  均值绝对值较大 (>1)")

                # 检查极端值
                extreme_threshold = 100
                if stats['abs_max'] > extreme_threshold:
                    print(f"  ⚠️  发现极端值 (> {extreme_threshold})")

    print(f"\n{'='*60}")
    print("统计完成")

if __name__ == "__main__":
    main()