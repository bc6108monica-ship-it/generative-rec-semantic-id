#!/usr/bin/env python3
"""
检查多模态特征文件中是否有nan或异常值
"""
import json
import numpy as np
from pathlib import Path
import sys

def check_feature_file(file_path, max_lines=10):
    """检查一个特征文件中的nan值"""
    print(f"检查文件: {file_path}")
    nan_count = 0
    inf_count = 0
    total_values = 0
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

                        # 检查nan和inf
                        nan_mask = np.isnan(emb_array)
                        inf_mask = np.isinf(emb_array)

                        nan_in_line = np.sum(nan_mask)
                        inf_in_line = np.sum(inf_mask)

                        if nan_in_line > 0 or inf_in_line > 0:
                            print(f"  行 {line_num}: 发现 {nan_in_line}个NaN, {inf_in_line}个Inf")
                            print(f"    item_id: {data.get('anonymous_cid', 'N/A')}")
                            # 打印有问题的索引
                            if nan_in_line > 0:
                                nan_indices = np.where(nan_mask)[0]
                                print(f"    NaN索引: {nan_indices[:5]}{'...' if len(nan_indices) > 5 else ''}")
                            if inf_in_line > 0:
                                inf_indices = np.where(inf_mask)[0]
                                print(f"    Inf索引: {inf_indices[:5]}{'...' if len(inf_indices) > 5 else ''}")

                        nan_count += nan_in_line
                        inf_count += inf_in_line
                        total_values += len(emb_array)

                        # 检查数值范围
                        if len(emb_array) > 0:
                            abs_max = np.max(np.abs(emb_array))
                            if abs_max > 1000:  # 非常大的值
                                print(f"  行 {line_num}: 绝对值过大, max|value| = {abs_max:.4f}")

                    line_count += 1

            except json.JSONDecodeError as e:
                print(f"  行 {line_num}: JSON解析错误: {e}")
            except Exception as e:
                print(f"  行 {line_num}: 错误: {e}")

    return nan_count, inf_count, total_values, line_count

def main():
    data_dir = Path("data/TencentGR_1k/creative_emb")
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        sys.exit(1)

    # 检查所有特征目录
    feat_dirs = list(data_dir.glob("emb_*"))
    print(f"找到 {len(feat_dirs)} 个特征目录")

    for feat_dir in feat_dirs:
        if feat_dir.is_dir():
            print(f"\n{'='*60}")
            print(f"检查目录: {feat_dir.name}")
            print(f"{'='*60}")

            part_files = list(feat_dir.glob("part-*"))
            if not part_files:
                print("  没有找到part文件")
                continue

            # 检查前几个文件
            for i, part_file in enumerate(part_files[:2]):  # 只检查前2个文件
                print(f"\n文件 {i+1}/{min(2, len(part_files))}: {part_file.name}")
                nan_count, inf_count, total_values, lines = check_feature_file(part_file, max_lines=20)

                if lines > 0:
                    print(f"  统计: 检查了 {lines} 行, {total_values} 个值")
                    print(f"        NaN: {nan_count} ({nan_count/max(1,total_values)*100:.2f}%)")
                    print(f"        Inf: {inf_count} ({inf_count/max(1,total_values)*100:.2f}%)")

                # 如果发现大量nan，提前停止
                if nan_count > 0 or inf_count > 0:
                    print(f"  ⚠️  在 {part_file.name} 中发现异常值!")

    print(f"\n{'='*60}")
    print("检查完成")

if __name__ == "__main__":
    main()