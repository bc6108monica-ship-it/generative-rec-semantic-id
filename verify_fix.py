#!/usr/bin/env python3
"""
验证特征标准化修复
"""
import sys
import numpy as np
from pathlib import Path

# 添加sasrec目录到路径
sys.path.insert(0, 'sasrec')
from dataset import load_mm_emb

def main():
    data_dir = Path("data/TencentGR_1k")
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        sys.exit(1)

    mm_path = data_dir / "creative_emb"
    if not mm_path.exists():
        print(f"多模态特征目录不存在: {mm_path}")
        sys.exit(1)

    # 测试所有特征
    feat_ids = ['81', '82', '83', '84', '85', '86']
    print("加载特征并标准化...")
    mm_emb_dict = load_mm_emb(mm_path, feat_ids, normalize=True)

    print("\n标准化后的特征统计:")
    for feat_id in feat_ids:
        if feat_id in mm_emb_dict:
            emb_dict = mm_emb_dict[feat_id]
            if not emb_dict:
                print(f"  特征 #{feat_id}: 空字典")
                continue

            # 收集所有值
            all_values = []
            for emb in emb_dict.values():
                if isinstance(emb, np.ndarray):
                    all_values.append(emb)
                elif isinstance(emb, list):
                    all_values.append(np.array(emb, dtype=np.float32))

            if all_values:
                all_arrays = np.vstack(all_values)
                mean = np.mean(all_arrays, axis=0)
                std = np.std(all_arrays, axis=0)

                print(f"  特征 #{feat_id}:")
                print(f"    样本数: {len(emb_dict)}")
                print(f"    形状: {all_arrays.shape[1] if all_arrays.shape else 'N/A'}")
                print(f"    全局均值: {mean.mean():.6f} (范围: {mean.min():.6f} 到 {mean.max():.6f})")
                print(f"    全局标准差: {std.mean():.6f} (范围: {std.min():.6f} 到 {std.max():.6f})")
                print(f"    值范围: {all_arrays.min():.6f} 到 {all_arrays.max():.6f}")

                # 检查nan
                nan_count = np.sum(np.isnan(all_arrays))
                inf_count = np.sum(np.isinf(all_arrays))
                if nan_count > 0 or inf_count > 0:
                    print(f"    ⚠️  发现 {nan_count}个NaN, {inf_count}个Inf")
                else:
                    print(f"    ✓ 无NaN/Inf值")

                # 检查是否已标准化（均值接近0，标准差接近1）
                if abs(mean.mean()) < 0.1 and 0.9 < std.mean() < 1.1:
                    print(f"    ✓ 已标准化 (均值~0, 标准差~1)")
                else:
                    print(f"    ⚠️  标准化不完全 (均值={mean.mean():.3f}, 标准差={std.mean():.3f})")
            else:
                print(f"  特征 #{feat_id}: 无有效值")
        else:
            print(f"  特征 #{feat_id}: 未加载")

    print("\n验证完成")

if __name__ == "__main__":
    main()