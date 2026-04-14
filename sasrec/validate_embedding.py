#!/usr/bin/env python3
"""
测试修复后的load_mm_emb函数
"""
import sys
sys.path.append('.')
from sasrec.dataset import load_mm_emb
from pathlib import Path
import json
import numpy as np

print("测试数据加载...")

# 尝试本地数据路径
data_dir = Path("data/TencentGR_1k")
if data_dir.exists():
    print(f"✅ 找到本地数据目录: {data_dir}")
    mm_path = data_dir / "creative_emb"
else:
    # 尝试用户提到的路径
    data_dir = Path("/root/autodl-tmp/data")
    if data_dir.exists():
        print(f"✅ 找到autodl数据目录: {data_dir}")
        mm_path = data_dir / "creative_emb"
    else:
        print("❌ 未找到数据目录")
        sys.exit(1)

if not mm_path.exists():
    print(f"❌ 多模态特征目录不存在: {mm_path}")
    sys.exit(1)

print(f"多模态特征目录: {mm_path}")

# 列出所有特征目录
print("\n目录结构:")
for feat_dir in mm_path.glob("emb_*"):
    if feat_dir.is_dir():
        part_files = list(feat_dir.glob("part-*"))
        json_files = list(feat_dir.glob("*.json"))
        print(f"  {feat_dir.name}: {len(part_files)}个part文件, {len(json_files)}个json文件")
    else:
        print(f"  {feat_dir.name} (文件)")

# 测试只加载81和82特征
feat_ids = ['81', '82']
print(f"\n加载特征: {feat_ids}")
mm_emb_dict = load_mm_emb(mm_path, feat_ids)

print(f"\n加载结果:")
for feat_id in feat_ids:
    if feat_id in mm_emb_dict:
        emb_dict = mm_emb_dict[feat_id]
        print(f"  特征 #{feat_id}: {len(emb_dict)} items")
        if emb_dict:
            # 显示第一个item的embedding信息
            sample_key = list(emb_dict.keys())[0]
            sample_emb = emb_dict[sample_key]
            print(f"    样本: item_id={sample_key}, shape={sample_emb.shape}, dtype={sample_emb.dtype}")
    else:
        print(f"  特征 #{feat_id}: 未加载")

# 检查一个part文件的内容格式
print(f"\n检查part文件格式...")
feat_81_dir = mm_path / "emb_81_32"
if feat_81_dir.exists():
    part_files = list(feat_81_dir.glob("part-*"))
    if part_files:
        test_file = part_files[0]
        print(f"检查文件: {test_file}")
        try:
            with open(test_file, 'r') as f:
                first_line = f.readline().strip()
                print(f"第一行内容 (前200字符): {first_line[:200]}...")
                data = json.loads(first_line)
                print(f"JSON解析成功:")
                print(f"  anonymous_cid: {data.get('anonymous_cid')}")
                print(f"  emb类型: {type(data.get('emb'))}")
                if 'emb' in data:
                    emb = data['emb']
                    if isinstance(emb, list):
                        print(f"  emb长度: {len(emb)}")
                        print(f"  前5个值: {emb[:5]}")
        except Exception as e:
            print(f"读取文件错误: {e}")
            import traceback
            traceback.print_exc()