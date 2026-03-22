#!/bin/bash
# meeting_demo.sh — LLM4Rec 三阶段流水线演示脚本
# RTX 5060 显存友好：epochs=1，batch_size=16
# 用法：./meeting_demo.sh（在 ~/llm4rec/ 目录下执行）

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEMO_CACHE="${REPO_ROOT}/.demo_cache"
# 创建演示所需的临时目录
mkdir -p "${DEMO_CACHE}/logs" \
         "${DEMO_CACHE}/tf_events" \
         "${DEMO_CACHE}/ckpt" \
         "${DEMO_CACHE}/emb" \
         "${DEMO_CACHE}/gr_ckpt"

# ─────────────────────────────────────────────────────────────────────────────
# 第一阶段 — SASRec：多模态序列表示学习
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================="
echo " [开始] 第一阶段 — SASRec：多模态表示学习"
echo "========================================================="
cd "${REPO_ROOT}/sasrec"

echo "[信息] 验证 SASRec 模块导入..."
python - <<'PYEOF'
import sys
sys.path.insert(0, '.')
import model
import dataset
from model import BaselineModel
from dataset import MyDataset
print("[OK]   model.py   -> BaselineModel（Transformer + Flash Attention）")
print("[OK]   dataset.py -> MyDataset    （JSONL 用户行为序列加载器）")
PYEOF

if [[ -n "${TRAIN_DATA_PATH}" && -d "${TRAIN_DATA_PATH}" ]]; then
    echo "[信息] 检测到 TRAIN_DATA_PATH，启动 1 轮 SASRec 训练..."
    export TRAIN_LOG_PATH="${DEMO_CACHE}/logs"
    export TRAIN_TF_EVENTS_PATH="${DEMO_CACHE}/tf_events"
    export TRAIN_CKPT_PATH="${DEMO_CACHE}/ckpt"
    export USER_CACHE_PATH="${DEMO_CACHE}"
    python -u main.py --num_epochs 1 --batch_size 16
else
    echo "[跳过] TRAIN_DATA_PATH 未设置或目录不存在。"
    echo "       执行 export TRAIN_DATA_PATH=/数据路径 后可启用完整训练。"
fi

echo "[完成] 第一阶段结束。"


# ─────────────────────────────────────────────────────────────────────────────
# 第二阶段 — RQ-VAE：层次化语义 ID 构建
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================="
echo " [开始] 第二阶段 — RQ-VAE：语义 ID 构建"
echo "========================================================="
cd "${REPO_ROOT}/rqvae/train"

echo "[信息] 验证 RQ-VAE 模块导入..."
python - <<'PYEOF'
import sys
sys.path.insert(0, '.')
import rqvae_model
import datasets
import trainer
from rqvae_model import RQVAE, VectorQuantizer
from trainer import Trainer
print("[OK]   rqvae_model.py -> RQVAE, VectorQuantizer（3 级残差量化码本）")
print("[OK]   datasets.py    -> CustomNpzFile         （NPZ 向量文件加载器）")
print("[OK]   trainer.py     -> Trainer               （含碰撞率监控的训练循环）")
PYEOF

# 注意：rqvae_train.py 的 __main__ 块中硬编码了 data_path="/emb/emb"（竞赛平台遗留）
# --data_path 命令行参数会被覆盖，但 epochs/batch_size 仍通过 argparse 正常生效
RQVAE_DATA="/emb/emb"
if [[ -d "${RQVAE_DATA}" ]] && ls "${RQVAE_DATA}"/*.npz > /dev/null 2>&1; then
    echo "[信息] 在 ${RQVAE_DATA} 找到 NPZ 向量文件，启动 1 轮 RQ-VAE 训练..."
    export USER_CACHE_PATH="${DEMO_CACHE}"
    python -u rqvae_train.py \
        --epochs 1 \
        --batch_size 16 \
        --eval_step 1 \
        --warmup_epochs 0 \
        --kmeans_iters 10
else
    echo "[跳过] ${RQVAE_DATA} 下未找到 .npz 文件（竞赛平台硬编码路径）。"
    echo "       将 SASRec 输出的向量文件放置到 ${RQVAE_DATA} 后可启用训练。"
fi

echo "[完成] 第二阶段结束。"


# ─────────────────────────────────────────────────────────────────────────────
# 第三阶段 — GR：Qwen2 生成式推荐 SFT 微调
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================="
echo " [开始] 第三阶段 — GR：Qwen2 生成式推荐（SFT）"
echo "========================================================="
# 必须从项目根目录运行，因为 train_gr.py 使用 "from gr.custom_dataset import *"（包相对导入）
cd "${REPO_ROOT}"

echo "[信息] 验证 GR 模块导入..."
python - <<'PYEOF'
import sys
sys.path.insert(0, '.')
from gr.arguments import ModelArguments, DataTrainingArguments
from gr.utils import load_model_and_tokenizer, load_item2token_dict
from gr.custom_dataset import CustomTrainDataset
print("[OK]   arguments.py      -> ModelArguments, DataTrainingArguments")
print("[OK]   utils.py          -> load_model_and_tokenizer, load_item2token_dict")
print("[OK]   custom_dataset.py -> CustomTrainDataset（IterableDataset，Token 序列构造器）")
PYEOF

GR_MODEL="${USER_CACHE_PATH:-$DEMO_CACHE}/qwen_init2"
GR_DATA="${USER_CACHE_PATH:-$DEMO_CACHE}/train_data.json"
GR_TOKENS="${USER_CACHE_PATH:-$DEMO_CACHE}/emb_infer/sinkhorn10"

if [[ -d "${GR_MODEL}" && -f "${GR_DATA}" && -d "${GR_TOKENS}" ]]; then
    echo "[信息] 检测到 Qwen2 权重、训练数据及商品→Token 映射表。"
    echo "[信息] 使用配置 gr/gr_train.json 启动 GR SFT 微调..."
    export TRAIN_CKPT_PATH="${DEMO_CACHE}/gr_ckpt"
    export USER_CACHE_PATH="${USER_CACHE_PATH:-$DEMO_CACHE}"
    python gr/train_gr.py --config gr/gr_train.json
else
    echo "[跳过] GR 阶段缺少以下一项或多项前置文件："
    echo "         Qwen2 权重     : ${GR_MODEL}"
    echo "         训练数据文件   : ${GR_DATA}"
    echo "         商品→Token 映射: ${GR_TOKENS}"
    echo "       设置 USER_CACHE_PATH 并准备上述文件后可启用 SFT 微调。"
fi

echo "[完成] 第三阶段结束。"


# ─────────────────────────────────────────────────────────────────────────────
# GPU 显存状态
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================="
echo " [信息] 运行结束后 GPU 显存状态"
echo "========================================================="
nvidia-smi

echo ""
echo "========================================================="
echo " [完成] LLM4Rec 全流水线演示结束。"
echo "========================================================="
