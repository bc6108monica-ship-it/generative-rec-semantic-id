# RQ-VAE Training Framework

这是一个基于 PyTorch 实现的 **RQ-VAE (Residual Quantized Variational Autoencoder)** 训练框架。

该项目旨在通过残差量化（Residual Quantization）将高维向量（Embeddings）压缩为离散的码本索引序列（Discrete Codes）。这种技术常用于生成式检索（Generative Retrieval）或向量压缩场景。项目包含了完整的训练流程、K-Means 初始化、Sinkhorn 对齐算法以及基于 MLP 的编码器/解码器架构。

## ✨ 核心特性

- **残差量化 (Residual Quantization):** 支持多级码本（Codebooks），逐级逼近原始向量，降低量化误差。
- **灵活的架构:**
  - **Encoder/Decoder:** 可配置层数和维度的 MLP 结构。
  - **Codebook:** 支持自定义每一层的码本大小 (`num_emb_list`) 和维度 (`e_dim`)。
- **高级初始化与优化:**
  - 支持 **K-Means** 初始化码本，加速收敛。
  - 集成 **Sinkhorn Algorithm** (可选)，用于优化码本利用率，避免“死码”问题。
- **监控指标:** 除了常规的 MSE/Reconstruction Loss，还内置了 **Collision Rate**（冲突率）监控，用于评估量化后的离散码是否能唯一标识原始 Item。
- **训练特性:** 集成 TensorBoard 日志、学习率预热 (Warmup)、梯度裁剪、最佳模型自动保存。

## 📂 目录结构

```
.
├── rqvae_train.py    # 训练主入口，负责参数解析和训练流程启动
├── trainer.py        # 训练器类，包含训练循环、验证、checkpoint保存逻辑
├── rqvae_model.py    # 模型定义 (RQVAE, ResidualVectorQuantizer, MLPLayers)
├── datasets.py       # 数据加载器 (CustomNpzFile)
├── utils.py          # 工具函数 (日志、路径管理等)
└── run.sh            # 启动脚本
```

## 💾 数据准备

数据加载器 (`datasets.py` 中的 `CustomNpzFile`) 期望数据为 `.npz` 格式。

- **输入路径:** 通过 `--data_path` 指定包含 `.npz` 文件的目录。
- **文件格式:** 每个 `.npz` 文件需包含以下 Key：
  - `embs`: 输入的向量数据 (Float)。
  - `ids`: 对应的 ID (虽不直接参与 Loss 计算，但可能用于后续索引构建)。

数据读取逻辑会自动遍历指定目录下的所有 `.npz` 文件并拼接。

## 🚀 快速开始

### 1. 修改启动脚本

编辑 `run.sh` 或直接运行命令。你可以通过环境变量或命令行参数调整配置。

Bash

```
# run.sh 示例
python -u rqvae_train.py \
  --data_path "/path/to/your/npz/data" \
  --ckpt_dir "./checkpoints" \
  --batch_size 8192 \
  --lr 3e-4 \
  --epochs 1000
```

### 2. 关键参数说明

在 `rqvae_train.py` 中定义了以下主要参数：

**模型架构参数:**

- `--layers`: MLP 隐藏层的维度列表。默认: `[4096, 2048, 1024, 512, 256, 128, 64]`。
- `--num_emb_list`: 每一层残差量化的码本大小。例如 `[2048, 2048, 1024]` 表示使用 3 层量化。
- `--e_dim`: 码本中 Embedding 的维度。默认 `32`。
- `--dropout_prob`: Dropout 概率。

**训练参数:**

- `--loss_type`: 损失函数类型，支持 `mse` 或 `l1`。
- `--quant_loss_weight`: 量化损失的权重。默认 `1.0`。
- `--beta`: Commitment loss 的系数。默认 `0.25`。
- `--kmeans_init`: 是否使用 K-Means 初始化码本 (推荐 True)。
- `--sk_epsilons`: Sinkhorn 算法的 epsilon 参数列表，对应每一层码本。

**优化器参数:**

- `--lr`: 学习率。
- `--learner`: 优化器类型 (AdamW, Adam, SGD 等)。
- `--lr_scheduler_type`: 学习率调度策略 (constant, linear)。
- `--warmup_epochs`: 预热轮数。

## 📊 输出与日志

训练过程中会产生以下输出：

1. **控制台日志:** 显示当前的 Epoch、Training Loss、Reconstruction Loss 以及 Evaluation 阶段的 Collision Rate。
2. **TensorBoard Logs:** 保存在 `./logs` 目录下。
   - `Step/loss_total`: 总损失
   - `Step/loss_recon`: 重建损失
   - `Epoch/collision_rate`: 验证集的码本冲突率
3. **Checkpoints:** 保存在 `--ckpt_dir` 指定的目录下。
   - `best_loss_model.pth`: Loss 最低的模型。
   - `best_collision_model.pth`: 冲突率最低的模型。
   - `epoch_x_collision_y.pth`: 按照一定规则保留的历史模型。

## 🧩 模型原理简述

RQ-VAE 的核心在于 **Residual Vector Quantizer (RVQ)**。

1. **Encoder:** 将输入 $x$ 映射到潜在空间 $z$。
2. **RVQ:**
   - 第 1 层量化器对 $z$ 进行量化得到 $z_1$，计算残差 $r_1 = z - z_1$。
   - 第 2 层量化器对 $r_1$ 进行量化得到 $z_2$，计算残差 $r_2 = r_1 - z_2$。
   - ...以此类推。
   - 最终的量化表示为 $\hat{z} = \sum z_i$。
3. **Decoder:** 将 $\hat{z}$ 还原为重构向量 $\hat{x}$。

Collision Rate (冲突率):

代码中通过 trainer.py 的 _valid_epoch 计算。它统计在验证集中，有多少不同的输入被映射成了完全相同的离散 Code 序列。冲突率越低，表示生成的离散索引区分度越高。

## ⚠️ 注意事项

- 确保输入数据的维度与 `--layers` 的第一层维度或者是自动推断的 `data.dim` 匹配。