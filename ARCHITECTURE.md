# 生成式推荐系统 (TIGER) —— 三层架构图

> GitHub 原生渲染 Mermaid 图表，直接打开即可查看

---

## 一、功能层：三阶段生成式推荐链路

```mermaid
graph TD
    START["👤 用户行为序列<br/>user → [item₁, item₂, item₃, ...]"]

    subgraph S1["🔷 Stage 1: SASRec 物品向量学习"]
        S1_IN["输入: 用户序列 + 稀疏特征 + 多模态embedding<br/>特征维度最高 4096"]
        S1_MODEL["模型: Transformer (1层, 1头, hidden=32)<br/>Loss: BCEWithLogitsLoss"]
        S1_OUT["输出: embeddings.npz<br/>每个物品 → 32维稠密向量"]
        S1_IN --> S1_MODEL --> S1_OUT
    end

    subgraph S2["🔷 Stage 2: RQ-VAE 语义ID量化"]
        S2_IN["输入: 32维物品向量"]
        S2_MODEL["模型: Encoder→残差量化器(3层)→Decoder<br/>码本: 2048 / 2048 / 1024<br/>Loss: MSE重建 + 量化损失"]
        S2_OUT["输出: worker_0_output.txt<br/>物品ID → &lt;a_42&gt;&lt;b_128&gt;&lt;c_7&gt;"]
        S2_IN --> S2_MODEL --> S2_OUT
    end

    subgraph S3["🔷 Stage 3: Qwen2 生成式推荐"]
        S3_IN["输入: 用户历史语义ID序列 + item2token映射"]
        S3_MODEL["模型: Qwen2ForCausalLM (全参数训练)<br/>Loss: CrossEntropy (仅对目标物品token)<br/>fp16, grad_accum=4, max_steps=80000"]
        S3_OUT["输出: 微调后的Qwen2<br/>自回归生成下一个物品语义ID"]
        S3_IN --> S3_MODEL --> S3_OUT
    end

    START --> S1
    S1_OUT --> S2
    S2_OUT --> S3
    S3_OUT --> RESULT["✅ 端到端跑通<br/>各阶段loss均收敛"]
```

---

## 二、系统层：模块设计与交互

### 2.1 为什么分成三阶段？

```mermaid
graph LR
    subgraph PROBLEM["核心问题"]
        P1["LLM 只能处理 token<br/>不能直接吃稠密向量"]
        P2["物品特征维度很高<br/>(最高4096维)"]
        P3["需要捕捉用户序列<br/>中的顺序依赖"]
    end

    subgraph SOLUTION["三阶段方案"]
        S1["Stage 1 SASRec<br/>把物品多模态信息<br/>压缩成32维向量<br/>同时建模序列依赖"]
        S2["Stage 2 RQ-VAE<br/>把连续向量离散化<br/>变成LLM能吃的token<br/>= 建造LLM词汇表"]
        S3["Stage 3 Qwen2<br/>自回归生成下一物品<br/>利用LLM长序列建模能力"]
    end

    PROBLEM --> SOLUTION
```

### 2.2 为什么选 RQ-VAE（残差量化）而不是普通 VQ？

| 方案 | 表达能力 | 冲突率 | 缺点 |
|---|---|---|---|
| 普通 VQ (1层码本) | 单层 2048 | 高，大量物品碰撞 | 一个 token 区分度不够 |
| 单层大码本 (20万+) | 单层 20万 | 一般 | 码本太大训练不稳定 |
| **RQ-VAE (3层残差)** ✅ | 2048×2048×1024 ≈ 43亿组合 | **~2.2%** | 稍复杂 |
| 哈希 | 无法反向传播 | — | 不可学习 |

### 2.3 为什么选 Qwen2 而不是其他 LLM？

| 方案 | 是否考虑 | 选择理由 |
|---|---|---|
| Qwen2 ✅ | 最终选择 | 中文推荐场景，Qwen系列对中文友好；架构标准，HuggingFace 生态完善 |
| LLaMA | 备选 | 主要面向英文，中文推荐场景欠佳 |
| GPT-2 | 备选 | 模型太老，长序列建模能力弱 |
| 从头训 Transformer | 备选 | SASRec已经做了小Transformer，LLM需要更大容量 |

### 2.4 数据流与模块交互

```mermaid
graph TD
    subgraph DATA["📦 原始数据"]
        D1["用户行为序列<br/>user → item list"]
        D2["物品多模态特征<br/>sparse + mm_emb"]
    end

    subgraph S1["Stage 1: SASRec"]
        S1A["训练: 序列预测<br/>主流程 main.py"]
        S1B["导出: feat2emb 对每个物品<br/>生成32维统一向量"]
    end

    subgraph FS["📁 中间文件"]
        F1["embeddings.npz<br/>(N个物品, 32维)"]
    end

    subgraph S2["Stage 2: RQ-VAE"]
        S2A["训练: AE + 残差量化<br/>rqvae_train.py"]
        S2B["推理: 碰撞消解50轮<br/>rqvae_infer.py"]
    end

    subgraph MAP["📁 语义ID映射"]
        F2["worker_0_output.txt<br/>item_id → &lt;a_X&gt;&lt;b_Y&gt;&lt;c_Z&gt;"]
    end

    subgraph S3["Stage 3: Qwen2 SFT"]
        S3A["构建 item2token dict"]
        S3B["CustomTrainDataset<br/>历史序列→token序列"]
        S3C["HuggingFace Trainer<br/>causal LM 训练"]
    end

    D1 --> S1A
    D2 --> S1A
    S1A --> S1B
    S1B --> F1
    F1 --> S2A
    S2A --> S2B
    S2B --> F2
    D1 --> S3A
    F2 --> S3A
    S3A --> S3B
    S3B --> S3C
```

### 2.5 关键设计决策

```mermaid
graph TD
    subgraph Q1["❓ 损失只算目标物品"]
        ANS1["CustomTrainDataset 中<br/>历史token label设为-100<br/>只有目标物品3个token参与loss<br/>→ 模型专注学'推荐'而非'复述历史'"]
    end

    subgraph Q2["❓ Sinkhorn平衡"]
        ANS2["RQ-VAE inference时<br/>对最后一层码本施加Sinkhorn<br/>死码率 15%→3%<br/>→ 码本利用率大幅提升"]
    end

    subgraph Q3["❓ 碰撞消解50轮迭代"]
        ANS3["多个物品映射到相同语义ID时<br/>对该批次重新量化<br/>逐轮消解冲突<br/>→ 最终碰撞率 ~2.2%"]
    end

    subgraph Q4["❓ SASRec特征融合"]
        ANS4["物品embedding + sparse特征emb<br/>+ mm_emb (Linear投影) + continual特征<br/>→ 通过itemdnn融合为统一32维<br/>→ 多模态信息无损压缩"]
    end
```

---

## 三、工程层：性能·稳定性·可观测性

```mermaid
graph TD
    subgraph PERF["⚡ 性能"]
        P1["fp16 混合精度<br/>GR阶段显存减半"]
        P2["Flash Attention<br/>SASRec使用sdpa"]
        P3["梯度累积×4<br/>等效batch=64不扩显存"]
        P4["RQ-VAE大batch=8192<br/>充分利用GPU"]
    end

    subgraph STABLE["🛡️ 稳定性"]
        ST1["梯度裁剪 max_norm=1.0<br/>防止梯度爆炸"]
        ST2["NaN检测<br/>trainer._check_nan()<br/>发现NaN直接抛异常"]
        ST3["Checkpoint保存<br/>SASRec每epoch<br/>RQ-VAE每20epoch<br/>GR每5000步"]
        ST4["碰撞消解保障<br/>推理时50轮迭代消冲突"]
    end

    subgraph OBS["📊 可观测性"]
        O1["TensorBoard<br/>SASRec + RQ-VAE"]
        O2["HuggingFace Trainer<br/>每100步 log一次"]
        O3["冲突率监控<br/>RQ-VAE每20epoch"]
        O4["数据质量检查<br/>check_nan.py<br/>check_stats.py<br/>verify_fix.py"]
    end

    subgraph DEPLOY["🚀 部署(未涉及)"]
        DP1["在线推理服务 ❌"]
        DP2["模型Serving ❌"]
        DP3["AB实验框架 ❌"]
        DP4["容灾/回滚预案 ❌"]
    end
```

---

## 附：关键指标速查

| 阶段 | 指标 | 数值 |
|---|---|---|
| SASRec | 训练 Loss | 1.8 → 1.2 |
| SASRec | 验证 Loss | ~1.2 (稳定) |
| RQ-VAE | 重建 MSE | ~0.06 |
| RQ-VAE | **冲突率** | **~2.2%** |
| RQ-VAE | 死码率(无Sinkhorn) | 15% |
| RQ-VAE | 死码率(有Sinkhorn) | <3% |
| Qwen2 SFT | 训练 Loss | 2.9 → 1.5 |
| Qwen2 SFT | 梯度范数 | ~1.7 (稳定) |

---

## 附：文件结构速查

```
llm4rec/
├── sasrec/           # Stage 1: 序列推荐 → 物品向量
│   ├── main.py       #   训练入口 + embed导出
│   ├── model.py      #   SASRec Transformer 模型
│   └── dataset.py    #   多模态特征数据加载
├── rqvae/            # Stage 2: 向量 → 语义ID
│   ├── train/        #   训练: Encoder + RQ + Decoder
│   │   ├── rqvae_train.py
│   │   ├── rqvae_model.py
│   │   └── trainer.py
│   └── infer/        #   推理: 碰撞消解 + 输出映射表
│       ├── rqvae_infer.py
│       └── rqvae_model.py
├── gr/               # Stage 3: LLM 生成式推荐
│   ├── train_gr.py   #   HuggingFace Trainer 主流程
│   ├── custom_dataset.py  # Token化数据集
│   ├── utils.py      #   模型加载
│   └── gr_train.json #   训练配置
├── data/             # 原始数据
├── check_nan.py      # 数据质量检查
├── check_stats.py
└── verify_fix.py
```
