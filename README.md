入门本项目需要有一定的生成式基础（不然码本都不知道是啥），建议先看懂谷歌的TIGER论文和lc-rec论文（https://github.com/RUCAIBox/LC-Rec/tree/main 这个论文有公开代码和公开数据集可跑），这也是我当时的学习路径。



**llm4rec**：参考 **TIGER** (Transfomer Index for GEnerative Recommenders) 架构实现的生成式推荐项目，参加的腾讯广告大赛。

1、先通过sasrec文件夹中代码获取每个item的多模态embedding表示，此时每个item用一个向量表示。

2、再通过rqvae文件夹中代码输入此embedding表示来构建码本，此时每个item用3个token或4个token表示（看码本层数）。

3、最后通过gr文件夹中代码构建生成式pipeline，即用之前交互的item去预测下一个item。



写到简历中可以按如下方法：

**项目名称：基于LLM与语义ID的生成式推荐系统研发** 

**项目角色：** 核心算法开发 

**项目描述：** 针对传统ID推荐稀疏性问题，设计并实现了一套基于 **Semantic ID** 的端到端生成式推荐架构，利用 LLM 的推理能力提升推荐效果。

- **多模态表征学习 (Step 1)**： 构建基于 **SASRec** 的序列化表征模块，融合 Item 的多模态信息（文本/图像/属性）生成高维稠密向量，有效捕捉 Item 间的细粒度语义关联，解决了传统 ID Embedding 冷启动难的问题。
- **层次化离散索引构建 (Step 2)**： 引入 **RQ-VAE (Residual Quantized VAE)** 技术实现向量量化，将 Item 的稠密向量映射为 3-4 层深度的**层次化语义 ID (Semantic IDs)**。构建了紧凑的 Codebook，将海量商品库压缩至有限的 Token 空间，使 LLM 能够直接处理推荐数据。
- **生成式预训练与微调 (Step 3)**： 基于 **Qwen2** 底座模型搭建生成式流水线。设计流式数据加载器 (IterableDataset)，将用户历史交互序列转化为 Semantic ID 序列，采用 **Next Item Prediction** 任务进行 **SFT (监督微调)**。利用 DeepSpeed 进行分布式训练优化，实现了从用户行为到目标 Item 的自回归生成推荐。

