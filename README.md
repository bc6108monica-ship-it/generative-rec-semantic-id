**llm4rec**：参考 **TIGER** (Transfomer Index for GEnerative Recommenders) 架构实现的生成式推荐项目，参加的腾讯广告大赛。

1、先通过sasrec文件夹中代码获取每个item的多模态embedding表示，此时每个item用一个向量表示。

2、再通过rqvae文件夹中代码输入此embedding表示来构建码本，此时每个item用3个token或4个token表示（看码本层数）。

3、最后通过gr文件夹中代码构建生成式pipeline，即用之前交互的item去预测下一个item。




