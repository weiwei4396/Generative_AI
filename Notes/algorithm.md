### Bioinformatics Algorithms
#### 1. 2018_NM_scVI (Deep generative modeling for single-cell transcriptomics)
[文章地址](https://www.nature.com/articles/s41592-018-0229-2) | [代码地址](https://github.com/scverse/scvi-tools)

scVI是一个基于VAE的深度生成概率模型，用层次贝叶斯框架建模scRNA-seq count，并显式校正测序深度、批次效应和零膨胀噪声。

- 单细胞基因表达量为整数，适合泊松或者负二项分布建模。一般来说，基因的方差是大于均值的，我个人更倾向于用负二项分布，如果测序比较深，也可以用泊松分布。
- 负二项分布可以看做是泊松分布和伽马分布混合得到，伽马分布用于建模基因表达的变异性。
- scVI中考虑了对数据dropout现象进行建模。
![scVI](https://github.com/weiwei4396/Generative_AI/blob/main/Pictures/scVI.png)
- 在概率模型图中，外圈的三个变量：library size $`l_n`$，latent space $`z_n`$，batch_id $`s_n`$，它们维度都是细胞数 $`n`$。
- 内圈的四个变量，维度都是**细胞数×基因数**($`n×g`$)，其中$`x_{ng}`$是在我们数据中真实观测到的基因表达的UMI数量，$`x_{ng}`$来源的$`y_{ng}`$表示基因$`g`$在细胞$`n`$中潜在的真实表达值，因为dropout现象真实值并不会全被检测到，因此变量$`h_{ng}`$用来控制dropout现象，它是个伯努利变量，由神经网络预测得到，取决于细胞状态的低维表征$`z_n`$和实验批次$`s_n`$。
- $`y_{ng}`$服从泊松分布，
- 灰色圆圈表示已知数据，白色圆圈表示未知数据。




