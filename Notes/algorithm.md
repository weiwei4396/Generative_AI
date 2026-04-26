### Bioinformatics Algorithms
#### 1. 2018_NM_scVI (Deep generative modeling for single-cell transcriptomics)
[文章地址](https://www.nature.com/articles/s41592-018-0229-2) | [代码地址](https://github.com/scverse/scvi-tools)
scVI 是一个基于VAE的深度生成概率模型，用层次贝叶斯框架建模单细胞 RNA-seq 计数数据，并显式校正测序深度、批次效应和零膨胀噪声。
- 单细胞基因表达量为整数，适合泊松或者负二项分布建模。一般来说，基因的方差是大于均值的，我个人更倾向于用负二项分布，如果测序比较深，也可以用泊松分布。
- scVI中考虑了对数据dropout现象进行建模。
![scVI](https://github.com/weiwei4396/Generative_AI/blob/main/Pictures/scVI.png)
- 




