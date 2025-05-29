# 第8章：神经协同过滤与自编码器在推荐中的应用

传统的矩阵分解 (MF) 通常使用用户和物品潜因子的点积来预测评分或交互。虽然有效，但点积这种简单的线性操作可能不足以捕捉用户和物品之间复杂的非线性关系。本章将介绍如何使用神经网络来增强协同过滤，主要包括神经协同过滤 (NCF) 框架以及基于自编码器 (Autoencoder) 的推荐模型。

## 8.1 神经协同过滤 (Neural Collaborative Filtering, NCF)

由 He et al. (2017) 提出的神经协同过滤 (NCF) 框架，旨在用深度神经网络的强大表达能力来替代矩阵分解中的点积操作，从而学习更复杂的用户-物品交互函数。

### 8.1.1 NCF框架概述

NCF 的核心思想是：不再局限于用户和物品潜因子之间的简单线性组合（如点积），而是使用一个多层感知机 (Multi-Layer Perceptron, MLP) 来学习它们之间的交互。

**通用框架结构：**

1.  **输入层 (Input Layer)**：
    *   输入通常是用户ID和物品ID。这些ID通常是高维稀疏的one-hot编码向量。

2.  **嵌入层 (Embedding Layer)**：
    *   用户ID和物品ID分别通过各自的嵌入矩阵映射为低维稠密的潜因子向量（嵌入向量）。
    *   用户 $u$ 的嵌入向量：$\mathbf{p}_u = P^T \mathbf{v}_u^U$
    *   物品 $i$ 的嵌入向量：$\mathbf{q}_i = Q^T \mathbf{v}_i^I$
    *   其中 $\mathbf{v}_u^U$ 和 $\mathbf{v}_i^I$ 是one-hot向量，$P$ 和 $Q$ 是可学习的嵌入矩阵。

3.  **神经协同过滤层 (Neural CF Layers)**：
    *   这是NCF的核心，负责将用户嵌入向量 $\mathbf{p}_u$ 和物品嵌入向量 $\mathbf{q}_i$ 作为输入，通过一系列神经网络层来建模它们的交互。
    *   NCF框架提出了两种具体的实现方式，并将它们融合：
        *   **广义矩阵分解 (Generalized Matrix Factorization, GMF)**
        *   **多层感知机 (Multi-Layer Perceptron, MLP)**

4.  **输出层 (Output Layer)**：
    *   将神经协同过滤层的输出（一个向量）通过一个全连接层（通常不带或带Sigmoid激活）映射为一个预测值 $\hat{y}_{ui}$，表示用户 $u$ 对物品 $i$ 的偏好得分或交互概率。

### 8.1.2 NCF的具体模型

#### a. 广义矩阵分解 (Generalized Matrix Factorization, GMF)

GMF 可以看作是传统矩阵分解的一个直接推广。它将用户嵌入 $\mathbf{p}_u$ 和物品嵌入 $\mathbf{q}_i$ 进行元素积 (element-wise product, Hadamard product)，然后通过一个输出层得到预测值：
\[ \phi_{GMF}(\mathbf{p}_u, \mathbf{q}_i) = \mathbf{p}_u \odot \mathbf{q}_i \]
\[ \hat{y}_{ui}^{GMF} = \sigma(\mathbf{h}^T (\mathbf{p}_u \odot \mathbf{q}_i)) \]
其中 $\mathbf{h}$ 是输出层的权重向量，$\sigma$ 是激活函数（如Sigmoid）。如果 $\mathbf{h}$ 固定为全1向量且不使用激活函数，GMF就退化为标准的矩阵分解（点积）。GMF允许从数据中学习 $\mathbf{h}$，从而赋予不同潜因子维度不同的重要性。

#### b. 多层感知机 (Multi-Layer Perceptron, MLP)

MLP部分则使用标准的神经网络结构来学习用户和物品嵌入之间的交互：
1.  将用户嵌入 $\mathbf{p}_u$ 和物品嵌入 $\mathbf{q}_i$ 进行拼接 (concatenate)：$\mathbf{z}_1 = \begin{bmatrix} \mathbf{p}_u \\ \mathbf{q}_i \end{bmatrix}$
2.  将拼接后的向量输入到一个多层感知机中：
    \[ \mathbf{z}_2 = \phi_2(\mathbf{W}_2^T \mathbf{z}_1 + \mathbf{b}_2) \]
    \[ ... \]
    \[ \mathbf{z}_L = \phi_L(\mathbf{W}_L^T \mathbf{z}_{L-1} + \mathbf{b}_L) \]
    其中 $\mathbf{W}_l, \mathbf{b}_l, \phi_l$ 分别是第 $l$ 层的权重、偏置和激活函数（通常使用ReLU）。
3.  最后通过输出层得到预测值：
    \[ \hat{y}_{ui}^{MLP} = \sigma(\mathbf{h}^T \mathbf{z}_L) \]
MLP的深层结构使其能够学习用户和物品之间高度非线性的复杂交互关系。

#### c. 神经矩阵分解 (Neural Matrix Factorization, NeuMF)

NeuMF 是NCF框架下最终提出的融合模型，它结合了GMF的线性和MLP的非线性建模能力。

*   **融合策略**：NeuMF让GMF和MLP分别学习独立的嵌入向量，然后将它们最后一层隐藏层的输出进行拼接，再通过一个全连接层得到最终的预测值。
    \[ \mathbf{p}_u^{GMF}, \mathbf{q}_i^{GMF} \leftarrow \text{Embeddings for GMF} \]
    \[ \mathbf{p}_u^{MLP}, \mathbf{q}_i^{MLP} \leftarrow \text{Embeddings for MLP} \]
    \[ \phi^{GMF} = \mathbf{p}_u^{GMF} \odot \mathbf{q}_i^{GMF} \]
    \[ \phi^{MLP} = \text{MLP}(\begin{bmatrix} \mathbf{p}_u^{MLP} \\ \mathbf{q}_i^{MLP} \end{bmatrix}) \]
    \[ \hat{y}_{ui} = \sigma \left( \mathbf{h}^T \begin{bmatrix} \phi^{GMF} \\ \phi^{MLP} \end{bmatrix} \right) \]
*   **预训练与联合训练**：为了更好地训练NeuMF，可以先分别预训练GMF和MLP模型，然后用它们的参数初始化NeuMF的对应部分，再进行联合微调。

### 8.1.3 NCF的损失函数与优化

*   **损失函数**：
    *   对于显式反馈（如评分预测），可以使用平方损失：$L = \sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} (y_{ui} - \hat{y}_{ui})^2$
    *   对于隐式反馈（如点击预测，通常是二分类问题），可以使用二元交叉熵损失 (Binary Cross-Entropy)：
        \[ L = - \sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log (1 - \hat{y}_{ui}) \]
        其中 $\mathcal{Y}$ 是观察到的正反馈集合 ($y_{ui}=1$)，$\mathcal{Y}^-$ 是负反馈集合 ($y_{ui}=0$，通常通过负采样得到)。
*   **优化器**：常用Adam等基于梯度的优化算法。

### 8.1.4 NCF的优缺点

**优点：**
*   通过MLP引入非线性，能够捕捉比MF更复杂的用户-物品交互模式。
*   提供了一个灵活的框架，可以将不同的交互函数（如GMF和MLP）进行融合。
*   在一些数据集上表现优于传统MF。

**缺点：**
*   模型复杂度增加，参数量和计算量比MF大。
*   MLP部分的拼接操作可能不如点积或元素积那样直接地建模交互，有时效果提升不明显或依赖于精细的调参。
*   对于非常稀疏的数据，性能可能仍然受限。

## 8.2 基于自编码器的推荐模型 (Autoencoder-based Recommendation)

自编码器 (Autoencoder, AE) 是一种无监督的神经网络，其目标是学习输入数据的有效表示（编码），使得该表示能够尽可能精确地重构原始输入（解码）。在推荐系统中，自编码器被用来学习用户或物品的低维潜在表示。

### 8.2.1 AutoRec (Autoencoders for Collaborative Filtering)

由 Sedhain et al. (2015) 提出的 AutoRec 是一个较早将自编码器思想应用于协同过滤的模型。

*   **核心思想**：将用户（或物品）的部分评分向量作为输入，训练自编码器来重构该用户（或物品）的完整评分向量。隐藏层的激活可以被视为用户（或物品）的低维潜因子表示。
*   **两种形式**：
    *   **User-based AutoRec (U-AutoRec)**：输入是某个用户对所有物品的评分向量 $\mathbf{r}^{(u)} = (r_{u1}, r_{u2}, ..., r_{uN})$。模型试图预测 $\hat{\mathbf{r}}^{(u)}$。
        \[ h(\mathbf{r}^{(u)}; \theta) = f(W \cdot g(V \mathbf{r}^{(u)} + \mu) + b) \]
        其中 $V, W$ 是权重矩阵，$\mu, b$ 是偏置，$g, f$ 是激活函数。目标是最小化重构误差，通常只考虑已观测到的评分：
        \[ \min_{\theta} \sum_{u=1}^{M} \sum_{i: r_{ui} \text{ observed}} (r_{ui} - h_i(\mathbf{r}^{(u)}; \theta))^2 + \frac{\lambda}{2} (\|W\|_F^2 + \|V\|_F^2) \]
    *   **Item-based AutoRec (I-AutoRec)**：输入是某个物品被所有用户评分的向量 $\mathbf{r}^{(i)} = (r_{1i}, r_{2i}, ..., r_{Mi})$。模型试图预测 $\hat{\mathbf{r}}^{(i)}$。I-AutoRec 通常表现更好，因为物品的评分向量通常比用户的评分向量更稠密。

*   **优点**：模型结构简单，概念清晰。
*   **缺点**：
    *   输入向量维度非常高（用户数或物品数），导致参数量巨大。
    *   对于非常稀疏的输入，重构效果可能不佳。
    *   本质上还是浅层模型（单隐藏层）。

### 8.2.2 CDAE (Collaborative Denoising Auto-Encoders)

Wu et al. (2016) 提出的 CDAE 在AutoRec的基础上引入了去噪 (Denoising) 和用户偏置的思想。

*   **去噪**：在输入层对用户-物品交互向量进行部分随机损坏（如随机将一些观察到的交互置为0），模型需要从损坏的输入中恢复原始的完整交互。\这增强了模型的鲁棒性和泛化能力。
*   **用户偏置**：在编码器的输入中加入一个用户特定的偏置节点（或用户嵌入向量），使得模型能够学习到用户相关的潜在表示。
    输入变为用户 $u$ 的损坏后的物品交互向量 $\tilde{\mathbf{r}}_u$ 和用户 $u$ 的嵌入 $\mathbf{v}_u$。

### 8.2.3 Deep Autoencoders (e.g., Multinomial Variational Autoencoder - Multi-VAE)

更深层次的自编码器，如变分自编码器 (Variational Autoencoder, VAE)，也被用于推荐。

*   **Multi-VAE (Liang et al., 2018)**：
    *   专门为隐式反馈数据设计，假设用户的物品选择行为服从多项式分布 (Multinomial distribution)。
    *   使用VAE的框架，编码器将用户的历史交互（如点击的物品列表）映射为一个概率分布（通常是高斯分布的均值和方差），解码器从这个分布中采样一个潜因子向量，并试图重构用户对所有物品的交互概率。
    *   损失函数包含重构损失（如多项式似然的负对数）和KL散度正则项（使得潜因子分布接近标准正态分布）。
    *   Multi-VAE在处理隐式反馈和捕捉用户兴趣的不确定性方面表现出色。

### 8.2.4 自编码器在推荐中的一般思路

1.  **输入**：通常是用户对物品的交互向量（显式评分或隐式反馈）。
2.  **编码器 (Encoder)**：将高维稀疏的输入映射到一个低维稠密的潜在空间，得到用户的潜因子表示。
3.  **解码器 (Decoder)**：从潜因子表示重构用户对所有物品的偏好（预测评分或交互概率）。
4.  **训练**：最小化重构误差（只针对已观察到的交互进行计算），有时会加入正则化项或噪声。
5.  **预测**：对于一个给定的用户，将其交互数据输入编码器得到潜因子，然后用解码器预测其对未交互物品的偏好。

### 8.2.5 自编码器推荐模型的优缺点

**优点：**
*   能够学习非线性的用户/物品表示。
*   去噪自编码器能提高模型的鲁棒性。
*   VAE等概率模型能捕捉数据的不确定性。

**缺点：**
*   当用户/物品数量巨大时，输入/输出层维度很高，导致参数量大，计算开销大。
*   对于极度稀疏的数据，重构可能仍然困难。
*   模型设计和调参相对复杂。

## 8.3 总结

神经协同过滤 (NCF) 和基于自编码器的模型代表了使用神经网络改进传统协同过滤方法的两种不同思路。NCF通过直接用MLP替换点积来增强交互建模的非线性能力。自编码器则通过学习输入数据的压缩表示（潜因子）并重构原始输入来进行推荐。这些方法都展示了深度学习在捕捉复杂用户偏好和提升推荐性能方面的潜力。然而，它们也带来了模型复杂度和计算成本的增加，需要在实际应用中权衡。