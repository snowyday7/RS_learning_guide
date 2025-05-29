# 第7章：因子分解机及其变种 (Factorization Machines and Variants)

在处理高维稀疏数据，尤其是包含大量类别特征的推荐场景中，如何有效地捕捉特征之间的交互关系至关重要。因子分解机 (Factorization Machines, FM) 及其后续的深度学习变种，如DeepFM、NFM、xDeepFM等，为解决这一问题提供了强大的工具。

## 7.1 因子分解机 (Factorization Machines, FM)

由 Steffen Rendle 在2010年提出的因子分解机 (FM)，是一种通用的监督学习模型，特别擅长处理高维稀疏数据下的特征组合问题。它可以看作是带特征交叉的线性模型和矩阵分解的推广。

### 7.1.1 FM模型结构

对于一个给定的特征向量 $\mathbf{x} = (x_1, x_2, ..., x_n)$，FM模型预测目标 $y$ 的公式如下（以二阶FM为例）：
\[ \hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j \]
其中：
*   $w_0$：全局偏置项。
*   $w_i$：第 $i$ 个特征的权重（线性部分）。
*   $\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^{k} v_{if} v_{jf}$：第 $i$ 个特征和第 $j$ 个特征的交叉项权重。这里是FM的核心思想，它将每个特征 $x_i$ 关联一个 $k$ 维的隐向量 $\mathbf{v}_i = (v_{i1}, v_{i2}, ..., v_{ik})$。两个特征的交叉权重通过它们对应隐向量的点积来计算。
*   $k$：隐向量的维度（潜因子数量），是一个超参数。

**与多项式回归的区别**：
传统的多项式回归模型中，交叉项 $x_i x_j$ 的权重 $w_{ij}$ 是完全独立的参数。在数据稀疏的情况下（很多 $x_i x_j=0$），$w_{ij}$ 很难被准确学习。而FM通过引入隐向量 $\mathbf{v}_i$ 和 $\mathbf{v}_j$ 来参数化 $w_{ij}$，使得即使 $x_i$ 和 $x_j$ 从未同时出现过，只要它们分别与其他特征共同出现过（从而学习到各自的 $\mathbf{v}_i$ 和 $\mathbf{v}_j$），模型依然可以估计它们的交叉强度。这大大增强了模型在稀疏数据下的泛化能力。

**计算复杂度**：
直接计算二阶交叉项的复杂度是 $O(kn^2)$。通过数学变换，可以将其优化到 $O(kn)$：
\[ \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^{k} \left( \left( \sum_{i=1}^{n} v_{if} x_i \right)^2 - \sum_{i=1}^{n} v_{if}^2 x_i^2 \right) \]

### 7.1.2 FM的应用

FM可以用于回归、分类（如通过Sigmoid函数 $\sigma(\hat{y})$）、排序等任务。
在推荐系统中，特征向量 $\mathbf{x}$ 可以包含：
*   用户ID (one-hot编码)
*   物品ID (one-hot编码)
*   用户画像特征 (如年龄段、性别、职业等)
*   物品属性特征 (如类别、品牌、标签等)
*   上下文特征 (如时间、设备等)

FM能够自动学习这些不同来源特征之间的二阶交互。

### 7.1.3 FM的优缺点

**优点：**
*   有效处理高维稀疏数据。
*   能够学习特征间的二阶交互，即使这些特征组合在训练数据中很少或从未出现。
*   计算效率高（线性时间复杂度）。
*   通用性强，可用于多种预测任务。

**缺点：**
*   只能捕捉二阶特征交互，对于更高阶的复杂交互模式可能无能为力。
*   隐向量维度 $k$ 的选择对模型性能有影响。

## 7.2 基于深度学习的FM变种

为了克服FM仅能表达二阶交互的局限性，并利用深度神经网络强大的非线性建模能力，研究者们提出了一系列结合FM和DNN的模型。

### 7.2.1 Wide & Deep Learning (Google, 2016)

虽然不是直接的FM变种，但Wide & Deep模型为后续许多结合浅层和深层结构的模型提供了思路。

*   **Wide部分 (Memorization)**：通常是一个广义线性模型，输入包括原始特征和人工设计的交叉特征（例如，AND(user_installed_app, impression_app)）。它能有效记忆一些频繁出现的、直接的规则。
*   **Deep部分 (Generalization)**：是一个前馈神经网络（MLP），输入是原始特征（尤其是类别特征）的低维嵌入向量。它能学习特征之间的高阶非线性关系，提高模型的泛化能力。
*   两者共同训练，它们的输出（通常是logit）相加后通过Sigmoid函数得到最终预测。

**优点**：结合了记忆和泛化的能力。
**缺点**：Wide部分依赖人工特征工程，设计有效的交叉特征需要领域知识且耗时。

### 7.2.2 DeepFM (Huawei, 2017)

DeepFM旨在结合FM的低阶交互能力和DNN的高阶交互能力，并且实现端到端的训练，无需人工特征工程。

*   **模型结构**：DeepFM包含两部分：FM部分和DNN部分，它们共享相同的输入特征嵌入层。
    *   **FM部分**：与标准的FM类似，负责学习一阶特征和二阶特征交互。其输出是：
        \[ y_{FM} = \text{Linear}(\mathbf{x}) + \text{FM_Interaction}(\mathbf{x}) \]
        其中 $\text{Linear}(\mathbf{x}) = w_0 + \sum w_i x_i$，$\text{FM_Interaction}(\mathbf{x}) = \sum \sum \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$。
    *   **DNN部分**：是一个标准的前馈神经网络。输入是将所有特征的嵌入向量拼接 (concatenate) 起来，然后经过多个隐藏层（通常使用ReLU激活函数）学习高阶特征交互。其输出是：
        \[ y_{DNN} = \text{MLP}(\text{concat}(\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_n)) \]
        其中 $\mathbf{e}_i$ 是第 $i$ 个特征域的嵌入向量。
    *   **最终预测**：
        \[ \hat{y} = \sigma(y_{FM} + y_{DNN}) \]

*   **共享嵌入层**：这是DeepFM的一个重要特点。原始的类别特征（如user_id, item_id, gender, category）先被转换为one-hot编码，然后每个激活的one-hot特征（即值为1的维度）都通过一个共享的嵌入矩阵映射为一个低维稠密向量。这些嵌入向量同时服务于FM部分的隐向量计算和DNN部分的输入。

**优点：**
*   端到端学习，不需要人工设计交叉特征。
*   同时学习低阶和高阶特征交互。
*   共享嵌入层提高了效率和特征利用率。

**缺点：**
*   DNN部分学习到的高阶交互是隐式的，具体是哪些特征在如何交互不够明确。

### 7.2.3 NFM (Neural Factorization Machine) (National University of Singapore, 2017)

NFM旨在将FM的二阶交互思想更自然地融入到神经网络中，并在此基础上学习更高阶的交互。

*   **模型结构**：
    1.  **嵌入层 (Embedding Layer)**：同DeepFM，将稀疏输入特征映射为稠密嵌入向量。
    2.  **特征交叉池化层 (Bi-Interaction Pooling Layer)**：这是NFM的核心。对于输入的一组特征嵌入向量 $\{\mathbf{v}_1 x_1, \mathbf{v}_2 x_2, ..., \mathbf{v}_n x_n\}$（这里 $x_i$ 通常是0或1，表示特征是否存在），该层计算它们两两之间的元素积 (element-wise product, Hadamard product)：
        \[ f_{BI}(\mathcal{V}_x) = \sum_{i=1}^{n} \sum_{j=i+1}^{n} (\mathbf{v}_i x_i) \odot (\mathbf{v}_j x_j) \]
        这个操作的结果是一个 $k$ 维向量，它编码了所有二阶特征交互的信息。FM的二阶项可以看作是这个向量各元素之和（$\sum_{f=1}^k (f_{BI})_f$）。
    3.  **隐藏层 (Hidden Layers)**：将Bi-Interaction Pooling层的输出向量输入到一个或多个全连接层（MLP）中，以学习更高阶的非线性特征交互。
    4.  **预测层 (Prediction Layer)**：最后通过一个全连接层输出预测值。

*   **与DeepFM的区别**：
    *   DeepFM的DNN部分直接将所有嵌入向量拼接后输入MLP，而NFM的MLP输入是经过Bi-Interaction Pooling处理后的二阶交互向量。
    *   NFM更侧重于在FM的二阶交互基础上进行深化，而DeepFM是FM和DNN的并行结构。

**优点：**
*   能够比FM更深入地建模二阶交互之上叠加的非线性关系。
*   结构相对清晰。

**缺点：**
*   Bi-Interaction Pooling的求和操作可能会损失一些信息。

### 7.2.4 xDeepFM (eXtreme Deep Factorization Machine) (Microsoft Research Asia & USTC, 2018)

xDeepFM旨在显式地、向量级地学习高阶特征交互，同时避免了传统DNN中特征交互的隐式性和位级 (bit-wise) 交互的低效性。

*   **模型结构**：xDeepFM由三部分组成，它们共享相同的嵌入层：
    1.  **线性部分 (Linear Part)**：与FM中的一阶项类似，负责学习原始特征的线性贡献。
    2.  **压缩交互网络 (Compressed Interaction Network, CIN)**：这是xDeepFM的核心创新。CIN通过一种特殊的网络结构，在向量级别显式地学习不同阶数的特征交互。
        *   CIN的输入是所有特征域的嵌入向量组成的矩阵 $X^0 \in \mathbb{R}^{m \times k}$ ($m$ 是特征域数量，$k$ 是嵌入维度)。
        *   CIN包含多层，第 $h$ 层的输出 $X^h \in \mathbb{R}^{H_h \times k}$ 是由 $X^{h-1}$ 和 $X^0$ 通过一种“外积-卷积”类似的操作计算得到的。具体来说，$X^h_{i,*} = \sum_{j=1}^{H_{h-1}} \sum_{l=1}^{m} W^{h,j,l}_{i} (X^{h-1}_{j,*} \odot X^0_{l,*})$，其中 $W$ 是可学习的参数（可以看作卷积核）。这个操作使得每一层的输出都包含了更高一阶的特征交互信息，并且交互是在整个嵌入向量层面进行的。
        *   每一层 $X^h$ 都会经过一个sum pooling操作（$\sum_{i=1}^{H_h} X^h_{i,*}$），得到一个 $k$ 维向量，然后将所有层的sum pooling结果拼接起来，再通过一个线性层得到CIN的最终输出 $p^+ = [p^1, p^2, ..., p^D]$ (D是CIN的深度)。
    3.  **普通DNN部分 (Plain DNN Part)**：与DeepFM中的DNN类似，负责隐式地学习高阶特征交互。
    *   **最终预测**：
        \[ \hat{y} = \sigma(y_{Linear} + y_{CIN} + y_{DNN}) \]

**优点：**
*   CIN能够显式地、向量级地学习有界度的高阶特征交互。
*   结合了显式高阶交互（CIN）和隐式高阶交互（DNN）。
*   在许多数据集上表现优于DeepFM等模型。

**缺点：**
*   CIN的计算复杂度相对较高，参数量也可能较大。

## 7.3 总结

因子分解机及其深度学习变种是现代推荐系统中非常重要的一类模型，它们在处理类别特征、学习特征交互方面表现出色。从FM的二阶交互，到DeepFM的并行低阶与隐式高阶交互，再到NFM的二阶交互深化，以及xDeepFM的显式向量级高阶交互，这些模型不断演进，以期更有效地从数据中提取有用的模式。

选择哪个模型取决于具体的应用场景、数据特性、计算资源以及对模型复杂度和可解释性的要求。通常，这些模型在点击率 (CTR) 预估、转化率 (CVR) 预估等任务中应用广泛。