# 第4章：矩阵分解 (Matrix Factorization)

矩阵分解 (Matrix Factorization, MF) 是一类在协同过滤中非常流行且有效的基于模型的推荐算法。其核心思想是将高维稀疏的用户-物品交互矩阵（如评分矩阵）分解为两个或多个低维稠密的因子矩阵的乘积。这些低维因子可以被看作是用户和物品的潜在特征 (Latent Features)。

## 4.1 矩阵分解的核心思想

假设我们有一个用户-物品评分矩阵 $R$ (大小为 $M \times N$，其中 $M$ 是用户数，$N$ 是物品数)，$R_{ui}$ 表示用户 $u$ 对物品 $i$ 的评分。由于大部分用户只对少量物品进行过评分，这个矩阵通常是非常稀疏的。

矩阵分解的目标是找到两个低维矩阵：

*   用户因子矩阵 $P$ (大小为 $M \times K$)
*   物品因子矩阵 $Q$ (大小为 $N \times K$)

使得它们的乘积 $P Q^T$ 能够近似重构原始的评分矩阵 $R$：
\[ R \approx P Q^T \]

其中，$K$ 是潜因子的数量，通常远小于 $M$ 和 $N$ ($K \ll M, K \ll N$)。

*   $P$ 的每一行 $\vec{p}_u$ (长度为 $K$) 代表用户 $u$ 的潜在特征向量。
*   $Q$ 的每一行 $\vec{q}_i$ (长度为 $K$) 代表物品 $i$ 的潜在特征向量。

用户 $u$ 对物品 $i$ 的预测评分 $\hat{r}_{ui}$ 可以通过用户 $u$ 的潜在特征向量 $\vec{p}_u$ 和物品 $i$ 的潜在特征向量 $\vec{q}_i$ 的点积来计算：
\[ \hat{r}_{ui} = \vec{p}_u \cdot \vec{q}_i^T = \sum_{k=1}^{K} p_{uk} q_{ik} \]

这些潜因子 $K$ 可以被理解为一些抽象的维度，比如电影的“喜剧成分”、“动作成分”、“艺术性”等，或者用户的“对喜剧的偏爱程度”、“对动作片的偏爱程度”等。模型通过学习数据自动捕获这些潜在的关联。

## 4.2 常见的矩阵分解模型

### 4.2.1 奇异值分解 (Singular Value Decomposition, SVD)

标准的SVD是矩阵分解的一种经典方法，可以将任意实矩阵 $R$ 分解为：
\[ R = U \Sigma V^T \]
其中 $U$ 和 $V$ 是正交矩阵，$U^T U = I$, $V^T V = I$，$\Sigma$ 是一个对角矩阵，对角线上的元素称为奇异值，并且按降序排列。

为了进行降维和预测，可以选择最大的 $K$ 个奇异值及其对应的 $U$ 和 $V$ 的列向量，得到近似矩阵：
\[ R_K = U_K \Sigma_K V_K^T \]
然而，SVD在推荐系统中的直接应用存在问题：
1.  **计算复杂度高**：对于大规模稀疏矩阵，SVD计算成本很高。
2.  **无法处理缺失值**：SVD要求输入矩阵是完整的，而评分矩阵通常有大量缺失值（未评分项）。如果用0或平均值填充，会引入噪声和偏差。

因此，实际推荐系统中常用的是针对评分预测任务进行优化的“FunkSVD”或类似的基于梯度下降的方法。

### 4.2.2 FunkSVD (Simon Funk's SVD / Latent Factor Model)

FunkSVD 是在 Netflix Prize 竞赛中被 Simon Funk 推广的一种矩阵分解方法。它并不直接进行SVD分解，而是通过最小化预测评分与真实评分之间的误差来直接学习用户因子矩阵 $P$ 和物品因子矩阵 $Q$。

**目标函数 (Objective Function)**：

最小化所有已知评分的平方误差和，并通常加入正则化项以防止过拟合：
\[ \min_{P,Q} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \vec{p}_u \cdot \vec{q}_i^T)^2 + \lambda (\|\vec{p}_u\|^2 + \|\vec{q}_i\|^2) \]
其中：
*   $\mathcal{K}$ 是所有已知评分 $(u,i)$ 的集合。
*   $r_{ui}$ 是用户 $u$ 对物品 $i$ 的真实评分。
*   $\hat{r}_{ui} = \vec{p}_u \cdot \vec{q}_i^T$ 是预测评分。
*   $\lambda$ 是正则化系数，控制模型复杂度。
*   $\|\vec{p}_u\|^2 = \sum_{k=1}^{K} p_{uk}^2$ 和 $\|\vec{q}_i\|^2 = \sum_{k=1}^{K} q_{ik}^2$ 是L2正则化项。

**优化算法**：

通常使用随机梯度下降 (Stochastic Gradient Descent, SGD) 或交替最小二乘 (Alternating Least Squares, ALS) 来优化目标函数。

*   **随机梯度下降 (SGD)**：
    对于每一个已知的评分 $r_{ui}$，计算预测误差 $e_{ui} = r_{ui} - \hat{r}_{ui}$。
    然后根据误差更新用户因子 $\vec{p}_u$ 和物品因子 $\vec{q}_i$：
    \[ \vec{p}_u \leftarrow \vec{p}_u + \eta (e_{ui} \cdot \vec{q}_i - \lambda \cdot \vec{p}_u) \]
    \[ \vec{q}_i \leftarrow \vec{q}_i + \eta (e_{ui} \cdot \vec{p}_u - \lambda \cdot \vec{q}_i) \]
    其中，$\eta$ 是学习率。
    这个过程会迭代进行，直到模型收敛或达到最大迭代次数。

*   **交替最小二乘 (ALS)**：
    ALS 的思想是：当固定 $P$ 时，目标函数是关于 $Q$ 的二次函数，可以解析地求解 $Q$；反之，当固定 $Q$ 时，目标函数是关于 $P$ 的二次函数，可以解析地求解 $P$。ALS通过交替固定一个矩阵并优化另一个矩阵来迭代求解。
    ALS 对于处理大规模稀疏数据和并行化计算比较友好。

### 4.2.3 考虑偏置项的SVD (SVD++ / Biased MF)

为了提高预测准确性，可以在FunkSVD的基础上引入偏置项 (Bias terms)：
\[ \hat{r}_{ui} = \mu + b_u + b_i + \vec{p}_u \cdot \vec{q}_i^T \]
其中：
*   $\mu$：全局平均评分。
*   $b_u$：用户偏置项，表示用户 $u$ 的评分倾向（例如，某些用户倾向于打高分，某些用户倾向于打低分）。
*   $b_i$：物品偏置项，表示物品 $i$ 本身的受欢迎程度（例如，某些电影本身质量较高，平均得分就高）。

目标函数相应地修改为：
\[ \min_{P,Q,b_u,b_i} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - (\mu + b_u + b_i + \vec{p}_u \cdot \vec{q}_i^T))^2 + \lambda (\|\vec{p}_u\|^2 + \|\vec{q}_i\|^2 + b_u^2 + b_i^2) \]

同样可以使用SGD进行优化，更新规则会相应地包含对 $b_u$ 和 $b_i$ 的更新：
\[ b_u \leftarrow b_u + \eta (e_{ui} - \lambda \cdot b_u) \]
\[ b_i \leftarrow b_i + \eta (e_{ui} - \lambda \cdot b_i) \]

### 4.2.4 考虑隐式反馈的矩阵分解 (MF for Implicit Feedback)

对于隐式反馈数据（如点击、购买、观看时长等，通常只有正反馈，没有明确的负反馈），评分 $r_{ui}$ 通常被处理为二元的 $c_{ui}$（1表示有交互，0表示无交互或未知）。

目标函数需要调整，例如 Yehuda Koren 等人提出的方法：
\[ \min_{P,Q} \sum_{(u,i)} w_{ui} (c_{ui} - \vec{p}_u \cdot \vec{q}_i^T)^2 + \lambda (\|\vec{p}_u\|^2 + \|\vec{q}_i\|^2) \]
其中 $w_{ui}$ 是置信度权重，对于观察到的交互 ($c_{ui}=1$)，$w_{ui}$ 可以设为一个较大的值（如 $1 + \alpha \cdot \text{interaction_frequency}$），对于未观察到的交互 ($c_{ui}=0$)，$w_{ui}$ 可以设为1。这表示我们对观察到的正反馈有较高置信度，而未观察到的交互可能是用户不喜欢，也可能只是用户没看到。

ALS 是解决这类问题常用的优化算法。

## 4.3 矩阵分解的优缺点

**优点：**

*   **较好的预测精度**：通过学习潜因子，能够捕捉到用户和物品之间更深层次的关联，通常比传统的基于邻域的协同过滤方法有更好的预测效果。
*   **处理数据稀疏性**：即使在稀疏数据上，MF也能通过低维表示学习到有意义的模式。
*   **模型相对简单，计算效率较高**：一旦模型训练完成，预测评分的计算非常快（向量点积）。训练过程虽然迭代，但可以通过SGD或ALS高效进行。
*   **可扩展性好**：特别是ALS，易于并行化处理大规模数据。
*   **灵活性**：可以方便地加入偏置项、时间动态性、隐式反馈等多种因素。

**缺点：**

*   **冷启动问题依然存在**：对于新用户或新物品，由于缺乏交互数据，难以学习其潜因子向量。需要结合其他方法（如基于内容的推荐或混合方法）来处理。
*   **可解释性较差**：学习到的潜因子通常是抽象的，难以赋予明确的语义解释，不如基于内容的推荐直观。
*   **依赖交互数据**：如果交互数据质量不高或存在严重偏差，模型效果会受影响。
*   **调参可能比较复杂**：潜因子数量 $K$、学习率 $\eta$、正则化系数 $\lambda$ 等超参数的选择对模型性能有较大影响。

## 4.4 矩阵分解的实现案例（基于FunkSVD的伪代码）

```python
import numpy as np

class FunkSVD:
    def __init__(self, n_users, n_items, n_factors=20, learning_rate=0.01, reg_lambda=0.1, n_epochs=30):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors       # 潜因子数量 K
        self.lr = learning_rate          # 学习率 eta
        self.reg = reg_lambda            # 正则化系数 lambda
        self.n_epochs = n_epochs         # 迭代次数

        # 初始化用户因子矩阵 P 和物品因子矩阵 Q (随机初始化)
        self.P = np.random.normal(scale=1./self.n_factors, size=(n_users, n_factors))
        self.Q = np.random.normal(scale=1./self.n_factors, size=(n_items, n_factors))
        
        # 可选：初始化偏置项
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0.0

    def fit(self, ratings_data): # ratings_data 是一个 (user_id, item_id, rating) 的列表/数组
        self.global_bias = np.mean([rating for _, _, rating in ratings_data])

        for epoch in range(self.n_epochs):
            np.random.shuffle(ratings_data) # 每轮迭代打乱数据顺序
            total_loss = 0
            for user_id, item_id, true_rating in ratings_data:
                # 预测评分
                pred_rating = self.predict_single(user_id, item_id)
                
                # 计算误差
                error = true_rating - pred_rating
                total_loss += error**2
                
                # 更新因子和偏置 (SGD)
                pu_old = self.P[user_id, :].copy()
                qi_old = self.Q[item_id, :].copy()
                
                self.P[user_id, :] += self.lr * (error * qi_old - self.reg * pu_old)
                self.Q[item_id, :] += self.lr * (error * pu_old - self.reg * qi_old)
                
                self.user_bias[user_id] += self.lr * (error - self.reg * self.user_bias[user_id])
                self.item_bias[item_id] += self.lr * (error - self.reg * self.item_bias[item_id])
            
            # 计算总损失 (加上正则化项)
            # total_loss += self.reg * (np.sum(self.P**2) + np.sum(self.Q**2) + np.sum(self.user_bias**2) + np.sum(self.item_bias**2))
            # print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {total_loss / len(ratings_data)}")
        return self

    def predict_single(self, user_id, item_id):
        prediction = self.global_bias + self.user_bias[user_id] + self.item_bias[item_id] \
                     + np.dot(self.P[user_id, :], self.Q[item_id, :])
        return prediction

    def predict(self, user_item_pairs): # user_item_pairs 是 (user_id, item_id) 的列表
        predictions = []
        for user_id, item_id in user_item_pairs:
            predictions.append(self.predict_single(user_id, item_id))
        return predictions

# 示例用法 (假设用户ID和物品ID已经映射到0-based整数索引)
# num_users = 100
# num_items = 50
# ratings = [(0, 0, 5), (0, 1, 3), (1, 0, 4), ... ] # (user_idx, item_idx, rating)

# model = FunkSVD(n_users=num_users, n_items=num_items, n_factors=10, n_epochs=20)
# model.fit(ratings)

# user_to_predict = 0
# items_to_predict_for_user = [(user_to_predict, item_idx) for item_idx in range(num_items) if (user_to_predict, item_idx, ANY_RATING) not in ratings]
# predicted_ratings = model.predict(items_to_predict_for_user)
# recommended_items = sorted(zip([p[1] for p in items_to_predict_for_user], predicted_ratings), key=lambda x: x[1], reverse=True)
# print(f"Recommendations for user {user_to_predict}: {recommended_items[:5]}")

```

矩阵分解是理解现代推荐系统（包括许多深度学习模型）的基础，因为它引入了“潜因子”这一核心概念，即将用户和物品嵌入到共享的低维空间中。