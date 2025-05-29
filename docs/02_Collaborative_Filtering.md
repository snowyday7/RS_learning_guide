# 第2章：协同过滤 (Collaborative Filtering)

协同过滤 (Collaborative Filtering, CF) 是推荐系统中最常用和最经典的技术之一。其核心思想是基于用户群体的行为来做推荐，即“物以类聚，人以群分”。它不需要物品本身的内容信息，而是依赖于用户对物品的交互行为数据（如评分、购买、点击、浏览等）。

## 2.1 协同过滤的核心思想

协同过滤的基本假设是：如果用户A和用户B在过去对很多物品有相似的偏好（例如，都喜欢某些电影），那么用户A未来可能会喜欢用户B喜欢过的、但用户A尚未接触过的物品。

## 2.2 协同过滤的主要类型

协同过滤主要分为两大类：基于用户的协同过滤 (User-based CF) 和基于物品的协同过滤 (Item-based CF)。此外，还有基于模型的协同过滤，如矩阵分解，我们将在后续章节详细介绍。

### 2.2.1 基于用户的协同过滤 (User-based CF)

**基本步骤：**

1.  **收集用户偏好数据**：构建用户-物品交互矩阵，其中矩阵的元素可以是用户的评分、是否购买等。
2.  **找到相似用户**：对于目标用户 $u$，计算其与其他所有用户之间的相似度。常用的相似度计算方法包括：
    *   **Jaccard 相似系数 (Jaccard Index)**：衡量两个集合的相似性，适用于二元数据（如是否喜欢）。
        \[ J(A, B) = \frac{|A \cap B|}{|A \cup B|} \]
    *   **余弦相似度 (Cosine Similarity)**：衡量两个向量在方向上的相似性，适用于评分数据。
        \[ sim(u, v) = \cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \cdot \|\vec{v}\|} \]
        其中，$\vec{u}$ 和 $\vec{v}$ 是用户 $u$ 和用户 $v$ 的评分向量。
    *   **皮尔逊相关系数 (Pearson Correlation Coefficient)**：衡量两个变量的线性相关程度，考虑了用户的评分尺度差异。
        \[ sim(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}} \]
        其中，$I_{uv}$ 是用户 $u$ 和用户 $v$ 共同评分过的物品集合，$r_{ui}$ 是用户 $u$ 对物品 $i$ 的评分，$\bar{r}_u$ 是用户 $u$ 的平均评分。
3.  **选择Top-N相似用户**：根据相似度得分，选择与目标用户最相似的 $N$ 个用户。
4.  **生成推荐**：将这些相似用户喜欢过但目标用户尚未接触过的物品，根据一定的加权平均（通常使用相似度作为权重）预测目标用户对这些物品的评分，然后推荐评分最高的物品。
    预测用户 $u$ 对物品 $j$ 的评分 $\hat{r}_{uj}$ 可以是：
    \[ \hat{r}_{uj} = \bar{r}_u + \frac{\sum_{v \in N_u} sim(u, v) \cdot (r_{vj} - \bar{r}_v)}{\sum_{v \in N_u} |sim(u, v)|} \]
    其中，$N_u$ 是用户 $u$ 的 $N$ 个最相似用户集合。

**优点：**
*   能够发现用户潜在的新兴趣，推荐结果具有惊喜性。
*   不需要物品的内容信息。

**缺点：**
*   **数据稀疏性**：在大规模系统中，用户-物品交互矩阵通常非常稀疏，导致难以找到足够多的共同评分物品来准确计算用户相似度。
*   **计算量大**：用户数量庞大时，计算用户之间的相似度矩阵非常耗时（$O(M^2)$，M为用户数）。
*   **冷启动问题**：新用户由于缺乏历史行为数据，难以找到相似用户，从而无法为其生成有效推荐。
*   **可扩展性差**：随着用户和物品数量的增加，计算复杂度会急剧上升。

### 2.2.2 基于物品的协同过滤 (Item-based CF)

Item-based CF 由亚马逊在2003年提出，旨在解决User-based CF的可扩展性问题。其核心思想是：如果用户喜欢物品A，并且物品A和物品B很相似，那么用户也可能喜欢物品B。

**基本步骤：**

1.  **收集用户偏好数据**：同User-based CF。
2.  **计算物品相似度**：对于目标物品 $i$，计算其与其他所有物品之间的相似度。这里的相似度是基于用户对这些物品的共同行为来计算的。例如，如果很多用户同时喜欢物品 $i$ 和物品 $j$，那么 $i$ 和 $j$ 的相似度就高。
    常用的相似度计算方法（将物品视为向量，向量的维度是用户，值为用户对该物品的评分）：
    *   **余弦相似度**：
        \[ sim(i, j) = \cos(\vec{i}, \vec{j}) = \frac{\vec{i} \cdot \vec{j}}{\|\vec{i}\| \cdot \|\vec{j}\|} \]
        其中，$\vec{i}$ 和 $\vec{j}$ 是物品 $i$ 和物品 $j$ 的用户评分向量。
    *   **调整余弦相似度 (Adjusted Cosine Similarity)**：考虑到不同用户评分尺度不一致的问题，先对每个用户的评分进行中心化处理（减去该用户的平均评分）。
        \[ sim(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}} \]
        其中，$U_{ij}$ 是同时对物品 $i$ 和物品 $j$ 评过分的用户集合。
3.  **选择Top-K相似物品**：对于目标用户 $u$ 喜欢过的每个物品 $i$，找到与 $i$ 最相似的 $K$ 个物品。
4.  **生成推荐**：将这些相似物品（目标用户尚未接触过的）根据一定的加权平均（通常使用物品相似度和用户对相似物品的评分作为权重）预测目标用户对这些物品的评分，然后推荐评分最高的物品。
    预测用户 $u$ 对物品 $j$ 的评分 $\hat{r}_{uj}$ 可以是：
    \[ \hat{r}_{uj} = \frac{\sum_{i \in S_u \cap N_j} sim(i, j) \cdot r_{ui}}{\sum_{i \in S_u \cap N_j} |sim(i, j)|} \]
    其中，$S_u$ 是用户 $u$ 喜欢过的物品集合，$N_j$ 是与物品 $j$ 最相似的物品集合。

**优点：**
*   **可扩展性更好**：物品的数量通常比用户数量少且相对稳定，物品相似度矩阵可以离线计算并存储，更新频率较低。
*   **推荐质量通常更高**：物品之间的相似性相对稳定，而用户的兴趣可能随时间变化。
*   **可解释性**：可以向用户解释为什么推荐某个物品（例如：“因为你喜欢A，而A和B很相似”）。

**缺点：**
*   **数据稀疏性**：如果物品的共同评分用户很少，计算出的相似度可能不准确。
*   **冷启动问题**：新物品由于缺乏用户交互数据，难以计算其与其他物品的相似度。
*   **覆盖率问题**：对于冷门物品，可能难以找到相似物品。

## 2.3 协同过滤的优缺点总结

**优点：**
*   **领域无关性**：不需要物品的内容信息，适用于各种类型的物品推荐。
*   **能够发现新颖的推荐**：可以推荐用户之前未曾了解但可能感兴趣的物品，具有惊喜性。
*   **实现相对简单**：基本算法逻辑清晰。

**缺点：**
*   **冷启动 (Cold Start)**：
    *   **用户冷启动**：新用户没有历史行为，无法计算其与其他用户的相似度或预测其偏好。
    *   **物品冷启动**：新物品没有被用户交互过，无法计算其与其他物品的相似度或被推荐。
*   **数据稀疏性 (Data Sparsity)**：在大型系统中，用户-物品交互矩阵通常非常稀疏（大部分用户只与少量物品有交互），这使得相似度计算不准确，推荐效果下降。
*   **可扩展性 (Scalability)**：随着用户和物品数量的增加，User-based CF 的计算量会变得非常大。Item-based CF 在这方面表现更好，但当物品数量也极大时，也会面临挑战。
*   **同质化推荐**：倾向于推荐热门物品，可能导致“马太效应”，热门的越来越热，冷门的无人问津。
*   **缺乏可解释性**（相对于基于内容的推荐）：虽然Item-based CF可以提供一定的解释，但总体上不如基于内容的推荐直观。

## 2.4 协同过滤的实现案例（伪代码）

### 2.4.1 User-based CF 伪代码

```python
function UserBasedCF(target_user, user_item_matrix, num_similar_users, num_recommendations):
    similarities = {}
    for other_user in user_item_matrix.users:
        if other_user == target_user:
            continue
        # 计算 target_user 和 other_user 的相似度 (e.g., Pearson)
        sim = calculate_similarity(user_item_matrix, target_user, other_user)
        similarities[other_user] = sim

    # 排序找到最相似的 N 个用户
    sorted_similar_users = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_n_similar_users = sorted_similar_users[:num_similar_users]

    recommendations = {}
    target_user_items = user_item_matrix.get_items_by_user(target_user)

    for similar_user, similarity_score in top_n_similar_users:
        if similarity_score <= 0: # 通常忽略负相关或不相关的用户
            continue
        similar_user_items = user_item_matrix.get_items_by_user(similar_user)
        for item in similar_user_items:
            if item not in target_user_items: # 推荐目标用户未接触过的物品
                if item not in recommendations:
                    recommendations[item] = {'weighted_score_sum': 0, 'similarity_sum': 0}
                
                # 预测评分 (加权平均)
                # 假设 r_vj 是 similar_user 对 item 的评分, r_v_bar 是 similar_user 的平均分
                # rating_of_similar_user_for_item = user_item_matrix.get_rating(similar_user, item)
                # avg_rating_similar_user = user_item_matrix.get_avg_rating(similar_user)
                # recommendations[item]['weighted_score_sum'] += similarity_score * (rating_of_similar_user_for_item - avg_rating_similar_user)
                # recommendations[item]['similarity_sum'] += abs(similarity_score)
                
                # 简化版：直接用相似度加权评分 (假设评分为1，代表喜欢)
                recommendations[item]['weighted_score_sum'] += similarity_score * 1 # 假设交互即为1分
                recommendations[item]['similarity_sum'] += abs(similarity_score)

    final_recommendations = {}
    # avg_rating_target_user = user_item_matrix.get_avg_rating(target_user)
    for item, scores in recommendations.items():
        if scores['similarity_sum'] > 0:
            # predicted_rating = avg_rating_target_user + scores['weighted_score_sum'] / scores['similarity_sum']
            predicted_rating = scores['weighted_score_sum'] / scores['similarity_sum'] # 简化版
            final_recommendations[item] = predicted_rating
    
    # 排序并返回Top-K推荐
    sorted_recommendations = sorted(final_recommendations.items(), key=lambda item: item[1], reverse=True)
    return sorted_recommendations[:num_recommendations]
```

### 2.4.2 Item-based CF 伪代码

```python
function ItemBasedCF(target_user, user_item_matrix, item_similarity_matrix, num_recommendations):
    target_user_rated_items = user_item_matrix.get_items_interacted_by_user(target_user)
    
    recommendations = {}
    
    for item_to_predict in user_item_matrix.get_all_items():
        if item_to_predict in target_user_rated_items: # 不推荐用户已经交互过的物品
            continue
            
        weighted_score_sum = 0
        similarity_sum = 0
        
        # 遍历用户交互过的物品，找到与待预测物品相似的物品
        for user_rated_item, user_rating_for_item in target_user_rated_items.items():
            # 从预先计算好的物品相似度矩阵中获取相似度
            similarity = item_similarity_matrix.get_similarity(item_to_predict, user_rated_item)
            
            if similarity > 0: # 只考虑正相关的物品
                weighted_score_sum += similarity * user_rating_for_item # user_rating_for_item 是用户对已知物品的评分
                similarity_sum += abs(similarity)
        
        if similarity_sum > 0:
            predicted_rating = weighted_score_sum / similarity_sum
            recommendations[item_to_predict] = predicted_rating
            
    # 排序并返回Top-K推荐
    sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
    return sorted_recommendations[:num_recommendations]

# 预计算物品相似度矩阵
function PrecomputeItemSimilarity(user_item_matrix):
    item_similarity_matrix = {}
    all_items = user_item_matrix.get_all_items()
    for item1 in all_items:
        item_similarity_matrix[item1] = {}
        for item2 in all_items:
            if item1 == item2:
                item_similarity_matrix[item1][item2] = 1.0
            elif item2 in item_similarity_matrix and item1 in item_similarity_matrix[item2]: # 已计算过
                 item_similarity_matrix[item1][item2] = item_similarity_matrix[item2][item1]
            else:
                # 计算 item1 和 item2 的相似度 (e.g., Adjusted Cosine Similarity)
                sim = calculate_item_similarity(user_item_matrix, item1, item2)
                item_similarity_matrix[item1][item2] = sim
    return item_similarity_matrix

```

**注意**：上述伪代码仅为概念演示，实际实现中需要处理数据加载、稀疏矩阵表示、相似度计算的具体细节、以及性能优化等问题。

协同过滤是许多更高级推荐算法的基础，理解其原理和优缺点对于学习后续的推荐技术至关重要。