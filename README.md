# Learning to Hash for Efficient Search over Incomplete Knowledge Graphs

### 1. 核心模型定义 (Core Models)

这两份文件是整个论文框架最核心的体现，分别对应原始的连续向量查询模型和论文提出的哈希查询模型。

- `netquery/model.py`：
  - 含义：定义了基础的连续型查询编码-解码器模型 `QueryEncoderDecoder`。
  - 作用：基于连续向量空间，处理图谱中的各类复杂查询（如 `1-chain` 到 `3-chain` 的路径查询，以及 `2-inter`、`3-inter_chain` 等带交集的逻辑查询）。它通过调用底层的 `encoder` 提取节点特征，再通过 `decoder`（路径或交集）计算查询嵌入与目标实体之间的余弦相似度，并使用 `margin_loss` 进行模型优化。
- `netquery/hashed_model.py`：
  - 含义：定义了哈希化的查询编码-解码器模型 `HashedQueryEncoderDecoder`（即论文提出的 HashKG 核心）。
  - 作用：在基础模型的基础上引入了 `tohash(x)` 方法，通过符号函数 `torch.sign()` 将连续的向量强制转换为二值（离散的哈希码）。在进行路径推理和交集运算时，强制开启了哈希标志（`sign=True`），使所有复杂的逻辑查询和实体匹配都在汉明空间内以极高的效率完成。

### 2. 神经网络组件层 (Neural Network Components)

这些文件负责知识图谱特征提取和操作的具体实现，并在论文“连续逼近离散”的训练技巧中发挥了关键作用。

- `netquery/encoders.py`：
  - 含义：包含将节点（Node）转换为嵌入向量（Embedding）的编码器，如 `DirectEncoder` 和基于 GCN 的 `Encoder`。
  - 作用：为了能够训练离散的哈希编码，论文采用了一种渐进式逼近策略（Relaxation）：利用 $\tanh(\beta x)$ 来逼近符号函数 $sgn(x)$。此文件中的 `forward` 前向传播显式地应用了 `torch.tanh(self.beta * embeds)`，为后续映射到汉明空间打下基础。
- `netquery/decoders.py`：
  - 含义：包含预测实体间关系得分的解码器。
  - 作用
    ：实现了多种关系和逻辑的解码操作，主要包括：
    - `BilinearMetapathDecoder` 和 `TransEMetapathDecoder`：用于路径推断（元路径），处理如 A $\rightarrow$ B $\rightarrow$ C 的多跳链式查询。
    - `SetIntersection`：用于交集逻辑操作，处理多个查询条件交集的汇聚问题。
    - 同样地，这些解码器内部都结合了 $\tanh(\beta x)$ 以及哈希化的 `tohash(x)` 操作以适配 HashKG 的思想。
- `netquery/aggregators.py`：
  - 含义：实现了基于邻居聚合操作的多种类聚合器（如 `MeanAggregator`, `PoolAggregator` 等）。
  - 作用：在使用图神经网络(GNN)作为节点编码器时，用于聚合目标节点周围邻居的特征嵌入，支持 dropout 以防止过拟合。

### 3. 训练与测试模块 (Training & Testing Scripts)

这部分代码负责串联上述组件，并在实际的数据集（如项目中提供的 bio 数据）上进行模型的训练与验证。

- `netquery/bio/train.py`：
  - 含义：针对生物医学(BIO)知识图谱的训练启动脚本。
  - 作用：读取数据并初始化模型。这其中最值得注意的是论文中关于参数 $\beta$ 的训练技巧：脚本支持加载前一个 $\beta-1$ 阶段训练好的权重模型，在此基础上使用更大的 $\beta$ 继续微调（代码中 $\beta$ 逐渐从 $1$ 增大到 $20$），通过逐步拉伸 $\tanh$ 函数的斜率，使得向量极其平滑地逼近二值哈希空间，解决了哈希直接离散化不可导（梯度无法反向传播）的问题。
- `netquery/bio/test.py`：
  - 含义：测试评估脚本。
  - 作用：在测试时，它会同时加载原始的连续型 `QueryEncoderDecoder` 和 `HashedQueryEncoderDecoder` 模型。通过对比这两个模型在复杂查询验证集上的表现，用来证明将其哈希化后，在损失微小精度的情况下大幅降低了计算复杂度和存储开销。
- `netquery/train_helpers.py`：
  - 含义：封装了训练的辅助调度函数。
  - 作用：包含 `run_train`, `run_eval`, `run_batch` 等函数。它控制了不同查询类别（先训练简单的边/`1-chain` 使得基础图嵌入收敛，然后再加入复杂的多跳 `path` 和 `inter` 逻辑查询）的交替学习与提前停止（early-stopping）机制。

### 4. 图数据处理与采样 (Graph & Data Utilities)

这部分主要是支持异构知识图谱在不完备情况下的采样逻辑。

- `netquery/graph.py`：
  - 含义：定义了异构图谱的数据结构 `Graph`，以及对应的查询类 `Query`、公式类 `Formula`。
  - 作用：提供极其复杂的子图/查询采样算法（如 `sample_query_subgraph_bytype`）。为了训练逻辑查询，它需要在不完备知识图谱上通过随机游走，自动生成用于训练的 `2-chain`, `3-chain`, `2-inter`, `3-chain_inter` 等结构的数据，并负责挑选难以区分的负样本（Hard Negative Samples）。
- `netquery/data_utils.py` / `netquery/bio/data_utils.py` / `netquery/utils.py`：
  - 含义：各种数据处理与功能性辅助工具。
  - 作用：包括了序列化图谱数据的加载、多进程并行采样、模型评测指标（如 AUC 曲线下面积：`eval_auc_queries`）、日志记录和 GPU 内存转移等杂项支持。

### 结合论文原文第 III 节（公式）剖析核心代码

论文的 **第三章 (PROPOSED METHOD)** 提出了将图谱嵌入到哈希空间的核心逻辑：用投影操作（Projection, $P$）处理关系推断，用交集操作（Intersection, $I$）处理逻辑汇聚，并利用 $\tanh(\delta x)$ 来平滑地进行离散化训练。

#### 1. 连续平滑逼近离散哈希（对应公式 2 与 公式 9）

- **论文描述：** 真正的哈希空间应当使用符号函数 $\text{sgn}(x)$（公式2）输出 +1 和 -1。但是 $\text{sgn}(x)$ 的导数为 0 无法反向传播，因此论文提出使用 $\lim_{\delta \to \infty} \tanh(\delta x) = \text{sgn}(x)$（公式9）来进行渐进式平滑逼近。
- **代码体现文件：`netquery/bio/train.py` & 各模块**
  - 在代码中，参数 $\delta$ 被写成了 `beta`。
  - `train.py` 中，支持读取上一个阶段 `beta-1` 的模型权重，用更大的 `beta` 继续训练。
  - 各大运算模块的最后，几乎都有类似 `act = torch.tanh(self.beta * act)` 的操作，将原本浮点数的向量慢慢“撑”到接近 +1 或 -1 的边缘。
  - 各大模块还定义了 `tohash` 方法：`torch.sign(torch.sign(x).add(0.1))`，在测试阶段强行切分到离散空间。

#### 2. 查询投影操作 (Projection $P$)（对应公式 3）

- **论文描述：** 对于链式（Chain）查询，论文使用 TransE 的投影操作：$v = P(r, h) = \text{sgn}(h + r)$ （公式3）。
- **代码体现文件：`netquery/decoders.py`**
  - 该文件包含了预测关系走势的解码器。对应类为 `TransEMetapathDecoder` 和 `BilinearMetapathDecoder`。
  - 比如 `TransEMetapathDecoder.project()` 的代码：
    ```python
    def project(self, embeds, rel, sign=False):
        # 对应公式3的 h+r (embeds + self.vecs[rel])
        trans_dist = embeds + self.vecs[rel].unsqueeze(1).expand(self.vecs[rel].size(0), embeds.size(1))
        if sign: # 强制哈希化
            trans_dist = self.tohash(trans_dist)
        return trans_dist
    ```

#### 3. 逻辑交集操作 (Intersection $I$)（对应公式 4）

- **论文描述：** 处理汇聚到同一个尾实体的多分支逻辑，公式为 $v = I(v_1, ..., v_n) = \text{sgn}\big(W_v \Psi(\text{ReLU}(M_{v_i} v_i), \forall i)\big)$ （公式4）。
- **代码体现文件：`netquery/decoders.py`**
  - 对应的类为 `SetIntersection`，负责在多跳分支汇聚时求交集。
  - 代码中的 `forward` 完美刻画了公式：
    ```python
    # 对应 ReLU(M_{vi} * vi)
    temp1 = F.relu(self.pre_mats[mode].mm(embeds1))
    temp2 = F.relu(self.pre_mats[mode].mm(embeds2))
    combined = torch.stack([temp1, temp2])
    # 对应 \Psi 聚合函数（如求平均 mean 或是求最小值 min）
    combined = self.agg_func(combined,dim=0)
    # 对应乘上权重矩阵 W_v
    combined = self.post_mats[mode].mm(combined)
    # 最终加上平滑 tanh(beta * x)
    ret = torch.tanh(self.beta * combined)
    ```

#### 4. 实体特征提取与编码 (Entity Embedding)（对应公式 7）

- **论文描述：** $e = \text{sgn}(M_e \cdot x_e / |x_e|)$ （公式7）。
- **代码体现文件：`netquery/encoders.py`**
  - 对应类为 `DirectEncoder`。
  - 前向传播 `forward` 函数首先获取特征 `self.features(nodes, mode)`（即 $x_e$）。
  - 随后应用松弛化的激活函数 `ret = torch.tanh(self.beta * embeds)` 来输出实体的嵌入表示。

#### 5. 余弦相似度打分计算（对应公式 1）

- **论文描述：** 哈希空间下的汉明距离可以等价替换为余弦相似度：$\text{score}(q, e) = \frac{q \cdot e}{q e}$（公式1）。
- **代码体现文件：`netquery/hashed_model.py**`
  - 对应整个框架的总调度室 `HashedQueryEncoderDecoder`。
  - `__init_`_ 中初始化了余弦相似度计算：`self.cos = nn.CosineSimilarity(dim=0)`。
  - `forward` 中，不论是多跳还是交集查询，最后都会将 `target_embeds` (目标实体) 和 `query_intersection` (计算出的查询表示) 传入 `self.cos()` 进行相似度计算作为两者的得分。

项目代码完全遵循了 **“计算图推理 $\rightarrow$ 连续域上的 $\tanh$ 平滑约束 $\rightarrow$ 强制哈希打分”** 这三步走的设计模式，`model.py` 提供全精度的原始实验对比，`hashed_model.py`、`encoders.py` 和 `decoders.py` 构成了论文提出的核心 **HashKG** 框架。

### 流程

1、Dataset

```
cd learning_to_hash
wget https://snap.stanford.edu/nqe/bio_data.zip
unzip bio_data.zip
huggingface-cli download lxwlxwlxw/TKRL_Reproduction data.zip --repo-type dataset --local-dir .
```

2、Requirement

```
conda create -n hashkg python=3.10 -y
conda activate hashkg
pip install torch scikit-learn numpy scipy
```

3、Pre-train

```
cd learning_to_hash/netquery/bio

for i in {1..19}; do
echo "Starting pretrain with beta = $i"
python [train.py](http://train.py) \
  --data_dir ../../bio_data \
  --lr 0.001 \
  --beta $i \
  --pretrain True
done
```

4、Fine Tune

```
python train.py \
  --data_dir ../../bio_data \
  --lr 0.001 \
  --beta 20 \
  --pretrain False
```

5、Test

```
python test.py \
  --data_dir ../../bio_data \
  --beta 20
```

### 后续训练和测试的命令行指令

在您彻底跑完我们之前讨论的 FB15k 海量采样步骤（生成几十万个多跳逻辑查询 `.pkl` 文件）之后，请先通过命令 `**cd /root/autodl-tmp/learning_to_hash/netquery/fb**` 进入到刚才为您新建的这套网络目录下，然后按需执行以下指令：

#### 方案 A：使用 WHE (加权层次编码) 进行训练和测试

```bash
# 1. 预训练阶段 (Pre-train, Beta 1 -> 19)
for i in {1..19}; do
    echo "Starting WHE pretrain with beta = $i"
    python train.py \
      --data_dir ../../fb15k_data \
      --lr 0.001 \
      --beta $i \
      --pretrain True \
      --type_encoder whe
done

# 2. 微调阶段 (Fine-tune, 加入所有复杂图查询进行深层训练)
python train.py \
  --data_dir ../../fb15k_data \
  --lr 0.001 \
  --beta 20 \
  --pretrain False \
  --type_encoder whe

# 3. 最终测试评估 (Test, 自动输出全精度与Hash化推理的对比AUC)
python test.py \
  --data_dir ../../fb15k_data \
  --beta 20 \
  --type_encoder whe
```

#### 方案 B：使用 RHE (递归层次编码) 进行训练和测试

指令与上方几乎一致，您只需要将 `--type_encoder` 指定为 `rhe` 即可！

```bash
# 1. 预训练阶段
for i in {1..19}; do
    python train.py --data_dir ../../fb15k_data --lr 0.001 --beta $i --pretrain True --type_encoder rhe
done

# 2. 微调阶段
python train.py --data_dir ../../fb15k_data --lr 0.001 --beta 20 --pretrain False --type_encoder rhe

# 3. 最终测试评估
python test.py --data_dir ../../fb15k_data --beta 20 --type_encoder rhe
```

所有底层的矩阵操作均已做了 `torch.bmm` 批处理加速，并在 Tensor 级别处理了 `-1` (空节点 padding) 问题，确保可以在您的 RTX 4090 GPU 上将并行计算效率拉满。当采样完成后，您可以直接复制上述命令开始您的论文复现实验！