# DIT Architecture

## 0. 输入和预处理
Embedding (The Setup)在进入 DiTBlock 之前，需要把原始图像和条件信号转换成同样的维度 $D$。

* Image Embedding (Patchify):  
像 ViT 一样，将图像（或 Latent）切成 $P \times P$ 的小块（Patches），然后通过一个 Linear Projection 映射到维度 $D$。 位置编码： 加上固定的 2D Sine-Cosine Positional Embedding。
1. Intake: $(B, C, H, W)$ 
2. Output: $x \in (B, L, D)$，其中 $L = (H/P) \times (W/P)$。

* Condition Embedding: 
将时间步 $t$ 用 MLP 转换成 Embedding。将类别标签 $y$ 用 Embedding Layer 转换。融合： 两者相加得到全局条件向量。
1. Intake: $t$ (scalar) 和 $y$ (class index)。
2. Output: $c \in (B, D)$。

## 1. DIT Block
Diffusion Transformer 核心组件DiTBlock 是 Diffusion Transformer 的基本计算单元，它将传统的 Transformer 结构与 Adaptive Layer Norm (adaLN-Zero) 机制结合，实现对生成过程的精细条件控制。

1. 模块输入 (Intake)$x$ (Hidden States): Tensor $[B, L, D]$，表示图像 Patch 的序列特征。$c$ (Conditioning): Tensor $[B, D]$，包含时间步 $t$ 和类别标签的全局嵌入。

2. 数学逻辑 (Mathematical Flow)DiTBlock 遵循 Pre-Norm 结构，并在每个残差支路引入了基于条件的调制（Modulation）和门控（Gating）。Step 1: 调制参数生成利用线性投影从条件信号 $c$ 中生成 6 个缩放与移位参数：$$(\beta_1, \gamma_1, \alpha_1, \beta_2, \gamma_2, \alpha_2) = \text{MLP}(c)$$Step 2: 自注意力层 (Self-Attention)对输入进行归一化和线性变换（Modulate）后计算注意力，并由 $\alpha_1$ 控制残差强度：$$\hat{x}_1 = \text{LayerNorm}(x) \cdot (1 + \gamma_1) + \beta_1$$ $$x \leftarrow x + \alpha_1 \cdot \text{Attention}(\hat{x}_1)$$ Step 3: 前馈网络层 (MLP)同理，对特征进行二次调制后通过 MLP 层：$$\hat{x}_2 = \text{LayerNorm}(x) \cdot (1 + \gamma_2) + \beta_2$$ $$x \leftarrow x + \alpha_2 \cdot \text{MLP}(\hat{x}_2)$$

3. 输出 (Output)$x'$: Tensor $[B, L, D]$。形状不变，但特征已根据条件信号完成了全局空间交互。

4. 关键设计：adaLN-Zero初始化优势： 线性层初始化为全 0 时，门控参数 $\alpha \rightarrow 0$。这意味着在训练初期，每个 DiTBlock 表现为恒等映射（Identity Function），极大地稳定了深层扩散模型的初始训练过程。动态控制： 模型通过 $\beta$ 和 $\gamma$ 在每一个 Block 动态调整特征分布，使网络能根据去噪进度（时间步 $t$）调整关注点。


## 2. DIT Final layer 
Intake: 最后一层 Block 的输出 $x (B, L, D)$。
1.  通过 FinalLayer 进行最后一次基于 $c$ 的调制。
2.  用线性层将 $D$ 映射回 $P \times P \times \text{out\_dim}$。
Output: $y \in (B, L, P^2 \cdot C)$。
Unpatchify: 把 $L$ 个小方块重新拼回原图像尺寸 $(B, C, H, W)$，得到预测的噪声。