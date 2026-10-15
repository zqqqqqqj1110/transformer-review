# DeepSeek 与原版 Transformer 的不同

> 参照物：你手写的原版《Attention Is All You Need》(2017) encoder-decoder：Post-LN + LayerNorm、sin/cos 绝对位置编码、ReLU FFN、标准 MHA。
> 对比对象：**DeepSeek-V3**（架构，V2 基本相同）和 **DeepSeek-R1**（在 V3 上做的训练）。建议先读 [qwen-vs-transformer.md](qwen-vs-transformer.md)，下面很多"现代标配"不再重复展开。

---

## 一句话总结

DeepSeek-V3 也是 **decoder-only** 的 Transformer，同样用 Pre-Norm + RMSNorm、RoPE、SwiGLU。
它跟 Qwen / LLaMA 真正不一样的是三个新机制：
1. **MLA（多头潜在注意力）**：把 K、V 压成一个低维向量来缓存，KV Cache 极小；
2. **DeepSeekMoE**：细粒度专家 + 共享专家，外加**无辅助损失的负载均衡**；
3. **MTP（多 token 预测）**：训练时一次多预测一个 token。

R1 在架构上和 V3 一模一样，区别全在训练：用 **GRPO 强化学习**训出长链推理能力。

---

## 逐项对比

| # | 部位 | 原版 Transformer | DeepSeek-V3 | 为什么改 |
|---|---|---|---|---|
| 1 | 整体结构 | Encoder + Decoder，有 cross-attention | **只有 Decoder**，只用 causal mask | 同 Qwen：自回归生成 |
| 2 | 归一化 | Post-LN + LayerNorm | **Pre-Norm + RMSNorm**；MLA 压缩后的低维向量上还各加了一个 RMSNorm | 深层训练稳定 |
| 3 | 位置编码 | sin/cos 加到 embedding 上 | **RoPE**，但只作用在一个单独拆出来的小分量上（"解耦 RoPE"，见 MLA）；用 YaRN 从 4K 扩到 128K | RoPE 会破坏 MLA 的矩阵吸收，所以要拆开 |
| 4 | 注意力 | 标准 MHA，缓存完整的 K、V | **MLA**：每个 token 只缓存一个 512 维的潜在向量 + 64 维的 RoPE key | KV Cache 比同规模 MHA 小几十倍，效果不降反升（GQA/MQA 省显存是靠砍头数，会掉点） |
| 5 | FFN | 每层一个 ReLU FFN | 前 3 层是普通 SwiGLU FFN；**其余 58 层是 MoE**：1 个共享专家 + 256 个路由专家，每 token 选 8 个 | 总参数 671B，每 token 只激活 37B，算力省但容量大 |
| 6 | 专家路由 | 无 | 用 **sigmoid** 打亲和度分数，top-8 后再归一化；每个 token 最多发到 4 台机器 | sigmoid 比 softmax 在专家多时更稳；限机器数降低通信 |
| 7 | 负载均衡 | 无 | **无辅助损失**：给每个专家的分数加一个偏置 b，只用于选 top-k、不进门控权重；某专家过载就把 b 调低，空闲就调高；另加一个很小的序列级均衡损失兜底 | 传统辅助损失会和主任务抢梯度、伤效果 |
| 8 | 训练目标 | 预测下一个 token | **next-token + MTP**：额外的模块串行地预测第 2 个 token，共享 embedding 和输出头 | 训练信号更密；推理时 MTP 模块还能做投机解码，加速约 1.8 倍 |
| 9 | Bias / Dropout | 有 bias，dropout 0.1 | 无 bias，不用 dropout | 同 Qwen |
| 10 | 词表 / Embedding | 37K BPE，输入输出共享，乘 √d | 约 128K 字节级 BPE，输入输出**不共享** | — |
| 11 | 规模 | d=512，6+6 层，8 头 | d=7168，**61 层，128 头**，每头 128 维；总参数 671B，激活 37B | — |
| 12 | 训练工程 | 单机 8 卡 P100 | **FP8 混合精度**训练、DualPipe 流水线并行、14.8T token；总共约 278.8 万 H800 GPU 小时 | 首个在这个规模上验证 FP8 训练可行的开源模型 |

> 新一点的 **V3.2** 还加了 **DSA（DeepSeek 稀疏注意力）**：用一个轻量"索引器"给历史 token 打分，每个 query 只对 top-k 个 token 做注意力，长文本时把 O(n²) 降下来。面试提一句即可。

---

## 重点一：MLA 到底怎么省 KV Cache

**原版 MHA**：每层每个 token 要缓存 K、V，共 `2 × 头数 × 每头维度` 个数。V3 规模下是 `2 × 128 × 128 = 32768`。

**MLA 的做法**：

1. 把输入 h 先**压缩**成一个低维潜在向量：`c_KV = W_DKV · h`（512 维）。**只缓存它。**
2. 用的时候再**解压**：`k_C = W_UK · c_KV`，`v = W_UV · c_KV`，恢复成 128 个头的 K、V。
3. **矩阵吸收**：推理时 `qᵀ k_C = qᵀ W_UK c_KV = (W_UKᵀ q)ᵀ c_KV`，可以把 W_UK 并到 Q 那一侧，W_UV 并到输出投影 W_O 里，于是根本不用真把 K、V 解压出来，直接拿 c_KV 算注意力。
4. **解耦 RoPE**：RoPE 是个跟位置有关的旋转矩阵，夹在 q 和 W_UK 中间，就没法把 W_UK 提前吸收掉了。所以 DeepSeek 把 key 拆成两部分：
   - `k_C`：从 c_KV 解压出来的部分，**不加位置编码**，可以吸收；
   - `k_R = RoPE(W_KR · h)`：一个单独的 64 维小向量，**所有头共享**，专门带位置信息，也要缓存。
   - 最终 `k = [k_C ; k_R]`，`q = [q_C ; q_R]` 拼接起来做点积。
5. Q 也做了低秩压缩（1536 维），这个不是为了缓存，是为了训练时省激活显存。

**结果**：每层每 token 只缓存 `512 + 64 = 576` 个数，是 MHA 的约 **1/57**；比 GQA 还省，而且论文里效果比 MHA 还好。

面试一句话：**MLA 用低秩联合压缩代替缓存完整 KV，靠矩阵吸收避免解压开销，用解耦的共享 RoPE key 解决位置编码和吸收的冲突。**

---

## 重点二：DeepSeekMoE 和普通 MoE 的区别

| | 传统 MoE（如 GShard、Mixtral） | DeepSeekMoE |
|---|---|---|
| 专家粒度 | 少量大专家（Mixtral：8 选 2） | **细粒度**：把专家切小、数量变多（256 选 8），组合方式多得多，专家更专精 |
| 共享专家 | 无 | **1 个共享专家**所有 token 都走，学通用知识，路由专家就不用重复学 |
| 门控 | softmax | sigmoid，选完 top-k 再归一化 |
| 负载均衡 | 辅助损失 | **偏置调节，无辅助损失**（+ 极小的序列级损失） |

> 和 Qwen 对比：Qwen2-MoE 也有共享专家，Qwen3-MoE 去掉了共享专家；DeepSeek-V3 保留。

---

## 重点三：R1 的训练（架构不变）

| 阶段 | 做了什么 |
|---|---|
| **R1-Zero** | 直接在 V3-Base 上做纯 RL，不做 SFT。奖励是**规则奖励**：答案对不对（数学题对答案、代码跑测试）+ 格式对不对（思考写在 `<think>` 里）。模型自己学会了变长思考、反思（"aha moment"）。缺点：可读性差、中英混杂 |
| **R1** | ① 少量高质量长 CoT 数据做冷启动 SFT → ② 面向推理的 RL（加语言一致性奖励）→ ③ 拒绝采样生成约 60 万推理数据 + 20 万通用数据再 SFT → ④ 覆盖所有场景的 RL（推理用规则奖励，通用用奖励模型） |
| **蒸馏** | 用 R1 生成的数据直接 SFT 小模型（Qwen、LLaMA），小模型推理能力大涨 |

**GRPO（组相对策略优化）** vs PPO：
- PPO 需要一个和策略模型一样大的 **critic（价值模型）**来估计优势；
- GRPO **去掉 critic**：同一个问题采样一组（比如 16 个）回答，各自打分，优势 = `(r_i − 组内均值) / 组内标准差`；
- 再加上对参考模型的 KL 约束。省掉一半显存，适合规则奖励。

---

## 没变的东西

- Scaled dot-product attention（`softmax(QKᵀ/√d) V`，只是 Q、K 的维度是 128+64）
- 多头 → 拼接 → 输出投影
- 残差连接；每层 = 注意力子层 + FFN（或 MoE）子层
- 输出层到词表 + softmax

---

## 对照代码：MLA 和 MoE（简化版）

省略了 KV Cache、矩阵吸收和 mask 细节，只为了看清结构。`RMSNorm`、`SwiGLU`、`apply_rope` 同 Qwen 笔记。

```python
class MLA(nn.Module):
    def __init__(self, d=7168, n_heads=128, d_head=128, d_rope=64, d_c=512, d_cq=1536):
        super().__init__()
        self.h, self.dh, self.dr = n_heads, d_head, d_rope
        # Q：先压缩再解压（省训练激活）
        self.w_dq  = nn.Linear(d, d_cq, bias=False); self.q_norm = RMSNorm(d_cq)
        self.w_uq  = nn.Linear(d_cq, n_heads * d_head, bias=False)        # q_C：不带位置
        self.w_qr  = nn.Linear(d_cq, n_heads * d_rope, bias=False)        # q_R：每头一份，带 RoPE
        # KV：联合压缩成 c_KV —— 推理时只缓存它
        self.w_dkv = nn.Linear(d, d_c, bias=False); self.kv_norm = RMSNorm(d_c)
        self.w_uk  = nn.Linear(d_c, n_heads * d_head, bias=False)         # k_C
        self.w_uv  = nn.Linear(d_c, n_heads * d_head, bias=False)         # v
        self.w_kr  = nn.Linear(d, d_rope, bias=False)                     # k_R：所有头共享，也要缓存
        self.w_o   = nn.Linear(n_heads * d_head, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        c_q  = self.q_norm(self.w_dq(x))
        q_c  = self.w_uq(c_q).view(B, T, self.h, self.dh)
        q_r  = apply_rope(self.w_qr(c_q).view(B, T, self.h, self.dr), cos, sin)

        c_kv = self.kv_norm(self.w_dkv(x))                                # (B, T, 512)  ← 缓存
        k_r  = apply_rope(self.w_kr(x).view(B, T, 1, self.dr), cos, sin)  # (B, T, 1, 64) ← 缓存
        k_c  = self.w_uk(c_kv).view(B, T, self.h, self.dh)
        v    = self.w_uv(c_kv).view(B, T, self.h, self.dh)

        q = torch.cat([q_c, q_r], -1)                                     # 每头 128 + 64 维
        k = torch.cat([k_c, k_r.expand(-1, -1, self.h, -1)], -1)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)    # 缩放用 √(128+64)
        return self.w_o(out.transpose(1, 2).reshape(B, T, -1))

class DeepSeekMoE(nn.Module):
    def __init__(self, d=7168, d_expert=2048, n_routed=256, n_shared=1, top_k=8):
        super().__init__()
        self.shared  = nn.ModuleList(SwiGLU(d, d_expert) for _ in range(n_shared))
        self.experts = nn.ModuleList(SwiGLU(d, d_expert) for _ in range(n_routed))
        self.gate    = nn.Linear(d, n_routed, bias=False)
        self.register_buffer("bias", torch.zeros(n_routed))   # 负载均衡偏置，不靠梯度更新
        self.top_k = top_k

    def forward(self, x):                                   # x: (N, d)，已展平成 token
        scores = torch.sigmoid(self.gate(x))                # sigmoid，不是 softmax
        _, idx = (scores + self.bias).topk(self.top_k, -1)  # 偏置只影响"选谁"
        w = scores.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)                     # 门控权重用原始分数归一化
        out = sum(e(x) for e in self.shared)                # 共享专家：所有 token 都走
        for j in range(self.top_k):                         # 朴素实现，真实代码按专家分组批量算
            for e_id in idx[:, j].unique():
                m = idx[:, j] == e_id
                out[m] += w[m, j:j+1] * self.experts[e_id](x[m])
        return out
    # 训练中每步后：过载专家 bias -= γ，空闲专家 bias += γ

class DeepSeekBlock(nn.Module):
    def __init__(self, d, layer_id):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = MLA(d)
        self.ffn  = SwiGLU(d, 18432) if layer_id < 3 else DeepSeekMoE(d)   # 前 3 层是 dense
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)   # Pre-Norm
        return x + self.ffn(self.norm2(x))
```

---

## 面试速答版（30 秒）

> DeepSeek-V3 是 decoder-only 的 MoE Transformer，基础组件和 LLaMA/Qwen 一样，都是 Pre-Norm + RMSNorm、RoPE、SwiGLU。三个关键创新：一是 MLA，把 K、V 低秩联合压缩成 512 维潜在向量来缓存，推理时通过矩阵吸收直接用潜在向量算注意力，再用一个解耦的共享 RoPE key 带位置信息，KV Cache 只有 MHA 的约 1/57；二是 DeepSeekMoE，细粒度的 256 个路由专家选 8 个，加 1 个共享专家，用 sigmoid 门控和动态偏置做无辅助损失的负载均衡；三是 MTP，多预测一个 token，训练信号更密，推理还能投机解码。工程上用了 FP8 训练。R1 架构不变，用 GRPO 做强化学习，GRPO 去掉 critic，用组内回答奖励的均值和标准差算优势。

---

## 参考

- DeepSeek-V2 论文（MLA、DeepSeekMoE 首次提出）：arXiv 2405.04434
- DeepSeek-V3 技术报告：arXiv 2412.19437
- DeepSeek-R1 论文：arXiv 2501.12948
- DeepSeekMath（GRPO 首次提出）：arXiv 2402.03300
