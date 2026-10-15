# Qwen 与原版 Transformer 的不同

> 参照物：你手写的原版《Attention Is All You Need》(2017) encoder-decoder：Post-LN + LayerNorm、sin/cos 绝对位置编码、ReLU FFN、标准 MHA。
> 对比对象：Qwen2 / Qwen2.5 / Qwen3（dense 为主，MoE 单列）。数字以 Qwen2.5-7B 为例。

---

## 一句话总结

Qwen 本质上是 **decoder-only 的 Transformer**，骨架（残差、多头注意力、FFN、堆叠 N 层）没变，变的是几个"零件"：
**Pre-Norm + RMSNorm、RoPE、SwiGLU、GQA**，再加上 Qwen3 的 **QK-Norm**。这套组合基本就是 LLaMA 以来的"现代 LLM 标配"。

---

## 逐项对比

| # | 部位 | 原版 Transformer | Qwen2 / 2.5 / 3 | 为什么改 |
|---|---|---|---|---|
| 1 | 整体结构 | Encoder + Decoder，decoder 里有 cross-attention | **只有 Decoder**，没有 encoder，也没有 cross-attention | 生成式语言模型只需"根据前文预测下一个 token"，decoder-only 结构简单、易扩展 |
| 2 | 归一化位置 | **Post-LN**：`x = LN(x + Sublayer(x))` | **Pre-Norm**：`x = x + Sublayer(Norm(x))`，最后一层后再加一个 final norm | Post-LN 深层时梯度不稳定、离不开 warmup；Pre-Norm 让残差通路是"干净的恒等映射"，几十上百层也好训 |
| 3 | 归一化方式 | LayerNorm（减均值、除标准差、有 γ 和 β） | **RMSNorm**：只除以均方根，只有 γ，没有 β | 效果相当，少算一个均值，更快 |
| 4 | 位置编码 | sin/cos **绝对**位置编码，**加**到 embedding 上，只在输入处一次 | **RoPE 旋转位置编码**：每一层对 Q、K 按位置做旋转，V 不动；不往 embedding 上加任何东西 | 旋转后 `q·k` 只依赖相对位置差；外推长文本更好。Qwen2.5/3 的 `rope_theta = 1,000,000`（原 RoPE 是 10000），配合 YaRN 扩到 128K 上下文 |
| 5 | FFN | `W2 · ReLU(W1 x)`，2 个矩阵，中间维 4×d | **SwiGLU**：`down( SiLU(gate x) ⊙ up x )`，3 个矩阵，中间维约 5.3×d（7B：3584 → 18944） | 门控 + 平滑激活效果更好；三个矩阵所以中间维缩到约 8/3×4d 的量级保持参数量 |
| 6 | 注意力头 | 标准 MHA：Q、K、V 头数相同（8 头） | **GQA（分组查询注意力）**：多个 Q 头共享一组 K/V。7B 是 28 个 Q 头、只有 4 个 KV 头 | 推理时 KV Cache 缩小到 1/7，显存和带宽大幅下降，效果几乎不掉 |
| 7 | Bias | Linear 层都有 bias | Qwen2/2.5：**只有 Q、K、V 投影有 bias**，其余全无 bias；Qwen3：**连 QKV bias 也去掉** | 去 bias 训练更稳、省参数；Qwen2 保留 QKV bias 被认为有利于长度外推 |
| 8 | QK-Norm | 无 | **Qwen3 新增**：对每个头的 q、k 再做一次 RMSNorm，然后再上 RoPE | 防止注意力 logits 过大，训练更稳定（替代了 QKV bias 的作用） |
| 9 | 注意力 mask | encoder 用 padding mask，decoder 用 causal mask | 只有 **causal mask**（因果遮挡） | decoder-only 自回归 |
| 10 | Embedding 缩放 / 共享 | embedding 乘 √d_model；输入 embedding 与输出投影共享权重 | 不乘 √d；**小模型共享**（Qwen2.5 的 0.5B/1.5B/3B、Qwen3 的 0.6B–4B），**大模型不共享** | 词表 15 万时 embedding 占小模型参数大头，共享省参数；大模型不在乎，分开效果更好 |
| 11 | 分词 / 词表 | BPE，约 37K 共享词表（英德） | **字节级 BPE**，约 **151K** 词表，覆盖多语言和代码 | 不会出现 OOV；中文、代码压缩率更高 |
| 12 | Dropout | 残差、attention 处都有 dropout 0.1 | 基本 **不用 dropout** | 万亿 token 预训练几乎不会过拟合，dropout 反而浪费 |
| 13 | 规模 | base：d=512，6+6 层，8 头，FFN 2048 | 7B：d=3584，28 层，28/4 头，FFN 18944；上下文 32K–128K | — |
| 14 | 训练目标 | 有监督机器翻译，label smoothing | 海量文本 **next-token prediction 预训练**（Qwen3 约 36T token）→ SFT → RLHF/GRPO 等对齐；Qwen3 还有思考/非思考双模式 | 从"任务模型"变成"通用基座 + 对齐" |

---

## MoE 版本（Qwen2-57B-A14B、Qwen3-30B-A3B / 235B-A22B）

在上面 dense 结构的基础上，**把每层的 FFN 换成一组专家（多个 SwiGLU FFN）+ 一个路由器**：
- 每个 token 由 router 打分，只激活 top-k 个专家（Qwen3-MoE：128 个专家选 8 个）。
- Qwen2-MoE 带"共享专家"（所有 token 都走），Qwen3-MoE 去掉了共享专家，加了全局 batch 级负载均衡损失。
- 总参数大、每 token 计算量小：235B 总参数，每 token 只激活约 22B。

注意力部分和 dense 一样，不变。

---

## 没变的东西（面试时也要说出来）

- Scaled dot-product attention：`softmax(QKᵀ/√d_k) V`
- 多头拆分 → 各自注意力 → 拼接 → 输出投影 `o_proj`
- 残差连接
- 每层 = 注意力子层 + FFN 子层，N 层堆叠
- 输出 = 线性层到词表 + softmax

---

## 对照代码：一个 Qwen 风格的 decoder 层

拿你手写的原版来对照，改动点都标了注释（省略了 KV Cache 和 mask 细节）：

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))   # 只有 γ，没有 β
        self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))   # 原版：W2(ReLU(W1 x))

class Attention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, qk_norm=False):
        super().__init__()
        self.h, self.kv_h, self.hd = n_heads, n_kv_heads, d // n_heads
        bias = not qk_norm                                     # Qwen2/2.5 有 QKV bias，Qwen3 没有
        self.q = nn.Linear(d, n_heads * self.hd, bias=bias)
        self.k = nn.Linear(d, n_kv_heads * self.hd, bias=bias) # GQA：K/V 头更少
        self.v = nn.Linear(d, n_kv_heads * self.hd, bias=bias)
        self.o = nn.Linear(n_heads * self.hd, d, bias=False)
        self.q_norm = RMSNorm(self.hd) if qk_norm else nn.Identity()   # Qwen3 的 QK-Norm
        self.k_norm = RMSNorm(self.hd) if qk_norm else nn.Identity()

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_norm(self.q(x).view(B, T, self.h,    self.hd)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(B, T, self.kv_h, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv_h, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)       # RoPE 只作用于 Q、K
        rep = self.h // self.kv_h                                      # GQA：把 KV 头复制给每组 Q 头
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # 只有 causal mask
        return self.o(out.transpose(1, 2).reshape(B, T, -1))

class QwenBlock(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, d_ff, qk_norm=False):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_heads, n_kv_heads, qk_norm)
        self.mlp  = SwiGLU(d, d_ff)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)   # Pre-Norm；原版是 LN(x + attn(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

整个模型 = `embedding → N × QwenBlock → final RMSNorm → lm_head`，没有加位置编码这一步（位置信息全在每层的 RoPE 里）。

---

## 面试速答版（30 秒）

> Qwen 是 decoder-only 的 Transformer。相比原版：用 Pre-Norm + RMSNorm 替代 Post-LN + LayerNorm，训练更稳；用 RoPE 替代正弦绝对位置编码，表达相对位置、支持长上下文外推；FFN 用 SwiGLU 替代 ReLU；注意力用 GQA 减小 KV Cache；Qwen2 只在 QKV 上保留 bias，Qwen3 去掉 bias 改用 QK-Norm；词表是 15 万的字节级 BPE；还有 MoE 版本把 FFN 换成路由专家。训练上是海量文本自回归预训练再做 SFT 和 RLHF。
