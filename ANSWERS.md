# CS336 作业 5：对齐

## 3.4 Experiments

### Problem: `prompting_baselines` — Run OLMo-2-0425-1B on GSM8K

我使用 `allenai/OLMo-2-0425-1B` 对 GSM8K 测试集中的全部 1,319
道题进行了评测。采样参数为：温度 1.0、top-p 1.0（vLLM 默认值）、
最大生成长度 512 tokens、随机种子 336。对于两种 R1 prompt，模型生成
`</answer>` 后停止，并在输出中保留该停止字符串。

#### （a）评测结果

| Prompt | 格式为 1、答案为 1 | 格式为 1、答案为 0 | 格式为 0、答案为 0 |
|---|---:|---:|---:|
| `question_only` | 7（0.53%） | 406（30.78%） | 906（68.69%） |
| `r1_zero` | 1（0.08%） | 712（53.98%） | 606（45.94%） |
| `r1_zero_three_shot` | 271（20.55%） | 1,004（76.12%） | 44（3.34%） |

每种 prompt 的三个类别之和均为 1,319。三种 prompt 的格式遵循率分别为
31.31%、54.06% 和 96.66%。因此，三样例 R1 prompt 无论是在引导模型
遵循指定格式方面，还是在产生正确答案方面，都明显优于另外两种 prompt。

三个类别的定义是：

1. 类别 1：`format_reward = 1`，`answer_reward = 1`，即格式正确、答案正确；
2. 类别 2：`format_reward = 1`，`answer_reward = 0`，即格式正确、答案错误；
3. 类别 3：`format_reward = 0`，`answer_reward = 0`，即格式错误、答案也被
   grader 判为错误。

为了专门检查 parser 假阴性，我从失败记录中筛选出“响应正文出现标准答案数值”
的疑似解析失败样本，再按数据顺序分别人工核对类别 2 和类别 3 的前 10 条。
我不仅检查数值是否出现，还检查了计算过程以及模型最终表达的答案是否确实正确。

| 类别 | 人工检查数 | 实际答案正确但被 grader 判错 |
|---|---:|---:|
| 类别 2：格式正确、答案奖励为 0 | 10 | 6 |
| 类别 3：格式错误、答案奖励为 0 | 10 | 4 |

类别 2 中实际正确的 6 条，其标准答案分别为 70、23、187、623、3 和 25。
例如，标准答案为 23、187 和 623 的输出都完成了正确计算，并在 `<answer>`
中给出了正确结果，但由于 `<answer>` 内还包含解释文字和单位，answer grader
没有正确识别最终数值。

类别 3 中实际正确的 4 条，其标准答案分别为 24、27、30 和 2,125。例如，
标准答案为 27 的水果题和标准答案为 30 的精灵题都计算正确，但输出缺少严格要求
的标签组合；标准答案为 24 的礼品袋题包含多个损坏或错位的标签；标准答案为
2,125 的积木题给出了正确 `<answer>`，但前面的整体格式不满足 grader 的严格
匹配条件。

这说明两类失败中都确实存在“数学答案正确，但由于解析或格式要求而被判错”的
输出。类别 3 中尤其需要注意：`answer_reward = 0` 并不一定表示 grader 已经
验证了数学答案错误，因为格式检查失败后，grader 会直接返回两个 0。上述样本是
为了发现 parser 假阴性而进行的定向检查，因此 `6/10` 和 `4/10` 不能当作全部
失败输出中的无偏比例估计。

代表性示例如下：

1. **三样例 R1 prompt 的正确回答。** 在标准答案为 18 的鸭蛋问题中，
   模型正确计算出 `16 - 3 - 4 = 9` 个鸡蛋，以及 `9 * 2 = 18` 美元，
   最后输出 `<answer> 18 </answer>`。
2. **零样例 R1 prompt 中格式正确但答案错误的回答。** 在标准答案为 460
   的加班工资问题中，模型正确算出了正常工资为 400 美元、加班时薪为
   12 美元，但只计算了一个小时的加班工资，最终返回
   `<answer> ... $412 ... </answer>`。
3. **`question_only` 中格式和答案均错误的回答。** 在标准答案为 20 杯的
   鸡饲料问题中，模型返回 `The answer is: 65`，不仅答案错误，也没有使用
   `\boxed{}`。
4. **三样例 R1 prompt 中格式正确但答案错误的回答。** 在标准答案为 366
   的下载量问题中，模型错误地把“减少 30%”理解为直接减去 30，最终输出
   `<answer> 430 </answer>`。

#### （b）不同 prompt 对模型行为的影响

仅仅向这个基础模型提供问题，并不足以稳定地引导它进行问答。在
`question_only` 条件下，许多生成结果更像是对网页文本或训练语料的噪声续写：
模型会修改原问题、引入无关故事、重复指令、给出多个互相矛盾的 boxed 数值，
或者完全不输出 boxed 答案。因此，1,319 条输出中有 906 条的格式奖励为 0，
最终只有 7 条获得了正确性奖励。

零样例 `r1_zero` prompt 更容易促使模型尝试显式推理，并使用指定的
`<think>`/`<answer>` 结构。然而，它的推理过程通常不完整，或者存在计算错误。
例如，在加班工资问题中，模型已经识别出 Eliza 工作了 5 个加班小时，却只计算了
1 个小时的加班工资。该 prompt 的格式遵循率高于 `question_only`，但最终只有
1 条回答正确。

三个示例显著约束了模型的输出行为。在 `r1_zero_three_shot` 条件下，模型通常会
先给出一段简短的算术推理，再在规定的标签中输出单个答案；只有 44 条输出的格式
奖励为 0。它还正确解答了 271 道题，远多于另外两种零样例 prompt。不过，大部分
格式正确的回答仍然是错误的。常见原因是模型模仿了示例的表面结构，却使用了错误的
运算，例如把减少 30% 错误地处理为减去 30。总体而言，few-shot prompting
显著提高了模型的行为一致性和 GSM8K 准确率，但仅靠格式指令仍不足以让这个 1B
基础模型稳定地完成数学推理。

## 4.1 Deriving on-policy GRPO

### 4.1.5 Baselines

#### Problem: `baseline_calcs` — Compute the variance of the policy gradient estimator

##### （a）不使用 baseline 时的方差

令单个样本对应的 policy gradient 项为

$$
Z_i=r(A_i)\nabla_\theta\log\pi_\theta(A_i).
$$

由于 $p=\sigma(\theta)$，因此

$$
\frac{\partial p}{\partial \theta} = p(1-p).
$$

下面分别讨论 $A_i=1$ 和 $A_i=0$。

当 $A_i=1$ 时，该动作出现的概率为 $p$，奖励为 $r(1)=1$，且

$$
\begin{aligned}
\nabla_\theta\log\pi_\theta(A_i=1)
&=\nabla_\theta\log p\\
&=\frac{1}{p}p(1-p)\\
&=1-p.
\end{aligned}
$$

因此

$$
Z_i=1-p,\qquad \text{with probability }p.
$$

当 $A_i=0$ 时，该动作出现的概率为 $1-p$，奖励为 $r(0)=0$，且

$$
\begin{aligned}
\nabla_\theta\log\pi_\theta(A_i=0)
&=\nabla_\theta\log(1-p)\\
&=-\frac{1}{1-p}p(1-p)\\
&=-p.
\end{aligned}
$$

由于此时奖励为 0，因此

$$
Z_i=0\cdot(-p)=0,\qquad \text{with probability }1-p.
$$

综上，$Z_i$ 的分布为

$$
Z_i=
\begin{cases}
1-p, & \text{with probability }p,\\
0, & \text{with probability }1-p.
\end{cases}
$$

由此可得

$$
\begin{aligned}
\mathbb E[Z_i]
&=p(1-p)+(1-p)\cdot0\\
&=p(1-p).
\end{aligned}
$$

以及

$$
\begin{aligned}
\mathbb E[Z_i^2]
&=p(1-p)^2+(1-p)\cdot0^2\\
&=p(1-p)^2.
\end{aligned}
$$

所以单个样本项的方差为

$$
\begin{aligned}
\mathrm{Var}(Z_i)
&=\mathbb E[Z_i^2]-\mathbb E[Z_i]^2\\
&=p(1-p)^2-p^2(1-p)^2\\
&=p(1-p)^3.
\end{aligned}
$$

题目中的 estimator 是 $n$ 个独立同分布样本的均值：

$$
\hat g = \frac{1}{n} \sum_{i=1}^n Z_i.
$$

由于各个 $Z_i$ 相互独立，

$$
\begin{aligned}
\mathrm{Var}(\hat g)
&=\mathrm{Var}\left(\frac{1}{n}\sum_{i=1}^n Z_i\right)\\
&=\frac{1}{n}\mathrm{Var}(Z_i).
\end{aligned}
$$

最终得到

$$
\boxed{\mathrm{Var}(\hat g)=\frac{p(1-p)^3}{n}}.
$$

##### （b）使用常数 baseline $b$ 时的方差

加入 baseline $b$ 后，令单个样本对应的 policy gradient 项为

$$
Z_i=(r(A_i)-b)\nabla_\theta\log\pi_\theta(A_i).
$$

当 $A_i=1$ 时，该动作出现的概率为 $p$，并且

$$
r(1)=1,\qquad \nabla_\theta\log\pi_\theta(A_i=1)=1-p.
$$

因此

$$
Z_i=(1-b)(1-p),\qquad \text{with probability }p.
$$

当 $A_i=0$ 时，该动作出现的概率为 $1-p$，并且

$$
r(0)=0,\qquad \nabla_\theta\log\pi_\theta(A_i=0)=-p.
$$

因此

$$
Z_i=(0-b)(-p)=bp,\qquad \text{with probability }1-p.
$$

所以 $Z_i$ 的分布为

$$
Z_i=
\begin{cases}
(1-b)(1-p), & \text{with probability }p,\\
bp, & \text{with probability }1-p.
\end{cases}
$$

首先计算期望：

$$
\begin{aligned}
\mathbb E[Z_i]
&=p(1-b)(1-p)+(1-p)bp\\
&=p(1-p).
\end{aligned}
$$

可见，引入常数 baseline 并没有改变 policy gradient estimator 的期望。
再计算二阶矩：

$$
\mathbb E[Z_i^2]
=p(1-b)^2(1-p)^2+(1-p)b^2p^2.
$$

因此，单个样本项的方差为

$$
\begin{aligned}
\mathrm{Var}(Z_i)
&=\mathbb E[Z_i^2]-\mathbb E[Z_i]^2\\
&=p(1-b)^2(1-p)^2+(1-p)b^2p^2-p^2(1-p)^2\\
&=p(1-p)\left((1-b)^2(1-p)+b^2p-p(1-p)\right)\\
&=p(1-p)(1-p-b)^2.
\end{aligned}
$$

题目中的 estimator 是 $n$ 个独立同分布样本的平均：

$$
\hat g_b = \frac{1}{n} \sum_{i=1}^n Z_i.
$$

所以

$$
\mathrm{Var}(\hat g_b)=\frac{1}{n}\mathrm{Var}(Z_i).
$$

最终得到

$$
\boxed{
\mathrm{Var}(\hat g_b)
=\frac{p(1-p)(1-p-b)^2}{n}
}.
$$

##### （c）使用 population mean baseline 时的方差

由于奖励函数为 $r(A)=\mathbf 1\{A=1\}$，因此奖励的总体均值为

$$
b=\mathbb E[r(A)]
=1\cdot p+0\cdot(1-p)
=p.
$$

将 $b=p$ 代入第（b）问的结果：

$$
\begin{aligned}
\mathrm{Var}(\hat g_p)
&=\frac{p(1-p)(1-p-b)^2}{n}\\
&=\frac{p(1-p)(1-2p)^2}{n}.
\end{aligned}
$$

不使用 baseline 时，第（a）问得到

$$
\mathrm{Var}(\hat g)
=\frac{p(1-p)^3}{n}.
$$

为了判断 population mean baseline 是否减小方差，比较二者之差：

$$
\begin{aligned}
\mathrm{Var}(\hat g_p)-\mathrm{Var}(\hat g)
&=\frac{p(1-p)}{n}
\left((1-2p)^2-(1-p)^2\right)\\
&=\frac{p(1-p)}{n}
\left(1-4p+4p^2-1+2p-p^2\right)\\
&=\frac{p^2(1-p)(3p-2)}{n}.
\end{aligned}
$$

当 0 < p < 1 时，p²(1−p)/n > 0，所以方差之差的符号只由
3p−2 决定。因此：

- 当 0 < p < ⅔ 时，3p−2 < 0，population mean baseline
  会降低方差；
- 当 p = ⅔ 时，两种 estimator 的方差相等；
- 当 ⅔ < p < 1 时，3p−2 > 0，population mean baseline
  反而会增大方差。

特别地，当 p = ½ 时，

$$
\mathrm{Var}(\hat g_p)
=\frac{p(1-p)(1-2p)^2}{n}
=0.
$$

这是因为第（b）问的最优常数 baseline 是 b* = 1−p；只有在
p = ½ 时，population mean baseline b = p 恰好等于该最优值。

## 4.2 Implementing on-policy GRPO

### 4.2.1 Using Hugging Face models

#### Problem: `tokenize_prompt_and_output` — Prompt and output tokenization

实现见
[`tests/adapters.py` 中的 `run_tokenize_prompt_and_output`](tests/adapters.py)。

该函数首先使用同一个 tokenizer 分别编码 prompt 和 output，并设置
`add_special_tokens=False`。分别编码很重要：如果先拼接原始字符串再
tokenize，分词器可能在 prompt/output 边界处合并 token，而且无法可靠地确定
哪些 token 属于 response。

对于每个样本，函数将两部分 token ID 拼接为

```text
[prompt tokens, response tokens]
```

并构造与完整序列对齐的 mask：

```text
[0, ..., 0, 1, ..., 1]
```

其中 prompt token 对应 0，response token 对应 1。batch 中的序列在右侧使用
`tokenizer.pad_token_id` 补齐到相同长度，padding 对应的 mask 也设为 0。

为了构造 causal language modeling 输入，完整序列随后错开一位：

```python
input_ids = tokens[:, :-1]
labels = tokens[:, 1:]
response_mask = masks[:, 1:]
```

假设未移动的序列和 mask 为

```text
tokens = [P1, P2, P3, R1, R2, R3]
mask   = [ 0,  0,  0,  1,  1,  1]
```

则返回结果为

```text
input_ids    = [P1, P2, P3, R1, R2]
labels       = [P2, P3, R1, R2, R3]
response_mask= [ 0,  0,  1,  1,  1]
```

这里 `response_mask` 使用 `masks[:, 1:]`，因为它必须与 `labels` 对齐：
输入位置 P3 预测的 label 是第一个 response token R1，所以该位置的 mask
应该为 1。

最终三个 tensor 的形状均为

```text
(batch_size, max(prompt_and_output_lens) - 1)
```

其中 `input_ids` 和 `labels` 的 dtype 为 `torch.long`，`response_mask`
的 dtype 为 `torch.bool`。实现通过以下测试：

```bash
uv run pytest -k test_tokenize_prompt_and_output
```

#### Problem: `get_response_log_probs` — Response log-probs (and entropy)

实现见
[`tests/adapters.py` 中的 `run_get_response_log_probs`](tests/adapters.py)。

模型前向传播返回的 logits 形状为

```text
(batch_size, sequence_length, vocab_size)
```

先在词表维度上应用 `log_softmax`：

```python
all_log_probs = torch.log_softmax(logits, dim=-1)
```

这会得到每个位置对词表中所有 token 的条件 log-probability。题目只需要实际
label token 的 log-probability，因此使用 `torch.gather` 在最后一维按照
`labels` 取值：

```python
log_probs = torch.gather(
    all_log_probs,
    dim=-1,
    index=labels.unsqueeze(-1),
).squeeze(-1)
```

`labels.unsqueeze(-1)` 将 labels 的形状从
`(batch_size, sequence_length)` 变为
`(batch_size, sequence_length, 1)`，使其可以在词表维度上作为索引。
取值并移除最后一个大小为 1 的维度后，
`log_probs` 的形状为 `(batch_size, sequence_length)`。令
$\ell_{b,t}$ 表示 `log_probs[b, t]`，$y_{b,t}$ 表示该位置的 label token，则

$$
\ell_{b,t}
=\log p_\theta\!\left(y_{b,t}\mid x_{b,1:t-1}\right).
$$

如果 `return_token_entropy=True`，还要计算每个位置上整个词表分布的 entropy：

$$
H_{b,t}
=-\sum_{v\in\mathcal V}
p_\theta\!\left(v\mid x_{b,1:t-1}\right)
\log p_\theta\!\left(v\mid x_{b,1:t-1}\right).
$$

对应实现为：

```python
all_probs = torch.softmax(logits, dim=-1)
token_entropy = -(all_probs * all_log_probs).sum(dim=-1)
```

这里 `log_probs` 只衡量模型分配给实际 label token 的概率，而
`token_entropy` 衡量模型在整个词表上的不确定程度。当
`return_token_entropy=False` 时，返回的字典应只包含 `"log_probs"`；
为 True 时再加入 `"token_entropy"`。

实现通过以下测试：

```bash
uv run pytest -k test_get_response_log_probs
```

### 4.2.3 GRPO components

#### Problem: `compute_rollout_rewards` — Computing the rewards of rollouts

实现见
[`tests/adapters.py` 中的 `run_compute_rollout_rewards`](tests/adapters.py)。

对于 rollout batch 中的每个 response，函数将它与对应的 ground truth 一起
传给 `reward_fn`：

```python
reward_dicts = [
    reward_fn(response, ground_truth)
    for response, ground_truth in zip(
        rollout_responses,
        repeated_ground_truths,
    )
]
```

每次调用会返回包含总奖励和各奖励分量的字典，例如：

```python
{
    "reward": 1.0,
    "format_reward": 1.0,
    "answer_reward": 1.0,
}
```

训练时需要对所有 rollout 的总 reward 进行 reshape、求组均值和标准差等 tensor
运算，因此提取每个字典中的 `"reward"`，构造一维浮点 tensor：

```python
raw_rewards = torch.tensor(
    [reward["reward"] for reward in reward_dicts],
    dtype=torch.float32,
)
```

它的形状为

```text
(rollout_batch_size,)
```

其中

```text
rollout_batch_size
= n_prompts_per_rollout_batch * group_size
```

也就是说，一个 rollout batch 包含多个 prompt，每个 prompt 生成
`group_size` 个 response。扁平的 `raw_rewards` 可以在下一步 reshape 为

```text
(n_prompts_per_rollout_batch, group_size)
```

从而在每个 prompt 的组内计算 advantage。

其他 reward 分量用于记录训练状态，因此按整个 rollout batch 求均值并放入
metadata：

```python
metadata = {
    "mean_reward": (
        sum(item["reward"] for item in reward_dicts)
        / len(reward_dicts)
    ),
    "mean_format_reward": (
        sum(item["format_reward"] for item in reward_dicts)
        / len(reward_dicts)
    ),
    "mean_answer_reward": (
        sum(item["answer_reward"] for item in reward_dicts)
        / len(reward_dicts)
    ),
}
```

`raw_rewards` 保留每个 response 的训练信号，metadata 则只保存整个 batch 的
汇总统计。实现通过以下测试：

```bash
uv run pytest -k test_compute_rollout_rewards
```

#### Problem: `compute_group_normalized_rewards_grpo` — Group normalization

实现见
[`tests/adapters.py` 中的 `run_compute_group_normalized_rewards`](tests/adapters.py)。

输入的 `raw_rewards` 是长度为 `rollout_batch_size` 的一维 tensor。由于同一个
prompt 的 `group_size` 个 response 连续排列，首先将其恢复成分组形式：

```python
grouped_rewards = raw_rewards.reshape(-1, group_size)
```

其形状为

```text
(n_prompts_per_rollout_batch, group_size)
```

接着沿每一行计算同一个 prompt 内的 reward 均值和标准差：

```python
group_means = grouped_rewards.mean(dim=-1, keepdim=True)
group_stds = grouped_rewards.std(dim=-1, keepdim=True)
```

`keepdim=True` 使二者的形状保持为
`(n_prompts_per_rollout_batch, 1)`，因此 PyTorch 可以将它们广播到每个
group 的所有 response 上。GRPO advantage 为

$$
A_{i,j}=\frac{r_{i,j}-\mu_i}{\sigma_i+\epsilon}.
$$

其中

$$
\mu_i=\frac{1}{G}\sum_{j=1}^{G}r_{i,j},
$$

而实现按照题目要求直接使用默认的 `torch.std`。因此标准差使用
Bessel correction，即

$$
\sigma_i=\sqrt{\frac{1}{G-1}\sum_{j=1}^{G}(r_{i,j}-\mu_i)^2}.
$$

分母加入 `advantage_eps`，可以避免某一组所有 reward 相同时发生除零：

```python
advantages = (
    grouped_rewards - group_means
) / (group_stds + advantage_eps)
```

计算完成后，将 advantages 展平回与输入相同的形状：

```python
advantages = advantages.reshape(rollout_batch_size)
```

例如，当

```python
raw_rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
group_size = 2
```

分组后的 reward 为

```text
[[1, 0],
 [0, 1]]
```

每组均值均为 0.5，默认样本标准差均为

$$
\sqrt{\frac{(1-0.5)^2+(0-0.5)^2}{2-1}}=\sqrt{0.5}.
$$

因此输出约为

```text
[0.70710576, -0.70710576, -0.70710576, 0.70710576]
```

函数还返回 reward 和 advantage 的汇总统计作为 metadata。实现通过以下测试：

```bash
uv run pytest -k test_compute_group_normalized_rewards_grpo
```

## 4.3 Experiments

### Problem: `grpo_experiments_standard_on_policy` — Standard on-policy GRPO on GSM8K

#### （a）训练脚本和实验设置

完整训练入口为 [`scripts/grpo_train.py`](scripts/grpo_train.py)，四个随机种子的启动
脚本为 [`scripts/run-4-seeds.sh`](scripts/run-4-seeds.sh)。训练脚本接收模型、prompt、
训练/验证数据路径、采样参数和训练参数；初始化 OLMo-2-0425-1B、AdamW、vLLM server、
GSM8K 数据和 W&B 后，每一步依次同步权重、执行 on-policy rollout、计算组内归一化
advantage、更新 policy，并定期验证和记录 rollout。

实验使用题目建议的超参数：200 个 rollout steps、学习率 $10^{-5}$、256 responses/batch、
每个 prompt 生成 8 个 responses、32 次 gradient accumulation、temperature 1.0、最多
512 tokens、最大 gradient norm 1.0。每 10 steps 在 1,024 个 GSM8K validation
examples 上评测一次，每 40 steps 保存一次 rollout。四个完整 run 使用 seed 0、1、2、3；
W&B 中早期只运行 50 steps 的旧 seed-0 run 不计入最终统计。原始交互式曲线和 rollout
tables 见 [W&B workspace](https://wandb.ai/jerry520/cs336-assignment5-grpo/workspace?nw=nwuserjerry520)。

#### （b）50-step sanity check

前 50 steps 已经给出了清晰的正确性证据。四个 seed 的 validation reward 从 step 10
的 0.0000--0.0313 上升到 step 50 的 0.2695、0.2959、0.3955、0.3477；同时 validation
format reward 从 0.5127--0.6563 上升到 0.8828--0.9727。也就是说，提升同时出现在
答案正确性和格式遵循上，而不是只由某一个 batch 的偶然 reward 造成。训练没有出现
NaN、无限 gradient norm 或 reward 崩溃，因此我继续运行了完整的 200 steps。

#### （c）四个随机种子的完整结果

下图由 W&B public histories 生成。每条浅色实线代表一个 seed；loss、gradient norm、
entropy 和 response length 面板中的黑线是四个 seed 的均值，灰色区域是逐 step 的
min--max 区间。reward 面板的实线为 total reward，虚线为 format reward。

![Standard on-policy GRPO metrics across four seeds](results/grpo_standard_metrics.png)

各指标的变化如下：

- Policy-gradient loss 在 0 附近呈有噪声的负值（大多约为 $-0.01$ 到 $-0.04$），
  没有持续增大或发散。该 surrogate loss 本身不要求单调下降。
- Gradient norm 从接近 0 上升后主要在约 0.8--1.3 附近波动，偶尔出现约 1.8 的峰值，
  但没有持续增长；训练使用 `max_grad_norm=1.0` 进行 clipping。
- Token entropy 从约 0.9--1.2 快速下降，在约 step 60 后稳定在 0.10--0.16，说明 policy
  随着训练变得更确定。四个 seed 的下降轨迹相似。
- Train total reward 从接近 0 提升到约 0.4--0.6，但因为它只基于当前 rollout batch，
  step 间波动明显。Train format reward 很快达到约 0.9--0.98。
- Validation total reward 的趋势比 train reward 平滑：约 step 60 后达到 0.36--0.41，
  随后缓慢提升，并在 step 200 收敛到 0.4570--0.4658。Validation format reward
  最终为 0.9355--0.9580。
- Validation average response length 在训练前期的 seed 间差异较大，但后期收敛到
  135--144 tokens。长度没有无界增长，因此 reward 提升不像是由不断延长回答造成的。

最终验证结果为：

| Seed | Final val reward / accuracy | Final val format reward | Final average response length |
|---:|---:|---:|---:|
| 0 | 0.4570 | 0.9355 | 135.14 |
| 1 | 0.4658 | 0.9580 | 144.01 |
| 2 | 0.4629 | 0.9355 | 136.10 |
| 3 | 0.4639 | 0.9580 | 139.22 |
| **Mean** | **0.4624** | **0.9468** | **138.62** |

四个 seed 的最终 validation accuracy 均值为 46.24%，样本标准差为 0.38 个百分点，
范围为 45.70%--46.58%（最大差距 0.88 个百分点）。因此本实验不仅明显超过题目要求的
25% 平均准确率，而且最终结果的 seed 方差很小。需要注意的是，早期“开始学习”的时间
存在更明显的 seed 差异；例如 step 50 的 validation reward 范围仍有 0.2695--0.3955。

#### 训练前后的 rollout 示例

训练前的 `r1_zero` baseline 经常遵循标签格式，但算术推理错误。例如：

1. 对于“16 个蛋，吃 3 个、烘焙用 4 个，剩余每个卖 2 美元”，模型错误地把问题扩展成
   每日 144 个蛋并输出 `<answer>... $288 ...</answer>`，而正确答案是 18。
2. 对于“蓝色布料 2 bolts，白色为其一半，总共多少”，模型错误地计算为
   `<answer>... totals 1 bolt ...</answer>`，而正确答案是 3。

step 200 的实际 rollout 已能稳定完成多步算术并给出简洁的标签化答案。例如：

1. 对于“5 包金枪鱼每包 2 美元、4 瓶水每瓶 1.5 美元、总共支付 56 美元，其他商品花费
   多少”，模型计算 $5\times2=10$、$4\times1.5=6$、$56-(10+6)=40$，并输出
   `<answer> 40 </answer>`，与 ground truth 一致。
2. 对于“第一周每天训练 2 小时，第二周每天训练 3 小时，两周共多少小时”，模型计算
   $2\times7=14$、$3\times7=21$、$14+21=35$，并输出
   `<answer> 35 </answer>`，与 ground truth 一致。

这些例子与定量结果一致：训练后模型不只是更经常生成合法的 `<think>`/`<answer>`
结构，也更经常完成正确的中间运算并把最终数值单独放入 answer 标签中。
