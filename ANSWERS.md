# CS336 作业 5：对齐

## 第一题：提示基线实验

我使用 `allenai/OLMo-2-0425-1B` 对 GSM8K 测试集中的全部 1,319
道题进行了评测。采样参数为：温度 1.0、top-p 1.0（vLLM 默认值）、
最大生成长度 512 tokens、随机种子 336。对于两种 R1 prompt，模型生成
`</answer>` 后停止，并在输出中保留该停止字符串。

### （a）评测结果

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

### （b）不同 prompt 对模型行为的影响

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

## 第二题：`baseline_calcs`

### （a）不使用 baseline 时的方差

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

### （b）使用常数 baseline $b$ 时的方差

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

### （c）使用 population mean baseline 时的方差

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

当 $0<p<1$ 时，$p^2(1-p)/n>0$，所以方差之差的符号只由
$3p-2$ 决定。因此：

- 当 $0<p<\frac{2}{3}$ 时，$3p-2<0$，population mean baseline
  会降低方差；
- 当 $p=\frac{2}{3}$ 时，两种 estimator 的方差相等；
- 当 $\frac{2}{3}<p<1$ 时，$3p-2>0$，population mean baseline
  反而会增大方差。

特别地，当 $p=\frac{1}{2}$ 时，

$$
\mathrm{Var}(\hat g_p)
=\frac{p(1-p)(1-2p)^2}{n}
=0.
$$

这是因为第（b）问的最优常数 baseline 是 $b^*=1-p$；只有在
$p=\frac{1}{2}$ 时，population mean baseline $b=p$ 恰好等于该最优值。
