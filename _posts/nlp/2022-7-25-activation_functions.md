---
layout: default
title:  "激活函数汇总"
permalink: /activation_functions.html
---
# Sigmoid
<div align=center> <img src="{{ site.url }}/assets/images/sigmoid.png" width="500"> </div>  

## 函数

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

## 导数

$$
\sigma'(x) = \sigma(x)\cdot(1-\sigma(x))
$$

## 优点
* 函数是可微的。
* 梯度平滑。
* Sigmoid 函数的输出范围是 0 到 1，因此它对每个神经元的输出进行了归一化，课用于将预测概率作为输出的模型。

## 缺点
* 激活函数计算量大（在正向传播和反向传播中都包含幂运算和除法）
* 不是zero-centered
* 梯度消失

# Tanh

<div align=center> <img src="{{ site.url }}/assets/images/tanh.png" width="500"> </div>  

## 函数

$$
\tau(x) = \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

## 导数

$$
\tau'(x) = 1 - \tau^{2}(x)
$$

## 优点

* zero-centered
* 相比sigmoid梯度消失有所改善
* 梯度更大，容易收敛

## 缺点

* 梯度消失

# ReLu

<div align=center> <img src="{{ site.url }}/assets/images/relu.png" width="500"> </div>  

## 函数

$$
\sigma(x) = max(0, x)
$$

## 导数

$$
\sigma'(x) = \left\lbrace
\begin{array}{cl}
0 & x < 0 \\
1 & x > 0 \\
\end{array}\right.
$$

## 优点

* 没有饱和区，不存在梯度消失问题，防止梯度弥散。
* 稀疏性。
* 计算简单。
* 收敛快。

## 缺点

* 会导致部分节点dead，当学习率变大时，dead节点更多

# Softmax

## 函数

$$
\text{Softmax}(x_{i}) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

# GeLu

<div align=center> <img src="{{ site.url }}/assets/images/gelu.png" width="500"> </div>  

## 函数

$$
\operatorname{gelu}(x)=x P(X \leq x) \quad X \sim \mathcal{N}(0,1)
$$

## 实现

$$
\operatorname{gelu}(x)=\frac{1}{2} x\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right.
$$

```
import numpy as np
import torch
def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(torch.tensor(x) / np.sqrt(2.0)))
    return x * cdf
```

## 导数

$$
\begin{gathered}
\operatorname{gelu}^{\prime}(x)=0.5 \tanh \left(0.0356774 x^{3}+0.797885 x\right) \\
+\left(0.0535161 x^{3}+0.398942 x\right) \operatorname{sech}^{2}\left(0.0356774 x^{3}+0.797885 x\right)+0.5
\end{gathered}
$$

## 优点
* GeLu可以看作是dropout和relu的结合
* 按照正态分布去dropout节点

## 缺点
* 计算复杂度高