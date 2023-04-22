---
title: Dynamic Programming
date: 2023-04-16
sidebar: auto
categories:
  - Algorithm
tags:
  - DP
---
## 动态规划（Dynamic Programming，DP）
### 概念
动态规划算法：是一种用于解决具有**重叠子问题**和**最优子结构性质**的优化问题的算法。它将原问题分解为一系列子问题，通过对这些子问题进行求解来得到原问题的最优解。
- 重叠子问题：在求解一个问题的过程中，会反复计算相同的子问题。
- 最优子结构性质：问题的最优解可以通过子问题的最优解来计算得到。
### 解法
接下来将从几个方面来描述，动态规划的解法：
- 状态转移方程：动态规划算法的核心就是找到状态转移方程，也就是当前状态与之前状态之间的关系。递推关系式可以根据问题的特点进行推导，通常采用数学归纳法的思路，先求解小规模的问题，然后利用小问题的解来逐步求解大问题的解。

- 优化策略：在一些情况下，为了优化动态规划算法的时间复杂度，需要采用一些优化策略，如空间压缩、滚动数组等。

- 缓存技术：动态规划算法通常会有大量的重复计算，为了避免这种情况，可以采用缓存技术，记录已经计算过的子问题的结果，避免重复计算。
## 状态转移方程
### 概念
状态转移方程是DP中描述当前状态和之前状态之间关系的数学表达式。它描述了当前状态的值可以通过之前某些状态的值递推得到。递推关系式通常由递推公式和边界条件两部分组成。

### 例子
假设有一排盒子，每个盒子都有一定的价值，我们要在不拿相邻的盒子的情况下，拿取尽可能多的价值。我们将每个盒子视为一个状态，假设第$i$个盒子的价值为$V_i$，我们用$f(i)$表示拿取前$i$个盒子所能得到的最大价值。这样，问题就被转化为了求解$f(n)$，即前$n$个盒子所能得到的最大价值。

对于递推关系式，我们需要找到$f(i)$和$f(i-1)$之间的关系。考虑拿取第$i$个盒子和不拿取第$i$个盒子两种情况：
- 拿取第$i$个盒子：由于不能拿相邻的盒子，所以前一个状态必须是$f(i-2)$，此时$f(i) = f(i-2) + V_i$。
- 不拿取第$i$个盒子：此时前一个状态是$f(i-1)$，此时$f(i) = f(i-1)$。

### 递推公式：
$f(i) = max(f(i-1), f(i-2) + V_i)$

注：其中就包含了，中间间隔很多次不拿的情况。（之前一直想不通）
### 边界公式
边界条件是指问题的初始状态，通常是简单的问题，可以直接得到解。在这个例子中，初始状态为$f(0) = 0$和$f(1) = V_1$。因此，状态转移方程的完整形式为：
$$
F=\begin{cases}
f(0) = 0 \\
f(1) = V_1\\
f(i) = max(f(i-1), f(i-2) + Vi), i > 1 
\end{cases}
$$