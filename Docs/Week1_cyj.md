# Week1

## 神经网络原理

1. 基本变换：层

    数学理解：通过对输入空间的操作，完成 **输入空间 -> 输出空间** 的变换
    - $\vec y = a(W\cdot\vec x + b)$
    - 升降维、缩放、旋转：$W\cdot\vec x$
    - 平移：$+\vec b$
    - 弯曲：激活函数

2. 理解视角

    数学视角：线性可分

    当网络有很多层时，对原始空间的“扭曲”会大幅增加，让我们更容易找到一个超平面分割空间

    神经网络的学习就是学习如何利用 **矩阵的线性变换 + 激活函数的非线性变换** ，将原始空间投向线性可分/稀疏的空间去分类/回归。
     - **增加维度**：增加线性转换的能力
     - **增加层数**：增加激活函数的次数，增加非线性转换次数

    [Neural Networks, Manifolds, and Topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)：

    > Theorem: Layers with N inputs and N outputs are homeomorphisms, if the weight matrix, W, is non-singular. (Though one needs to be careful about domain and range.)

    > The manifold hypothesis is that natural data forms lower-dimensional manifolds in its embedding space. There are both theoretical3 and experimental4 reasons to believe this to be true. If you believe this, then the task of a classification algorithm is fundamentally to separate a bunch of tangled manifolds.
    >
    > 流形假设是自然数据在其嵌入空间中形成低维流形。理论和实验都有理由相信这是真的。如果你相信这一点，那么分类算法的任务就是从根本上分离一堆纠结的流形。

    [可视化空间变化 demo](https://cs.stanford.edu/people/karpathy/convnetjs//demo/classify2d.html)：将神经网络每一层对空间的变化进行可视化，以全链接神经网络为例。

    ![Spatial_Transformation](./figures/Week1_cyj_Spatial_Transformation.PNG)


综上，神经网络可以理解为将输入空间不断变换，最终得到一个分类面。

增加全连接层可以提高数据线性可分的程度，但是跟数据的复杂情况有关。

以可视化空间变化 demo 为例，以 circle data  和 spiral data 为例，当激活函数为 relu 时，增加全连接层对 spiral 可以起到更好的分类效果，但是对 circle 则更大概率会过拟合。

## 卷积神经网络

[3D Visualization of a Covolutional Neural Network](http://scs.ryerson.ca/~aharley/vis/conv/)

![CNN_Visualization](./figures/Week1_cyj_CNNVisualization.PNG)



## 实验结果分析

TODO

## 其他工具

[神经网络绘图工具](http://alexlenail.me/NN-SVG/LeNet.html)

![Network_Architecture.PNG](./figures/Week1_cyj_Network_Architecture.PNG)