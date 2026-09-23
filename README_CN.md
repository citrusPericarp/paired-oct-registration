# 配对 OCT 配准代码

该代码读取独立下载的数据集，用 `real_A`、`real_B`、`mask_A`、`mask_B`、
`boundary_A` 和 `boundary_B` 生成 `aligned_B`、`aligned_mask_B`、
`aligned_boundary_B`、`valid_mask`、`evaluation_mask` 和可复现的变形参数。

本代码包只包含配准和已保存形变参数的重建；RetinaUNet训练及推理、图像恢复模型实验、
血管投影阴影分析和临床数据集均不包含在此代码包中。

```powershell
python -m pip install -r requirements-reproduction.txt -e .
paired-oct-register --dataset ..\dataset --output ..\output
```

只测试一张：

```powershell
paired-oct-register --dataset ..\dataset --output ..\output --sample-id 0001
```

数据位置完全由 `pairs.csv` 中的相对路径决定，不依赖作者电脑路径。

修正后的形变文件采用参数格式2。可直接利用保存的变换重建五类图像及边界产品：

```powershell
paired-oct-reconstruct --dataset ..\dataset --output ..\reconstructed
```

参考环境为Python 3.11.15、NumPy 2.0.1、SciPy 1.17.1、
OpenCV 4.11.0.86和Pillow 11.1.0。应在独立环境中按所附requirements安装，
OpenCV 5的插值结果不同。DTW固定采用同一数值路径，不随可选Numba的安装状态改变。
`paired-oct-register`从图像和发布结构先验重新估计变换，
`paired-oct-reconstruct`从保存的参数重建，两者用途不同。
RetinaUNet训练与结构先验推理属于独立的上游流程。

归档直接保存目标锚点、残差节点的深度边界和源横向坐标，重建时不再从
降低精度的观测分数恢复几何门控。旧版不包含这些字段的形变文件需先重新生成。

mask 和 boundary 是算法生成的结构先验，不是人工标注或临床真值；
`aligned_B` 是配准伪参考。分析黑色越界像素时，应使用 `valid_mask` 排除
无效位置。`aligned_mask_B` 是使用与 `aligned_B` 相同变形得到的 B
结构 mask；`aligned_boundary_B` 是从 `aligned_mask_B` 提取的上下边界和
原始有效列标记；`evaluation_mask` 是有效域内 `mask_A` 与 `aligned_mask_B`
的交集，用于配对评价，不是新的临床标注。
