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

数据位置完全由 `manifests/pairs.csv` 中的相对路径决定，不依赖作者电脑路径。

修正后的形变文件采用参数格式2。可直接利用保存的变换重建五类图像及边界产品：

```powershell
paired-oct-reconstruct --dataset ..\dataset --output ..\reconstructed
```

当前生成环境为Python 3.9.23、NumPy 1.26.4、SciPy 1.10.1、
OpenCV 4.11.0.86和Pillow 11.1.0。公共包要求Python>=3.10，当前数值依赖也支持3.10/3.11。应在独立环境中按所附requirements安装，
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

代码仓库：https://github.com/citrusPericarp/paired-oct-registration
（MIT 许可证）。临床图像和元数据单独管理，不包含在本仓库中。


## 2026-10-01 更新（软件 0.3.0）

取消 D-OCT 整图共同列覆盖比例门槛。可信局部域非空即可进入逐列估计，空域返回零残差。
逐列厚度检查、纹理 DTW、上下边界锚定、紧支撑和单调约束保持不变。
Python 兼容参数 `minimum_valid_fraction` 默认 0.0 并忽略其数值，旧 0.70/0.75
门槛不再适用。采用图像支持的中心凹判断、有界端点延拓与横移后局部几何支持。
DTW 的参考数值路径不变。算法版本 `20261001-no-whole-image-gate` 写入形变档案，
`format_version=2` 读取与重建语义保持不变。

`aligned_mask_B` 仍为原生 mask_B 的变换结果；`evaluation_mask` 仍是它与域内
mask_A 的交集。新增 `resegmented_mask_B`、`resegmented_boundary_B` 是冻结 v10
对新版 aligned_B 的预测，完整保留模型输出，不反馈本轮配准；测量时需显式定义
与 valid_mask 的交集。模型预测不属于临床真值。

独立 v10 包未包含在配准包或当前 dataset 目录中，公共下载入口尚待落实。包需包含 source、
`checkpoints/best_reviewed_terminal_trajectory.pt` 和
`source/configs/decoder_v10_terminal_connected_candidate_20260910.json`。
按该包说明安装 PyTorch/推理依赖后运行：

```powershell
python scripts/infer_registered_B.py --dataset ..\dataset --release-root X:\v10_bundle --output ..\registered_B_inference --device cuda --batch-size 4
```

脚本记录真实输入角色 registered_B，输出预测/来源记录和形态 QC，并写入
all/normal/abnormal 三个视图。不会拿变换结构先验当真值计算 Dice。
重建变换产品只需 deformation；重生成新增分割产品还需要冻结权重和 decoder。

本轮生成环境为 Python 3.9.23、NumPy 1.26.4、SciPy 1.10.1、OpenCV 4.11.0、
Pillow 11.1.0。历史 Python 3.11.15/NumPy 2.0.1/SciPy 1.17.1 环境属于上一版，不是当前数值依赖。
公共包安装支持 Python>=3.10；数值依赖请用当前 requirements-reproduction.txt。
