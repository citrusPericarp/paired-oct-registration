# 配对 OCT 配准

[English](README.md) | **简体中文**

本仓库提供术前、术后配对 OCT B-scan 的配准代码，以及基于已保存变换的图像重建工具。

流程以视网膜掩膜和边界为结构先验：**A-OCT** 完成全局横向对齐和逐列轴向校正，
**D-OCT** 通过基于纹理的动态时间规整细化局部配准。

## 安装

使用 Python 3.10 或 3.11，并安装锁定的数值依赖。

```bash
git clone https://github.com/citrusPericarp/paired-oct-registration.git
cd paired-oct-registration
python -m pip install -r requirements-reproduction.txt -e .
```

## 使用

配准单幅扫描：

```bash
paired-oct-register --dataset ../dataset --output ../output --sample-id 0001
```

处理清单中的全部扫描：

```bash
paired-oct-register --dataset ../dataset --output ../output
```

利用已保存的变换重建配准图像、掩膜和边界：

```bash
paired-oct-reconstruct --dataset ../dataset --output ../reconstructed
```

## 输入与输出

在数据根目录下准备 `manifests/pairs.csv`，包含以下字段：

```text
sample_id,real_A,real_B,mask_A,mask_B,boundary_A,boundary_B
```

文件路径均相对于数据根目录。`real_A`、`real_B` 分别为术前、术后图像，
掩膜和边界为模型生成的视网膜结构先验。边界 `.npz` 文件包含
`upper_y`、`lower_y` 和 `valid_columns`。重建命令还读取清单中的 `deformation` 路径。

| 输出 | 内容 |
|---|---|
| `aligned_B/` | 配准后的术后图像 |
| `aligned_mask_B/`、`aligned_boundary_B/` | 变换后的掩膜及其边界 |
| `valid_mask/` | 源采样坐标有效的像素区域 |
| `evaluation_mask/` | 术前与配准后视网膜掩膜的交集 |
| `deformation/` | 保存的配准参数（`.npz`，格式 2） |
| `registration_manifest.csv`、`failures.json` | 输出路径与处理状态 |

## 代码结构

- [`src/paired_oct_registration/`](src/paired_oct_registration/)：配准、重建与命令行接口。
- [`scripts/`](scripts/)：数据批处理工具。
- [`tests/`](tests/)：配准与重建检查。

## 许可证

[MIT](LICENSE)。
