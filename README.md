# 扰动预测六指标评估

输入**真值 H5AD、模型预测 H5AD、真实对照 H5AD、输出目录**四个路径，计算 [Virtual Cell Challenge 2026](https://virtualcellchallenge.org/) 对应的六项**原始指标**。兼容 Lingshu-Cell VCC 推理输出的 counts H5AD，也保留 GEARS `test_res.pkl` 输入。

使用 `cell-eval2==0.18.0` 的指标函数与聚合。H5AD 使用 `vcc2026` counts 配置；GEARS 使用 log-normalized 适配。只计算原始指标，不做 baseline/replicate 校准，不计算总分。

## 安装与更新

需要 Python 3.11 或更新版本。建议使用单独的评估环境，模型训练、推理环境继续用于生成预测。

```bash
source ~/.bashrc
conda create -n perturbation-eval python=3.11 pip -y  # 只需首次创建
conda activate perturbation-eval

cd /data/xiebo/evaluate
git pull --ff-only
python -m pip install -e .
```

`source ~/.bashrc` 适用于 Bash；使用 Zsh 时换成 `source ~/.zshrc`。如果已有 `GEARS` 环境的 Python ≥3.11，也可以在该环境安装，但依赖升级可能影响训练环境。

首次克隆时使用 `git clone https://github.com/Zemiss/perturbation-eval.git /data/xiebo/evaluate`；已有仓库只需上述 `git pull`。

## 运行

### H5AD：四个路径

```bash
perturbation-eval \
  --truth-path /data/truth.h5ad \
  --pred-path /data/merged.h5ad \
  --control-path /data/control.h5ad \
  --output-dir /data/results/model_run1
```

| 参数 | 文件内容 |
| --- | --- |
| `--truth-path` | 实验测到的真实扰动细胞 |
| `--pred-path` | 模型生成的扰动细胞，如 Lingshu-Cell 的 `merged.h5ad` |
| `--control-path` | 实验测到的真实未扰动对照细胞 |
| `--output-dir` | 保存 `summary.csv`、`summary.json` 的目录 |

不需要先生成 `merged_with_control.h5ad`。程序在内存中把独立真实对照分别用于预测和真值；真值或预测里已经附带的对照会被忽略。`--control-path` 始终是唯一的对照来源。

H5AD 输入规则：

- 三份文件的 `X` 都须是未经过 log1p/归一化的 counts：非负、有限、整数，每细胞总 counts 不超过 1,000,000。支持稠密或稀疏矩阵；不读取 `raw`/layers，不自动取整或猜测表达尺度。
- 扰动列优先使用 `obs.target_gene`，兼容 `gene_target`、`condition`、`target`。若同时存在多列，其内容须一致。对照标签支持 `non-targeting`、`ctrl`、`non-perturbation`。
- 对照文件可只含对照且没有扰动标签列，此时使用所有细胞；有标签列时只提取对照行，忽略其他扰动。文件必须提供实验对照，不能提供模型生成的对照。
- 基因按名称对齐，优先使用 `var_names`，必要时使用文件内的 `var.gene_name` 注释。预测的全部基因必须在真值、对照中唯一匹配；真值、对照中的额外基因会被移除并记录。没有对应注释时，不自动转换 Ensembl ID 与基因符号。
- 只评估预测中出现的扰动；真值中其他扰动被跳过并记录。预测中存在但真值中缺失的扰动会报错。两侧细胞不要求一一配对，数量可不同。
- 支持 VCC 单靶基因扰动。三个输入必须属于同一实验背景；如提供 `obs.cell_type`，只允许一个相同的细胞类型。其他批次、供体等背景须在输入前筛选。
- counts 分支直接使用原库的 pseudobulk 归一化、噪声修正、靶基因排除及 Wilcoxon/BH-FDR 规则；保留对照 CPM 过滤。JSON 记录三份实际输入路径、使用的细胞数、基因对齐、跳过的扰动与配置。

### GEARS：保留三个路径

```bash
gears-eval \
  --pred-path /data/xiebo/evaluate/outputs/GEARS/test_res.pkl \
  --truth-path /data/xiebo/Chreode/output/gears/data/ \
  --output-dir /data/xiebo/evaluate/results/GEARS_run2
```

同样支持 `perturbation-eval`、`python run.py`、`python evaluate_gears.py` 和安装后的 `python -m evaluate`，参数相同。默认固定使用 CPU/pdex，不需要 PyTorch、GEARS 包或 CUDA。

## GEARS 输入与计算规则

- `--pred-path`：可信的 GEARS PKL，包含细胞 × 基因的 `pred`、配对测试 `truth` 和每个细胞的 `pert_cat`；也可指定包含 `test_res.pkl` 的目录。只读取自己的可信 PKL。
- `--truth-path`：GEARS 数据根目录、Norman 目录，或明确的 H5AD。目录会查找 `norman/perturb_processed.h5ad` 或 `perturb_processed.h5ad`；发现两份时必须明确指定文件。
- 真实扰动表达使用 PKL 内的测试 `truth`；H5AD 的 `obs.condition == 'ctrl'` 和 `X` 提供真实对照。其他训练、验证或扰动细胞不加入评分。
- 基因名优先使用 H5AD 的 `var.gene_name`，否则使用 `var_names`。通常 PKL 没有基因名，因此假定两侧是同一 GEARS 数据生成的、完整且顺序相同的基因轴；维度不同时报错。若 PKL 有 `gene_names`，按名称校验并对齐。使用全部这些基因，不再读取 Chreode 词表。
- PKL 的预测、测试真值和 H5AD 的 `X` 应使用相同的 GEARS `log1p` 归一化尺度。程序不把表达值伪装成 counts，不取整，也不读取 `raw`/layers 猜测表达尺度。数值检查无法独立证明两个文件采用了相同的预处理。
- 预测负值在 log1p 空间截到 0；截断数量打印并写入 `summary.json`。真实表达负值、NaN/Inf、非法基因轴或标签会报错。
- 保留 `GENE+ctrl`、`GENE1+GENE2` 等完整扰动身份。PDS 排除整个测试面板涉及的靶基因；表达误差和 DE 排除每个扰动自身的全部靶基因。靶基因按符号匹配，大小写不同会对齐；未测量靶基因记录在结果中。
- 以 `vcc2026` 参数为基础：真实对照、余弦距离、midrank、`n-1` 排名分母、Wilcoxon、每扰动 BH-FDR 0.05。沿用原库 CPU lognorm 行为，不应用 counts 专用的 CPM 过滤。
- lognorm 表达/PDS 使用细胞表达均值，表达噪声修正使用原库的二阶矩、预测修正上限和面板预算约束；组合扰动扩展为排除全部组成基因。表达误差最终使用**分子之和／分母之和**，不平均逐扰动比值。

## 输出

只生成 `summary.csv` 和 `summary.json`，六项指标为：

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| `pds_cosine` | 扰动效应余弦检索排名 | ↑ |
| `expr_mse_unbiased_capped_norm` | 经噪声修正、上限约束并按真实效应归一化的表达误差 | ↓ |
| `de_wilcoxon_direction_fidelity_yield_raw` | DE 方向准确性与覆盖 | ↑ |
| `de_wilcoxon_direction_reach_raw` | 指定方向纯度下的覆盖深度 | ↑ |
| `de_wilcoxon_sig_jaccard` | 显著 DE 基因集合 Jaccard | ↑ |
| `de_wilcoxon_lfc_nmae` | log fold-change 归一化绝对误差 | ↓ |

JSON 另含完整配置、实际路径、靶基因、预处理记录和指标状态。原库未定义的指标写为 `null`；真实效应的噪声修正分母之和非正时会报错，不生成伪造结果。

指标定义见 [cell-eval2 的 VCC2026 文档](https://github.com/ArcInstitute/cell-eval2/tree/main/docs/vcc2026_metrics)。归一化 GEARS 数据与官方 counts 的表达比较空间不同，结果只能在相同数据、预处理和配置下比较。

## 最小验证

测试检查 H5AD 四参数命令、独立真实对照、基因对齐、不同细胞数量、与原库 counts 结果的一致性；同时保留 GEARS 三参数、lognorm 数值和组合扰动排除回归测试，并拒绝关键非法输入。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```
