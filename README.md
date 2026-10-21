# 扰动预测评估

统一读取 **预测扰动、真实扰动、真实对照三份 H5AD**，使用 `cell-eval2==0.18.0` 计算六项 VCC2026 原始指标。Chreode 与 Lingshu-Cell 共用同一份 truth/control/panel/conditions，最后用 `baseline.compare` 汇总成一张表。

```text
evaluate/          # H5AD 读取、指标和四路径命令入口
baseline/          # Lingshu、Chreode 的推理与格式整理
tests/             # 小数据回归测试
pyproject.toml     # 依赖与安装配置
README.md
```

## 安装

Python ≥3.11；评估环境无需 PyTorch 或 CUDA。

```bash
conda create -n perturbation-eval python=3.11 pip -y
conda activate perturbation-eval
cd /data/xiebo/evaluate
python -m pip install -e .
```

## 四路径评估

```bash
python -m evaluate \
  --pred-path /data/xiebo/outputs/lingshucell/prediction.h5ad \
  --truth-path /data/dataset/lingshucell/vcc/evaluation/truth.h5ad \
  --control-path /data/dataset/lingshucell/vcc/evaluation/control.h5ad \
  --output-dir /data/xiebo/results/lingshucell
```

`perturbation-eval` 与 `python -m evaluate` 等价。评估器只读 H5AD，不读 PKL，也不依赖模型仓库、条件表或外部基因映射。

| 参数 | 内容 |
| --- | --- |
| `--pred-path` | 预测扰动细胞 H5AD |
| `--truth-path` | 真实测试扰动细胞 H5AD |
| `--control-path` | 真实对照细胞 H5AD |
| `--output-dir` | 写入 `summary.csv`、`summary.json` 的目录 |

## 两模型汇总

```bash
python -m baseline.compare \
  --truth-path /data/dataset/lingshucell/vcc/evaluation/truth.h5ad \
  --control-path /data/dataset/lingshucell/vcc/evaluation/control.h5ad \
  --pred lingshucell=/data/xiebo/outputs/lingshucell/prediction.h5ad \
  --pred chreode=/data/xiebo/outputs/chreode/prediction.h5ad \
  --output-dir /data/xiebo/results/comparison
```

输出 `comparison.csv`、`comparison.json`、`comparison.md`，以及每个模型子目录下的 `summary.*`。`comparison.csv` 额外给出 `average_rank`，它只是六项排名的描述性汇总，不是官方 VCC 总分。

## 共用参考数据

两个模型必须在同一 panel 上评分。`baseline.prepare_reference` 可以同时接受基因列表、Chreode 原生词表或某个模型的预测来定义 panel，并生成固定的 truth/control/conditions：

```bash
cd /data/xiebo/evaluate
python -m baseline.prepare_reference \
  --panel-path /data/dataset/lingshucell/vcc/gene_names.csv \
  --panel-vocab /data/xiebo/Chreode \
  --truth-path /data/dataset/lingshucell/vcc/test/adata_Test.h5ad \
  --control-path /data/dataset/lingshucell/vcc/test/adata_Test.h5ad \
  --output-dir /data/dataset/lingshucell/vcc/evaluation \
  --max-cells-per-pert 400 \
  --max-control-cells 2000 \
  --conditions-out /data/dataset/lingshucell/vcc/evaluation/conditions.csv
```

- `--panel-path` 可以是基因列表（每行一个）或 H5AD；`--panel-vocab` 读取 Chreode 的 16,485 原生词表；两者与 truth/control 再取交集。
- `--max-cells-per-pert` 控制每扰动真实细胞数，并写入 `conditions.csv`；Chreode 和 Lingshu 都按这份表生成相同数量的细胞。
- 输出 `truth.h5ad`、`control.h5ad`、`panel.txt`；之后两个模型共用，不要按模型分别重做。

## H5AD 约定

- 每行一个细胞，每列一个基因，保留逐细胞表达；不能仅保存扰动均值。
- 表达默认读取 `X` 的原始非负整数 counts。归一化表达必须在文件内声明 `uns['evaluation']['expression_scale'] = 'lognorm'`；已有 `uns['vcc']['expression_scale']` 也兼容。
- 若 counts 在 layer 中，可在文件内设 `uns['evaluation'] = {'expression_scale': 'counts', 'layer': 'counts'}`。
- 扰动标签优先读取 `obs.target_gene`，兼容 `obs.gene_target`、`obs.condition`、`obs.target`；多列同时存在时须一致。预测和真实的标签集合必须相同；细胞数可以不同，不要求细胞配对。`prepare_reference` 会把单扰动统一写成 `GENE`（把 `GENE+ctrl` 还原为 `GENE`），组合保留 `GENE1+GENE2`；模型适配器必须导出相同标签。
- 对照标签支持 `ctrl`、`control`、`non-targeting`、`non-perturbation`、`NT`，忽略大小写。对照文件没有扰动标签列时，全部行视为真实对照；有标签列时只选对照行。
- 基因优先读取 `var.gene_name`、`gene_symbols`、`symbol`、`feature_name`，否则使用 `var_names`。要求唯一的基因 symbol；匹配忽略大小写，自动重排顺序。
- 基因面板固定为 truth 文件的全部基因。预测和对照必须覆盖这些基因，可以多出基因；缺基因直接报错，避免各模型分别取交集造成分数不可比。
- 如有 `obs.cell_type`，每份文件必须只包含同一细胞背景，三份文件的背景须一致。

counts 作为线性丰度；lognorm 用 `expm1` 转为线性丰度，再在固定参考基因上统一 `normalize_total(10,000) + log1p`。Chreode 导出器会把预测负值截到 0 并记录；评估器拒绝负数和 NaN/Inf。

## 模型入口

| 模型 | 训练 | 推理 | 输出 |
| --- | --- | --- | --- |
| Lingshu-Cell | 官方权重，无需本仓库训练 | `python -m baseline.lingshucell.infer` | counts，18,080 基因 |
| Chreode | `python -m baseline.chreode.train` | `python -m baseline.chreode.infer` | lognorm，原生词表内可映射基因 |

旧的 GEARS Norman 入口仍保留在 `baseline.gears.infer` / `export.py`，与 VCC 是不同实验，不参与本次两模型对比。

## 六项指标与输出

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| `pds_cosine` | 扰动效应检索排名 | ↑ |
| `expr_mse_unbiased_capped_norm` | 噪声修正后的归一化表达误差 | ↓ |
| `de_wilcoxon_direction_fidelity_yield_raw` | DE 方向准确性与覆盖 | ↑ |
| `de_wilcoxon_direction_reach_raw` | 指定方向纯度下的覆盖深度 | ↑ |
| `de_wilcoxon_sig_jaccard` | 显著 DE 基因集合一致性 | ↑ |
| `de_wilcoxon_lfc_nmae` | fold-change 归一化绝对误差 | ↓ |

使用真实对照、余弦距离、midrank、`n-1` 排名分母、Wilcoxon、每扰动 BH-FDR 0.05；CPU/pdex，关闭 counts 专用 CPM 过滤。PDS 排除面板全部靶基因；表达和 DE 排除各扰动自身的全部靶基因。

输出仅有 `summary.csv` 和 `summary.json`。JSON 同时记录输入路径、基因轴与指纹、真值/对照标准化表达指纹、表达尺度、靶基因及指标状态。未定义值记 `null`。这是六指标的 lognorm 适配，不做 baseline/replicate 校准或总分，不等同于官方 counts 成绩。

## 小规模验证

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

本地仅做格式及小数据测试，完整训练和推理在服务器运行。
