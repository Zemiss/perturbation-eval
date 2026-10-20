# 扰动预测评估

统一读取 **预测扰动、真实扰动、真实对照三份 H5AD**，使用 `cell-eval2==0.18.0` 计算六项 VCC2026 原始指标。每次评估一个模型，命令只需四个路径参数。基线推理及参考数据整理见 [baseline/README.md](baseline/README.md)。

```text
evaluate/          # H5AD 读取、指标和四路径命令入口
baseline/          # Lingshu、Chreode、GEARS 的推理与格式整理
tests/             # 小数据回归测试
pyproject.toml     # 依赖与安装配置
README.md
```

## 安装

Python ≥3.11；评估环境无需 PyTorch、GEARS 或 CUDA。

```bash
conda create -n perturbation-eval python=3.11 pip -y
conda activate perturbation-eval
cd /data/xiebo/evaluate
python -m pip install -e .
```

## 四路径评估

先准备好同一测试面板的真实扰动与对照文件，再分别评估模型。下面的 `evaluation/truth.h5ad`、`evaluation/control.h5ad` 是整理脚本生成的参考文件；官方原始下载保留 `train/`、`validation/`、`test/` 子目录，结构及整理命令见 [基线说明](baseline/README.md#路径)。

```bash
cd /data/xiebo/evaluate
python -m evaluate \
  --pred-path /data/xiebo/outputs/lingshucell/prediction.h5ad \
  --truth-path /data/dataset/lingshucell/vcc/evaluation/truth.h5ad \
  --control-path /data/dataset/lingshucell/vcc/evaluation/control.h5ad \
  --output-dir /data/xiebo/results/lingshucell

python -m evaluate \
  --pred-path /data/xiebo/outputs/chreode/prediction.h5ad \
  --truth-path /data/dataset/lingshucell/vcc/evaluation/truth.h5ad \
  --control-path /data/dataset/lingshucell/vcc/evaluation/control.h5ad \
  --output-dir /data/xiebo/results/chreode
```

`perturbation-eval` 与 `python -m evaluate` 等价。评估器不读取 PKL，不再使用 `vcc` 子命令，也不依赖模型仓库、条件表或外部基因映射。

| 参数 | 内容 |
| --- | --- |
| `--pred-path` | 预测扰动细胞 H5AD |
| `--truth-path` | 真实测试扰动细胞 H5AD |
| `--control-path` | 真实对照细胞 H5AD |
| `--output-dir` | 写入 `summary.csv`、`summary.json` 的目录 |

## H5AD 约定

- 每行一个细胞，每列一个基因，保留逐细胞表达；不能仅保存扰动均值。
- 表达默认读取 `X` 的原始非负整数 counts。归一化表达必须在文件内声明 `uns['evaluation']['expression_scale'] = 'lognorm'`，含义是 log1p(normalized abundance)。已有 Chreode 的 `uns['vcc']['expression_scale']` 也兼容。程序不通过数值猜测尺度。
- 若 counts 在 layer 中，可在文件内设 `uns['evaluation'] = {'expression_scale': 'counts', 'layer': 'counts'}`，评估仍只需四个参数。
- 扰动标签优先读取 `obs.target_gene`，兼容 `obs.gene_target`、`obs.condition`、`obs.target`；多列同时存在时须一致。预测和真实扰动的标签集合必须相同；细胞数可以不同，不要求细胞配对。保留 `GENE+ctrl`、`GENE1+GENE2` 的完整身份。
- 对照标签支持 `ctrl`、`control`、`non-targeting`、`non-perturbation`、`NT`，忽略大小写。如果对照文件没有扰动标签列，就将其全部行视为真实对照。有标签列时只选对照行，预测和真值内附带的对照会忽略，始终只使用 `--control-path` 的真实对照。
- 基因优先读取 `var.gene_name`、`gene_symbols`、`symbol`、`feature_name`，否则使用 `var_names`。要求唯一的基因 symbol；匹配忽略大小写，自动重排顺序。原始 ENSG ID 可通过参考整理脚本转换。
- 基因面板固定为真实扰动文件的全部基因。预测和对照必须覆盖这些基因，可以多出基因。缺基因时直接报错，避免各模型分别取交集造成分数不可比。VCC 两模型共享同一份整理后的 truth/control，详见基线文档。
- 如有 `obs.cell_type`，每份文件必须只包含同一细胞背景，三份文件的背景须一致。

counts 作为线性丰度；lognorm 用 `expm1` 转为线性丰度，再在固定参考基因上统一 normalize_total(10,000) + log1p。`expm1` 不恢复原始 counts。GEARS 导出器与 Chreode 推理脚本会将预测负值截到 0 并记录；评估器拒绝负数和 NaN/Inf。

## GEARS

新的 [推理入口](baseline/gears/infer.py) 直接写出 `/data/xiebo/outputs/gears/prediction.h5ad`，另将测试真值和真实对照写到数据目录。已有可信 `test_res.pkl` 可用 [转换入口](baseline/gears/export.py) 先转成三份 H5AD。两种方式都保留完整逐细胞结果和组合扰动；命令见 [基线说明](baseline/README.md)。随后调用同一个评估入口：

```bash
python -m evaluate \
  --pred-path /data/xiebo/outputs/gears/prediction.h5ad \
  --truth-path /data/dataset/gears/evaluation/truth.h5ad \
  --control-path /data/dataset/gears/evaluation/control.h5ad \
  --output-dir /data/xiebo/results/gears
```

只有相同数据、测试扰动、基因面板与配置的分数才可比较；GEARS Norman 与 VCC 是不同实验。

## 六项指标与输出

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| `pds_cosine` | 扰动效应检索排名 | ↑ |
| `expr_mse_unbiased_capped_norm` | 噪声修正后的归一化表达误差 | ↓ |
| `de_wilcoxon_direction_fidelity_yield_raw` | DE 方向准确性与覆盖 | ↑ |
| `de_wilcoxon_direction_reach_raw` | 指定方向纯度下的覆盖深度 | ↑ |
| `de_wilcoxon_sig_jaccard` | 显著 DE 基因集合一致性 | ↑ |
| `de_wilcoxon_lfc_nmae` | fold-change 归一化绝对误差 | ↓ |

使用真实对照、余弦距离、midrank、`n-1` 排名分母、Wilcoxon、每扰动 BH-FDR 0.05；CPU/pdex，关闭 counts 专用 CPM 过滤。PDS 排除面板全部靶基因；表达和 DE 排除各扰动自身的全部靶基因。表达误差保留原库噪声修正、预测修正上限及面板预算约束，按分子之和／分母之和聚合。

输出仅有 `summary.csv` 和 `summary.json`。JSON 同时记录输入路径、基因轴与指纹、真值/对照标准化表达指纹、表达尺度、靶基因及指标状态。未定义值记 `null`；真实效应的噪声修正分母之和非正时报错。这是六指标的 lognorm 适配，不做 baseline/replicate 校准或总分，不等同于官方 counts 成绩。指标实现可对照本地只读的 `cell-eval2` 仓库。

数据集及整理后的参考 H5AD 放 `/data/dataset/`；代码、权重、预测、运行记录和评分放 `/data/xiebo/`。

## 小规模验证

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

本地仅做格式及小数据测试，完整训练和推理在服务器运行。
