# H5AD 扰动预测评估

直接读取 `outputs/` 下的原生推理结果，从 `dataset/` 中读取真实测试细胞并筛选真实对照，输出 cell-eval2 的六项指标。无需另存 `truth.h5ad`、`control.h5ad`，也无需手工合并预测和对照。

## 安装与运行

Python 3.11 或 3.12。以下路径与上一级 `main.md` 一致：

```bash
conda create -n perturbation-eval python=3.11 pip -y
conda activate perturbation-eval
cd /data/xiebo/mycode/evaluate
python -m pip install -e .

python -m evaluate \
  --pred-path /data/xiebo/mycode/outputs/lingshucell/prediction.h5ad \
  --output-dir /data/xiebo/mycode/results/lingshucell

python -m evaluate \
  --pred-path /data/xiebo/mycode/outputs/gears/prediction.h5ad \
  --output-dir /data/xiebo/mycode/results/gears
```

两条命令使用相同的真值、真实对照、基因面板和 lognorm 指标口径。`perturbation-eval` 与 `python -m evaluate` 等价。预测文件和原始 dataset 均只读；结果目录只生成 `summary.csv` 与 `summary.json`。

## dataset 的默认读取方式

默认 `--data-root /data/dataset/lingshucell/vcc`：

| 角色 | 读取位置 | 选择的细胞 |
| --- | --- | --- |
| 真实扰动 | `test/adata_Test.h5ad` | 非对照的真实测试细胞 |
| 真实对照 | `train/adata_Training.h5ad` | 标签为 `non-targeting` 等对照别名的真实细胞 |

从训练集选择对照，与当前 GEARS VCC 入口的训练对照来源一致；评估阶段始终读取原始 dataset 中的真实细胞，不使用模型生成的对照。训练文件必须有扰动标签列，不能把一个没有标签的完整数据集当成纯对照。若训练集缺少真实对照则报错，不自动改用其他来源。

评估器内部为两侧使用同一批真实对照，供表达效应和 DE 指标计算；不会要求你运行合并脚本，不写出 `merged_with_control.h5ad`。预测和测试文件中原有的对照行不进入扰动部分，也不重复加入对照池。

换 dataset 路径时使用 `--data-root /path/to/vcc`。需要改变真值或对照来源时，可分别用 `--truth-path`、`--control-path` 显式覆盖原始 H5AD；它们都不是必填参数。显式指定的对照文件可以是含真实对照的完整数据集，也可以是纯对照文件；只有显式指定的纯对照文件允许缺少标签列。

服务器路径来自 `main.md`，本地没有服务器真实数据；这里只根据代码确认输入约定，并用小数据验证完整流程。

## outputs 中应保存什么

两个模型都保存逐细胞的预测扰动表达，行是细胞、列是基因，不取扰动均值，不往里面追加真实对照或真实响应。

| 模型 | 已核对的推理入口 | 原生 H5AD 内容 |
| --- | --- | --- |
| Lingshu-Cell | `workflows.infer.main` | `merged.h5ad`；`X` 为稀疏整数 UMI counts；`obs.target_gene`、`obs.cell_type`；基因名在 `var_names` |
| 当前 GEARS VCC | `scripts/vcc.py predict` | `prediction.h5ad`；`X` 为逐细胞连续 lognorm；`obs.target_gene`、`obs.cell_type`；`var.gene_name` 与 `var_names`；文件内声明 lognorm 尺度 |

`outputs/lingshucell/prediction.h5ad` 可以直接放 Lingshu 的原生 `merged.h5ad`（复制或重命名即可），也可以把 `--pred-path` 指向原生文件路径。GEARS 的 `--output` 直接指定 `outputs/gears/prediction.h5ad`。无需经过本仓库的推理或格式转换脚本。

当前 GEARS 导出器已把负预测截为 0 并记录数量；评估器本身拒绝负值、NaN、Inf，不会再次修改推理结果。

## 基因对齐与字段

使用默认 dataset 真值时，只读相邻 `Lingshu-Cell/data/` 下的两个注释文件：

- `vcc/gene_id_name.csv`：Ensembl → symbol 映射，实际格式是无表头的两列 TSV。
- `vocabs/legacy/tokenizer_vcc_cond_dense_gene1tok.json`：两模型当前共用的固定 18,080 基因词表。

这些文件只用于 ID 和基因顺序，不导入模型代码，也不修改仓库。若相邻词表不存在，则使用 dataset 下的 `gene_names.csv`（存在时），否则使用 truth 全部基因。其他布局可用 `--gene-map` 和 `--gene-panel` 指定；显式传入 `--truth-path` 时，默认使用该 truth 全部基因，不自动套用 VCC 注释文件。

`--gene-panel` 支持一列 TXT/CSV/TSV，或 Lingshu 词表 JSON。基因名优先读取 `var.gene_name`、`gene_symbols`、`symbol`、`feature_name`，否则读取 `var_names`；Ensembl 版本后缀可通过映射解析。基因必须唯一，匹配忽略大小写，自动重排。预测和真实对照必须覆盖固定面板，不按模型分别取交集或补零。

扰动标签自动识别 `obs.target_gene`、`gene_target`、`condition`、`target`；其他字段用 `--pert-col`。对照别名支持 `non-targeting`、`non-perturbation`、`ctrl`、`control`、`NT`，忽略大小写，其他值用 `--control-label`。`GENE+ctrl` 自动还原为 `GENE`；组合扰动保留全部靶基因。多列候选标签冲突时，需用 `--pert-col` 明确选择。

预测与真实测试的扰动集合必须相同，每个扰动的细胞数可以不同，不要求细胞配对。每次评估一个细胞背景；存在 `obs.cell_type` 时，所选细胞必须只有一个背景，输入之间一致。

## 两种表达尺度使用同一评估口径

默认 `--protocol lognorm`，因此可以直接比较 Lingshu 的 counts 与 GEARS 的 lognorm，两个模型无需使用不同的命令参数：

- 输入尺度读取 `uns['evaluation']['expression_scale']`，兼容 `uns['vcc']['expression_scale']`；未声明时默认为 counts，适用于 Lingshu 原生输出。
- counts 作为线性丰度，lognorm 通过 `expm1` 还原线性丰度；在固定基因面板上统一 `normalize_total(10000) + log1p`。
- 计算六项指标，使用真实对照、cosine、midrank、`n-1` 排名分母、Wilcoxon 和每扰动 BH-FDR 0.05，关闭 counts 专用 CPM 过滤。组合扰动对表达和 DE 排除自身全部靶基因，PDS 排除面板全部靶基因。

GEARS 原生文件已正确声明 lognorm，不需要再传尺度参数。其他模型未写尺度元数据时，可用 `--pred-input-type lognorm` 明确声明。不会根据小数值猜测尺度，也不支持 scaled/z-score/residual 矩阵。

表达默认读取 `X`，也可通过 `uns['evaluation']['layer']` 或 `--pred-layer`、`--truth-layer`、`--control-layer` 指定。三份原始输入各自读取尺度与 layer，真实对照不继承 truth 文件设置。

若只评估原始 counts，可显式选择 `--protocol vcc2026`，直接使用 `cell-eval2==0.19.0` 的 VCC2026 counts 预设及其过滤、bulk 表达比较。该模式仅支持 counts 和单基因扰动，其数值与默认 lognorm 适配不等价；与 GEARS 比较时应让两个模型都使用默认 lognorm 模式。

计算固定使用 CPU/pdex。`--num-threads 16` 可限制线程，默认使用全部 CPU。

## 六项指标与输出

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| `pds_cosine` | 扰动效应的区分与检索排名 | ↑ |
| `expr_mse_unbiased_capped_norm` | 噪声修正后的归一化表达误差 | ↓ |
| `de_wilcoxon_direction_fidelity_yield_raw` | DE 方向准确性与覆盖 | ↑ |
| `de_wilcoxon_direction_reach_raw` | 指定方向纯度下的覆盖深度 | ↑ |
| `de_wilcoxon_sig_jaccard` | 显著 DE 基因集合一致性 | ↑ |
| `de_wilcoxon_lfc_nmae` | fold-change 归一化绝对误差 | ↓ |

`summary.csv` 保存六项原始指标；`summary.json` 同时记录输入路径、真实对照来源、细胞数、表达尺度、基因面板与表达指纹、靶基因和实际配置。未定义值记为 `null`。不额外计算基线缩放或总分。

## 仓库与验证

```text
evaluate/       # H5AD 读取、dataset 参考选择、指标和命令入口
tests/          # 小数据格式、指标一致性和命令行验证
baseline/       # 保留的旧实验工具，不是当前评估的前置步骤
```

当前模型推理分别由原仓库负责。`baseline.compare` 保留旧的显式参考路径接口，使用 lognorm 口径。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

本地只运行小数据检查；大规模训练、推理和服务器真实数据评估由服务器执行。
