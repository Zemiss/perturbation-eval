# 扰动预测下游评估

给定单细胞扰动表达预测与对应真实数据，计算 **DE20 MSE ↓、DE20 Δr ↑、Pearson-Δ ↑、PDS ↑**。支持通用 H5AD/NPZ 输入，以及 GEARS Norman 的两目录评估入口。完整公式和聚合规则见“指标口径”。

仓库仅包含评估代码、示例配置与生成合成数据的测试。评估本身不需要模型权重、训练代码、GPU、GEARS 或 cell-eval 环境。

## 下载与安装

```bash
git clone https://github.com/Zemiss/perturbation-eval.git
cd perturbation-eval
```

已有可运行的 Chreode GEARS 环境时，激活该环境并安装本工具入口：

```bash
python -m pip install --no-deps -e .
python -m pip check
gears-eval --help
```

如果缺少评估依赖，用兼容现有 GEARS 环境的约束补齐：

```bash
python -m pip install -r requirements-gears-eval.txt
python -m pip check
```

独立评估环境可以使用 Python >= 3.10，直接安装全部评估依赖：

```bash
conda create -n perturbation-eval python=3.10 pip -y
conda activate perturbation-eval
python -m pip install -e .
```

安装后提供 `gears-eval` 和 `perturbation-eval` 两个命令，也保留原脚本和 `python -m evaluate` 方式。仓库可以克隆到任意目录，不要求父目录存在 Chreode 或 Lingshu-Cell 仓库。

## GEARS：只提供 Norman 数据目录与已有结果目录

评估已经产生的 GEARS 预测，使用专用入口，命令只需要两个目录参数：

```bash
gears-eval \
  --data-dir /path/to/chreode/output/gears/data \
  --output-dir /path/to/gears/outputs
```

也可在仓库根目录运行 `python evaluate_gears.py` 或安装后运行 `python -m evaluate.evaluate_gears`，参数完全相同。

脚本自动读取 `--data-dir/norman/perturb_processed.h5ad`；如果参数直接指向 `norman` 子目录，则读取其中的 `perturb_processed.h5ad`。这是 [GEARS 的标准数据布局](https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py)。固定使用 `obs.condition`、真实 `ctrl` 对照、`var.gene_name`（不存在时使用 `var_names`），保留 GEARS 已处理的表达尺度。

`--output-dir` 同时是**已有预测所在目录**和**本次评估输出目录**。自动优先读取 `predictions.npz`；不存在时读取 Chreode `scripts/gears/run_norman_gears.py --save-test-res` 保存的 `test_res.pkl`。NPZ 使用其中的 `gene_names` 对齐；PKL 使用 Norman 数据的原始基因列顺序，要求其预测来自同一份 GEARS 数据。PKL 必须是可信的本地 GEARS 运行产物。

两个格式都按提供的真实 H5AD 计算 pseudobulk 和对照，不使用预测产物中嵌入的 `truth`。只评估预测包含的测试扰动；不重新生成 split。PKL 在临时目录转换后评分，不改写或新增预测文件。

四项得分打印到终端，并直接保存到指定的 `--output-dir`：

- `gears_evaluation_summary.json`：四项得分及评估配置。
- `gears_evaluation_summary.csv`：四项得分汇总行。
- `001_GEARS_conditions.csv`：逐扰动明细。
- `001_GEARS_shared_genes.txt`：评估使用的共同基因。

原有训练 `summary.json`、预测和模型文件保留。没有 `predictions.npz` 或 `test_res.pkl` 时明确报错；该入口只评分已有预测，不训练或重新推理，也不需要安装 PyTorch/GEARS。

## 使用现有 Chreode 生成 GEARS 预测

这一步使用已安装 PyTorch/PyG/GEARS 的原训练环境。将以下路径替换为实际位置：

```bash
python /path/to/chreode/scripts/gears/run_norman_gears.py \
  --data-dir /path/to/chreode/output/gears/data \
  --output-dir /path/to/gears/outputs \
  --data-name norman \
  --split simulation \
  --seed 1 \
  --epochs 20 \
  --device cuda:0 \
  --save-test-res
```

`--save-test-res` 保存本工具需要的 `test_res.pkl`。已有预测时可以跳过训练，直接运行 `gears-eval`。本仓库不包含 GEARS 训练代码，也不会自动下载数据或模型。

## 安装与运行

安装后运行通用评估命令：

```bash
perturbation-eval \
  --pred-path /path/to/predictions \
  --truth-path /path/to/datasets \
  --output-dir /path/to/evaluation
```

也可以从仓库根目录直接运行脚本：

```bash
python run.py \
  --pred-path /path/to/predictions \
  --truth-path /path/to/datasets
```

两个路径都可为单个文件。目录模式递归寻找预测 `.h5ad` / `.npz` 与真实 `.h5ad`，依次按**相对路径一致（NPZ 改为 H5AD 后缀）、唯一同名文件、真实路径下只有一个 H5AD**配对。忽略隐藏目录及本次输出目录。无法唯一匹配时退出并提示使用清单，不按文件遍历顺序猜测。

如果同一输出目录保留了 `merged.h5ad`、`merged_with_control.h5ad` 和每个任务的生成分片，请用清单选定要评估的最终文件，避免重复评估。自动模式会评估发现的每个预测文件。

默认输出位置为 `<预测目录>/evaluation`，或 `<预测文件所在目录>/evaluation`。

## 输入契约

**H5AD**：`X` 是细胞 × 基因的表达矩阵，可以为稀疏或稠密矩阵；`obs` 保存扰动身份和背景标签；`var_names` 或 `var` 中的名称列标识基因。预测无需与真实数据逐细胞配对，两者细胞数可以不同。表达必须是有限数值，标签不可缺失。使用其他 layer 时，请先把评估表达写入 `X`。

扰动列自动识别 `target_gene`、`condition`、`cytokine`、`drug__dose`、`perturbation`、`perturbation_name`，也可通过 `--pert-col` 指定。**一个身份必须完整区分单基因/组合基因、药物及剂量等条件**。如果药物和剂量分开存放，请先构造 `drug__dose` 之类的完整身份列，并指定 `--pert-col drug__dose`。不自动重排组合基因、不修改大小写、不映射扰动名称。

同时存在多个候选扰动列、且它们把细胞划分成不同条件时，自动模式退出，要求 `--pert-col` 明确选择完整身份；例如 `target_gene=A` 不能取代区分 `A_dose1/A_dose2` 的 `condition`。

**预测 NPZ**：兼容 Chreode/GEARS 导出的数组：

```python
np.savez_compressed(
    "predictions.npz",
    pred=predicted_expression,       # [细胞或 pseudobulk 行数, 基因数]
    conditions=condition_names,      # 与 pred 行一一对应的字符串数组
    gene_names=gene_names,           # 与 pred 列一一对应的字符串数组
)
```

可以额外保存 `cell_type`、`donor`、`timepoint` 等背景数组，每个数组长度须等于行数。相同背景与扰动的行取算术平均；每个扰动已有一个 pseudobulk 时可直接使用。NPZ 中即使包含 `truth`，也**只使用 `--truth-path` 指定的真实数据**。使用字符串数组，避免 object/pickle 数组；通用入口不读取 `.pkl`，上述 GEARS 专用入口支持可信的 `test_res.pkl`。NPZ 整体读入内存，H5AD 的 `X` 分块读取。

## 对照、背景与测试范围

真实数据中必须存在匹配背景的未扰动对照。自动识别 `ctrl`、`control`、`non-targeting`、`non_targeting`、`non-perturbation`、`PBS`、`DMSO`、`DMSO_TF__0.0 uM`、`vehicle`、`NTC`（识别时忽略大小写，之后使用原始标签）。多个候选标签时必须显式指定 `--control-label`。

真实测试文件不含对照时，提供单独的**真实对照文件**：

```bash
perturbation-eval \
  --pred-path /path/to/predictions/norman/gears_predictions.npz \
  --truth-path /path/to/datasets/norman/test.h5ad \
  --control-path /path/to/datasets/norman/control.h5ad \
  --pert-col condition --control-label ctrl
```

预测文件不需要包含控制细胞；即使包含，预测对照也不参与计算。所有效应均减去同一个匹配的**真实**对照 pseudobulk。

自动背景分组识别细胞类型、细胞系、供体、时间列：`cell_type/celltype/cell_type_name/celltype_name`、`cell_line_id/cell_line/cell_line_name`、`donor/donor_id/donor_name`、`timepoint/time/time_point`。自动模式可对齐同族列名，但标签值必须一致。某个输入缺少背景列时，仅在该背景全局只有一个取值时补齐；多个取值时退出，避免把不同供体或细胞类型混合。

通过 `--group-cols cell_type,donor` 明确分组；`--group-cols ""` 明确合并所有背景。同背景的真实未扰动细胞提供对照，PDS 也只比较同背景的扰动。剂量默认不作背景分组，应编码在扰动身份中。时间默认需要同时间对照；不适用时显式调整 `--group-cols`。

评估范围由预测中非对照的“背景 × 扰动”决定。缺少对应真实样本或匹配对照会退出；真实数据中未被预测的条件记录在 `summary.json` 的 `unpredicted_truth_conditions`。PDS 候选集合也仅含本次预测的测试条件，请给各模型使用相同测试集合。

已有组合、未见组合、跨细胞系、细胞类型零样本和多药组合任务，分别在清单中指定对应的测试文件并用 `name` 标记。评估器不根据训练结果推断划分，也不把训练条件自动并入测试。

## 表达尺度与基因对齐

默认 `--preprocess none`，保留输入表达值；适用于 Chreode/GEARS 已经变换到相同尺度的输出和真值。原始 counts 使用：

```bash
perturbation-eval \
  --pred-path /path/to/predictions/vcc/merged.h5ad \
  --truth-path /path/to/datasets/vcc/test.h5ad \
  --pert-col target_gene --control-label non-targeting \
  --preprocess log1p --target-sum 10000
```

`log1p` 在**每个细胞、每个输入文件的完整原始基因面板**上先归一化总量到 10,000，再 log1p，之后对齐共同基因并取组均值。预测和真值需同为 counts，并建议使用相同原始面板；零总量细胞保留为零，负数 counts 退出。已经 log-normalized 的数据不要再次使用 `log1p`。若一方是 counts、另一方已经归一化，请先统一尺度，再用 `none`。仅有 pseudobulk 的 NPZ 不宜作细胞级归一化。

从 `var_names`、`gene_name`、`gene_names`、`gene_symbol`、`feature_name`、`symbol` 中选择共同基因最多的有效名称组合，按真实基因顺序对齐。可用 `--gene-col` 指定预测名称列、`--truth-gene-col` 指定真实/独立对照名称列；`var_names` 表示索引。自动模式跳过缺失/重复的候选注释；显式指定的名称列存在缺失/重复时退出，需要先解决映射碰撞。至少需要两个共同基因。不自动补零，不自动把 Ensembl ID 转成 symbol；名称列仍不兼容时需先完成映射。

最大共同基因数相同、但对应列的对齐不同，也会退出并要求显式名称列，避免把恰好同名的数字索引当作同一个基因。

H5AD 只读取 obs/var 元数据并分块聚合 X，不加载 layers、obsm、uns 或完整表达矩阵。默认 `--chunk-size 2048`，内存有限时可减小到 256。

## 多数据集的显式清单

大目录中通常有多个训练/验证/测试文件，仅凭两个根路径无法推断它们的对应关系。复制并修改 [examples/manifest.json](examples/manifest.json)，其中的文件名仅作示例，需替换为实际路径，预处理选项需符合真实表达尺度：

```bash
perturbation-eval \
  --pred-path /path/to/predictions \
  --truth-path /path/to/datasets \
  --manifest examples/manifest.json \
  --output-dir /path/to/evaluation
```

清单 `prediction` 相对预测根目录，`truth` 和可选 `control_path` 相对真实根目录；也接受绝对路径。每项可覆盖 `pert_col`、`control_label`、`group_cols`、`gene_col`、`truth_gene_col`、`preprocess`、`target_sum`、`chunk_size`、`top_k`。`group_cols: null` 表示自动，`[]` 表示合并背景。各项 `name` 必须唯一。全局命令行参数作为未覆盖字段的默认值。全部数据成功评估后才保存本次结果。

## 指标口径

设真实、预测和真实对照 pseudobulk 为 `p[t]`、`pred[t]`、`c[t]`；带符号效应 `l[t]=p[t]-c[t]`、`pred_l[t]=pred[t]-c[t]`。

| 输出键 | 计算 | 方向 |
| --- | --- | --- |
| `de20_mse` | 每个扰动的真实 Top20 基因上 `mean((pred_l-l)^2)`，再对扰动等权平均 | 越低越好 |
| `de20_delta_r` | 每个扰动真实 Top20 基因上的带符号效应 Pearson，再等权平均 | 越高越好 |
| `pearson_delta` | 所有被评估背景、扰动、共同基因的绝对效应展开后计算一次 Pearson | 越高越好 |
| `pds` | 同背景内 `d[t,j]=sum(abs(pred[t]-p[j]))`，`r[t]=count(d[t,j]<d[t,t])`，`1-2*sum(r)/T^2` | 越高越好 |

DE20 选择沿用本地 `Chreode/src/cellworldmodel/foundation/gears_shared_eval.py`：在共同基因中按真实 `abs(l)` 排序，**不是 Wilcoxon/p 值显著性排序**，也不读取预测 DE 排名。相同效应时按共同基因顺序选择。默认 K=20，不足 20 时采用实际共同基因数并在明细中记录 `de_k`。`--top-k` 可用于检查其他 K，但默认结果才是 DE20。

Pearson 常量向量按 Chreode 返回 0；同时输出 `undefined_correlations` 数量和逐扰动标记，便于识别退化输入。

多背景时每个背景独立计算 PDS，整体按被评估扰动数加权；等价于平均逐扰动的 `1-2*r[t]/T_group`。只有一个扰动的背景 PDS 固定为 1，命令行会提示其没有区分信息。严格等距不计“更近”。本协议的 PDS 可以为负数（单背景下限 `-1+2/T`），不截断、不做基线归一化。

Lingshu 的外部评估脚本调用 cell-eval。核对版本的上游 `pearson_delta` 使用逐扰动的**带符号**效应相关；`discrimination_score` 默认去掉靶基因，按排序位置使用 `1-rank/T`。本工具采用上述定义：Pearson-Δ 用整体**绝对**效应，PDS 包含靶基因、严格计数并使用系数 2。因此不保证与直接运行上游 `cell-eval` 的同名列一致。

## 输出与验证

- `summary.json`：每个文件对的四项得分、路径、实际标签/尺度/分组、基因覆盖、遗漏条件和退化相关计数。
- `summary.csv`：每个评估一行，便于横向比较；不把不同数据集的 Pearson 再平均冒充总体 Pearson。
- `001_<name>_conditions.csv`：每个背景 × 扰动的 DE20 得分、DE 基因、PDS 更近数量、候选数量和各输入行数。NPZ 行数可能是 pseudobulk 数，而非底层原始细胞数。
- `001_<name>_shared_genes.txt`：实际参与评估的共同基因及顺序。

运行测试：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m evaluate --help
```

测试包括手算公式、符号/绝对效应差异、平局 PDS、pseudobulk 均值、基因乱序、稀疏矩阵、背景/对照、NPZ 忽略嵌入真值、归一化、目录歧义和清单端到端输出。

参考：[Chreode 参考实现](https://github.com/MufanQiu/Chreode/blob/b3cc517276dadb8295776973367d4bfb2bb4894e/src/cellworldmodel/foundation/gears_shared_eval.py)；[Lingshu-Cell 外部评估封装](https://github.com/alibaba-damo-academy/Lingshu-Cell/blob/0cbde2feb6639f9411399794cf5e1c4831eb6c93/src/modules/evaluators/external/vcc_score.py)；[cell-eval 指标实现](https://github.com/ArcInstitute/cell-eval/blob/main/src/cell_eval/metrics/_anndata.py)（核对日期：2026-10-12）。本工具的四项公式和实现约定以本 README 为准；不保证等同于上游其他版本的同名指标。第三方数据、模型和代码的使用遵循各自的许可。
