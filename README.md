# 单细胞扰动预测评估：VCC2026 六项指标

从 **0.3.0** 起，默认入口使用 `cell-eval2==0.18.0` 的 **VCC2026 counts 配置**。提供预测、真值、输出三个路径，计算六项原始指标。直接调用原库的计算及汇总函数，不重写公式，不沿用旧版 DE20 或 PDS 定义。

## 安装

使用独立的 Python 3.11 环境；评估环境不需要安装 GEARS。已有 GEARS 训练环境可以继续负责生成预测。

```bash
conda create -n perturbation-eval python=3.11 pip -y
conda activate perturbation-eval
git clone https://github.com/Zemiss/perturbation-eval.git
cd perturbation-eval
python -m pip install -e .
perturbation-eval --help
```

已克隆仓库时，在仓库目录执行 `git pull --ff-only`，再在上述 Python 3.11 环境执行 `python -m pip install -e .` 更新。默认保留原 preset 的设备和 DE 后端自动选择。基础安装提供 `pdex==0.3.0`，无 CUDA 时使用 CPU；若检测到 CUDA 却缺少原库要求的 GPU DE 引擎，原库会报错。显式增加 `--device cpu` 可固定使用 CPU/pdex，便于不同模型使用同一后端比较。

## 运行：三个必填路径

```bash
perturbation-eval \
  --pred-path /data/outputs/pred_counts.h5ad \
  --truth-path /data/dataset/truth_counts.h5ad \
  --output-dir /data/evaluate/results
```

`--pred-path` 和 `--truth-path` 支持 H5AD 文件，或递归后**恰好包含一个 H5AD** 的目录。多个文件时必须明确指定文件，避免误把训练集、验证集或其他模型产物加入评分。

也可以在仓库目录运行 `python run.py`、`python evaluate_gears.py`，或安装后运行 `python -m evaluate`、`gears-eval`；它们使用相同的六指标 counts 入口。

**迁移注意：旧 GEARS 的归一化 `test_res.pkl`、`predictions.npz` 和 `perturb_processed.h5ad` 通常不满足这个输入要求。** 新入口不读取 PKL/NPZ，不自动还原 counts、不取整、不截断负值。即使输入 H5AD，也必须确认 `X` 确实保存原始 counts。数值校验无法判断一个已被人为取整的矩阵是否来自真实 counts。

## 输入契约

每对文件表示一个测试背景；不同细胞系、供体或时间背景分别运行。

- `X` 为细胞 × 基因的原始、未变换 UMI counts，支持稀疏和稠密矩阵；每个数有限、非负且为整数，每个细胞总 counts 不超过 `1,000,000`。整数可以用浮点 dtype 存储。
- `var_names` 为非空、唯一的基因名；两侧基因数量、名称和顺序必须完全相同。不取共享基因交集、不补零、不自动改名或排序。
- `obs.target` 保存完整扰动身份，默认对照标签为 `non-targeting`。两侧扰动集合完全一致，均包含对照；各扰动细胞数可以不同。
- 保留细胞级分布，不能先聚合为每个扰动一个均值后假装是单细胞预测。噪声修正和 Wilcoxon 检验需要细胞级输入。
- 本工具不推断测试划分、不将训练细胞追加到真值，也不从 H5AD 的 `raw` 或 layers 猜测 counts。

采用其他标签列或对照名称时：

```bash
perturbation-eval \
  --pred-path /data/outputs/pred_counts.h5ad \
  --truth-path /data/dataset/truth_counts.h5ad \
  --output-dir /data/evaluate/results \
  --pert-col condition --control-label ctrl
```

指标需要排除靶基因。扰动标签最好直接使用与 `var_names` 一致的单靶基因名。如果标签为 guide/construct ID，用 `--target-gene-map target_genes.json` 明确映射：

```json
{"ADNP-guide1": "ADNP", "BCL2-guide2": "BCL2"}
```

映射沿用原库的一扰动对一基因语义。Norman 的 `A+B` 双基因扰动不属于该单靶基因规范；本入口不会自动把它改成其中一个基因。药物、组合扰动或任意标签不能在没有定义靶基因排除规则时宣称等同于官方单靶基因评估。

## 六项指标与固定口径

| 输出键 | 含义 | 原始值方向 |
| --- | --- | --- |
| `pds_cosine` | 真实对照效应下的余弦检索排名 | ↑ |
| `expr_mse_unbiased_capped_norm` | 经采样噪声修正并按真实效应归一化的表达误差 | ↓ |
| `de_wilcoxon_direction_fidelity_yield_raw` | 差异表达方向的准确性与覆盖 | ↑ |
| `de_wilcoxon_direction_reach_raw` | 在指定方向纯度要求下的覆盖深度 | ↑ |
| `de_wilcoxon_sig_jaccard` | 显著响应基因集合的 Jaccard | ↑ |
| `de_wilcoxon_lfc_nmae` | 归一化的 log fold-change 绝对误差 | ↓ |

使用原 `vcc2026` preset：真实对照用于两侧效应及 DE 检验；表达/PDS 使用 counts 汇总后归一化到 `50,000` 再 log1p；DE 的细胞归一化总量为 `1,000,000`；控制细胞 CPM 阈值为 5；Wilcoxon、每扰动 BH-FDR 0.05；PDS 排除整个面板的靶基因并使用 midrank 和 `n-1` 分母。其余细节以原库和输出的完整配置为准。

表达误差的最终汇总使用原库的**分子之和／分母之和**，不能平均逐扰动比值。该指标没有逐扰动原始值；原库逐扰动表中保留它的分子、分母等诊断项，六项最终值从原聚合函数取得。

**输出是六项原始指标，不是 VCC 官方总分。** 官方 baseline/replicate 校准需要与当前真值匹配的参照数据；本工具不会使用其他背景的锚点或简单平均六项原始值。对 Norman 等自有 counts 数据，这是相同配置下的六项评估，不能直接冒充官方比赛参考数据上的成绩。

## 输出

- `summary.json`：六项原始指标、完整配置、实际输入路径、版本及各指标有效样本统计。原库未定义的值写为 `null` 并标记 `undefined`。
- `summary.csv`：一行六项原始指标。
- `results.csv`：原库逐扰动长表，包含诊断项。
- `agg_results.csv`：原库完整聚合表，包含四项诊断指标，可供进一步核对。
- `run_params.yaml`：完整配置。
- `run_meta.json`：原库生成的严格真值指纹、运行后端和依赖信息，供后续原库评分流程核验。

真实面板在采样修正后没有可测效应、导致表达误差分母非正时，沿用原库报错，不伪造六项完整结果。

## H20 / 可选 GPU

CPU 路径不需要 PyTorch/CUDA，可用 `--device cpu` 固定。需要 GPU 时，在独立评估环境中安装原库 GPU 依赖，再显式指定：

```bash
python -m pip install 'cell-eval2[gpu,gpudge]==0.18.0'
perturbation-eval \
  --pred-path /data/outputs/pred_counts.h5ad \
  --truth-path /data/dataset/truth_counts.h5ad \
  --output-dir /data/evaluate/results \
  --device cuda
```

此时 DE 使用原 `gpudge` 后端，缺少 GPU 依赖时不会静默切换到 CPU。它会安装自己的 PyTorch 依赖，须与服务器驱动匹配。比较不同模型时固定后端和环境；CPU/GPU 的 DE 数值不保证逐位一致。本仓库的自动测试验证 CPU 路径。

## 旧版四指标与验证

需要复核 0.2.0 的 Chreode GEARS 实验时，使用明确的旧协议入口 `gears-chreode-eval`。其他旧四指标任务使用 `perturbation-eval-legacy`。见 [旧协议说明](docs/legacy.md)。默认命令不再输出旧四项指标。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

测试包含直接对照原 `compute_metrics` 和 `aggregate_metrics` 的六项结果、稀疏/稠密 counts、非法输入、基因轴/标签检查、不同标签与靶基因映射、命令行输出，以及旧协议回归。

版本依据：用户提供的 `cell-eval2 0.18.0` 源码目录与固定 PyPI 版本的 51 个 Python/配置文件一致（核对日期：2026-10-16）。参考 [cell-eval2](https://github.com/ArcInstitute/cell-eval2)；公式及参数详见该版本的 `docs/vcc2026_metrics/` 和 `configs/vcc2026.yaml`。本仓库不包含真实实验数据、预测或权重。
