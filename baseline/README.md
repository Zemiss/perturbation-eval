# VCC H1 基线

在评估仓库根目录运行以下命令。两个模型共用 Lingshu 的 `data/vcc/pert_counts_Test.csv`，也可通过相同的 `--conditions` 文件覆盖。默认完整面板为 100 个扰动、132,670 个生成细胞。

```text
baseline/
├── common.py             # 共享条件表、基因映射和 H5AD 输入输出
├── lingshucell/infer.py   # 原始 VCC 推理
├── chreode/
│   ├── train.py          # VCC 后训练
│   ├── infer.py          # 同一 VCC 面板推理
│   └── model.py          # 外接 CRISPRi 适配头
├── gears/
│   ├── infer.py          # GEARS 直接导出 H5AD
│   └── export.py         # 旧 PKL 转 H5AD
└── prepare_reference.py  # 一次性整理共享 VCC 真值/对照
```

## 路径

| 内容 | 默认位置 |
| --- | --- |
| 原始 VCC 数据 | `/data/dataset/lingshucell/vcc/` |
| Lingshu 原仓库 | `/data/xiebo/Lingshu-Cell/` |
| Chreode 原仓库 | `/data/xiebo/Chreode/` |
| Lingshu 权重、ESM、clue | `/data/xiebo/models/lingshucell/` |
| Chreode 官方 release | `/data/xiebo/models/chreode/pretrained/` |
| Chreode 后训练权重 | `/data/xiebo/models/chreode/vcc/model.pt` |
| Chreode 训练记录 | `/data/xiebo/runs/chreode/vcc/` |
| Lingshu 推理记录和中间文件 | `/data/xiebo/runs/lingshucell/vcc_infer/` |
| Chreode 推理记录和临时矩阵 | `/data/xiebo/runs/chreode/vcc_infer/` |
| Lingshu 预测 | `/data/xiebo/outputs/lingshucell/prediction.h5ad` |
| Chreode 预测 | `/data/xiebo/outputs/chreode/prediction.h5ad` |

VCC 2025 下载数据保留官方的 split 子目录。下面按 `2025/` 的内容直接放在 `vcc/` 下示意，其他下载文件可原样保留：

```text
/data/dataset/lingshucell/vcc/
├── gene_names.csv
├── train/
│   ├── adata_Training.h5ad
│   └── pert_counts_Training.csv
├── validation/
│   ├── adata_Validation.h5ad
│   └── pert_counts_Validation.csv
├── test/
│   ├── adata_Test.h5ad
│   └── pert_counts_Test.csv
└── evaluation/                 # 后续整理生成，不属于原始下载
    ├── truth.h5ad
    └── control.h5ad
```

如果下载时保留了 `2025/` 这一层，继续保留即可；训练/推理命令加 `--data-root /data/dataset/lingshucell/vcc/2025`，其他命令的输入路径也加上这一层。无需将原文件移动或改名为 `train.h5ad`、`val.h5ad`、`test.h5ad`。

原始文件清单没有独立的 `control.h5ad`。对照来自现有 H5AD 中的真实 `non-targeting` 细胞，推理/参考整理脚本会按标签筛选；评估用的独立 `evaluation/control.h5ad` 在整理后才生成。推理对照与评分对照各自使用固定来源，评分对照来源需结合服务器文件的标签和实验批次核实。

来源：[Arc 官方目录说明](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/virtual-cell-challenge/README.md)及[2025 对象清单](https://storage.googleapis.com/storage/v1/b/arc-institute-virtual-cell-atlas/o?prefix=virtual-cell-challenge%2F2025%2F&fields=items%28name%29)。这里描述的是官方布局，本地未连接服务器核实实际下载路径。

## Lingshu 推理

激活 Lingshu 推理环境，准备 `vcc_perturbation_85M.pth`、`gene_embeddings.npz`、`whole_dataset_stringent_clue.csv` 后运行：

```bash
cd /data/xiebo/evaluate
python -m baseline.lingshucell.infer --dry-run
python -m baseline.lingshucell.infer
```

调用原仓库 `infer/legacy_vcc`：steps=3、CFG=2、non-targeting guidance、clue 注入。保留原始 18,080 基因 counts；`prediction.h5ad` 硬链接到运行目录的 `inference/merged.h5ad`，跨文件系统时改用复制。分批文件、Hydra 配置、日志和条件表都放在 `--run-dir` 中，`--output-dir` 仅存最终 H5AD。相同配置中断后可用 `--resume`，改变配置应指定新的 `--run-dir` 和 `--output-dir`。

## Chreode 后训练和推理

激活安装了 Chreode 依赖和 PyTorch 的环境。先用官方工具下载固定版本 encoder、dynamics。下载使用 `/data/xiebo/` 下的临时缓存，命令结束后自动清理，无需保留独立的 `cache/` 目录：

```bash
(
  set -e
  VCC_HF_TMP=$(mktemp -d /data/xiebo/.hf-download.XXXXXX)
  trap 'rm -rf -- "$VCC_HF_TMP"' EXIT
  HF_HOME="$VCC_HF_TMP" HF_HUB_CACHE="$VCC_HF_TMP/hub" \
  PYTHONPATH=/data/xiebo/Chreode/src python -m cellworldmodel.script.download_release \
    --manifest /data/xiebo/Chreode/releases/iclr2027.json \
    --output /data/xiebo/models/chreode/pretrained \
    --model encoder --model dynamics
)
```

以下使用官方 split 内的文件，相对路径均从 VCC 数据目录解析。训练/验证需各有真实对照和扰动细胞、互不重叠的细胞 ID，且均不得包含预测条件表中的测试扰动。标签默认读取 `obs.target_gene`，存在 `obs.cell_type` 时仅取 H1。原始 counts 默认读取 `X`；若存在 counts layer，两个命令通过训练配置保持一致。服务器实际标签、对照及尺度仍需核实。

```bash
cd /data/xiebo/evaluate
python -m baseline.chreode.train \
  --train-data train/adata_Training.h5ad \
  --val-data validation/adata_Validation.h5ad

python -m baseline.chreode.infer --control-data train/adata_Training.h5ad
```

可用 `--condition-col`、`--cell-type-col`、`--layer counts` 显式适配训练数据。训练时冻结官方 VAE，默认后训练 dynamics 与外接 surgery adapter，以验证集的 Sinkhorn divergence + MMD 选择 checkpoint；`--freeze-base` 可仅训练适配头。不会读取测试扰动表达训练或选模型。

这是本仓库新增的 **Chreode VCC 适配方案**，并非 Chreode 原论文已发布的 H1 实验。CRISPRi 使用负向、modality=1；复用 Lingshu 配套 ESM 作为固定基因先验，并关闭可训练的基因 ID residual。完整 VCC 扰动轴与 Chreode 表达词表分开，因此表达词表之外的靶基因仍可作为条件。测试目标缺少 ESM 时会报错。

## 输出和共享参考数据

Lingshu 和 Chreode 的最终预测分别为 `outputs/lingshucell/prediction.h5ad`、`outputs/chreode/prediction.h5ad`。输出目录只存最终 H5AD；条件表、配置和中间文件放在各自的 `--run-dir` 中。

两份 H5AD 都是细胞 × 基因，使用 `obs.cell_type=H1`、`obs.target_gene` 和基因 symbol。Lingshu 保留 counts；Chreode 保留非负 lognorm 解码表达，记录负值截断数量，并按 VCC 顺序输出可映射的基因。

Chreode 的表达基因面板较小，原始真值也可能使用 ENSG ID。两边预测完成后，**只整理一次共享参考数据**。先将 `VCC_EVAL_CONTROL_SOURCE` 设置为经核实、包含评分用真实对照细胞的现有 H5AD 的绝对路径；如果测试文件包含合适的 `non-targeting` 细胞，也可以直接指向同一测试文件，无需先单独提取。

```bash
cd /data/xiebo/evaluate
python -m baseline.prepare_reference \
  --pred-path /data/xiebo/outputs/chreode/prediction.h5ad \
  --truth-path /data/dataset/lingshucell/vcc/test/adata_Test.h5ad \
  --control-path "${VCC_EVAL_CONTROL_SOURCE:?先设置经核实的真实对照来源路径}" \
  --output-dir /data/dataset/lingshucell/vcc/evaluation
```

脚本读取已有基因 ID 映射，选择相同 H1 测试扰动和真实对照，在预测、真值、对照的共同基因上生成 `truth.h5ad`、`control.h5ad`。此后这两份文件固定，Lingshu 和 Chreode 都使用它们；不会按各模型分别变更基因面板。默认读取原始 counts `X`，其他已声明尺度/layer 也会保留；需要直接指定真实数据中的 counts layer 时加 `--layer counts`。这一步不运行模型、不改原始数据或原仓库。

随后使用 [主 README 的四路径评估命令](../README.md#四路径评估)，只替换预测路径和结果目录。评估器仅依赖三份 H5AD；六指标采用统一 lognorm 协议。

## GEARS 直接输出 H5AD

在已有 GEARS 环境和保存模型的情况下，使用本仓库入口推理：

```bash
cd /data/xiebo/evaluate
python -m baseline.gears.infer \
  --gears-repo /data/xiebo/GEARS \
  --data-dir /data/dataset/gears \
  --data-name norman \
  --model-dir /data/xiebo/models/gears/norman
```

沿用现有官方 GEARS 的 `PertData`、拆分和保存模型接口，默认 `simulation`、seed=1。必须与该模型训练时的数据、拆分和 seed 相同，可用 `--split`、`--seed` 调整；这不是 GEARS 的 VCC 训练实现。直接写出：

```text
/data/xiebo/outputs/gears/prediction.h5ad
/data/dataset/gears/evaluation/truth.h5ad
/data/dataset/gears/evaluation/control.h5ad
/data/xiebo/runs/gears/infer/run.json
```

真值使用 GEARS 测试输出中的逐细胞 `truth`，对照使用其 processed 数据的真实 `ctrl` 细胞。三份文件都声明 lognorm 尺度；`GENE+ctrl`、`GENE1+GENE2` 保持原样，不做均值压缩。

已有可信 PKL 可以单独转换，无需再次推理或加载模型：

```bash
python -m baseline.gears.export \
  --pred-path /data/xiebo/runs/gears/旧运行/test_res.pkl \
  --data-path /data/dataset/gears/norman/perturb_processed.h5ad
```

`--data-path` 必须是同一次 GEARS 数据处理生成的 lognorm 文件；PKL 无 `gene_names` 时按该文件的完整基因顺序解释，需确认来源一致。导出位置与直接推理相同，也可用 `--output-dir`、`--reference-dir` 覆盖；已存在的 H5AD 不会覆盖。PKL 只在这个转换入口读取，统一评估器只读 H5AD。

## 小规模检查

先准备独立的小条件表，例如两个目标各 32 个细胞，将同一文件传给 Lingshu/Chreode 的 `--conditions`；Chreode 后训练时也必须用这份条件表声明留出的测试面板。训练/验证/对照的真实文件结构仍需在服务器核实。本地只做格式及小型测试，完整训练和推理在服务器运行。
