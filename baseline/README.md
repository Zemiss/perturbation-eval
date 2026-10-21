# VCC H1 基线

在评估仓库根目录运行。Chreode 与 Lingshu-Cell 共用同一份 `truth.h5ad`、`control.h5ad`、`panel.txt` 和 `conditions.csv`，最后用 `baseline.compare` 汇总六项指标。

```text
baseline/
├── common.py              # 共享条件表、基因映射、H5AD 输入输出
├── compare.py             # 两模型汇总
├── prepare_reference.py   # 一次性整理共享 truth/control/panel/conditions
├── lingshucell/infer.py   # 官方 VCC 推理
├── chreode/
│   ├── train.py           # VCC 后训练
│   ├── infer.py           # 同一 VCC 面板推理
│   └── model.py           # 外接 CRISPRi 适配头
└── README.md
```

## 路径

| 内容 | 默认位置 |
| --- | --- |
| 原始 VCC 数据 | `/data/dataset/lingshucell/vcc/` |
| 共享参考 | `/data/dataset/lingshucell/vcc/evaluation/` |
| Lingshu 原仓库 | `/data/xiebo/Lingshu-Cell/` |
| Lingshu 权重、ESM、clue | `/data/xiebo/models/lingshucell/` |
| Chreode 原仓库 | `/data/xiebo/Chreode/` |
| Chreode 官方 release | `/data/xiebo/models/chreode/pretrained/` |
| Chreode VCC 后训练权重 | `/data/xiebo/models/chreode/vcc/model.pt` |
| 预测 | `/data/xiebo/outputs/{lingshucell,chreode}/prediction.h5ad` |
| 运行记录 | `/data/xiebo/runs/{lingshucell,chreode}/` |
| 汇总结果 | `/data/xiebo/results/comparison/` |

VCC 下载保留官方 split 子目录：

```text
/data/dataset/lingshucell/vcc/
├── gene_names.csv
├── train/adata_Training.h5ad
├── validation/adata_Validation.h5ad
├── test/adata_Test.h5ad
└── evaluation/                 # 由 prepare_reference 生成
    ├── truth.h5ad
    ├── control.h5ad
    ├── panel.txt
    └── conditions.csv
```

## 需下载的资产

### Lingshu-Cell

```bash
cd /data/xiebo
git clone https://github.com/alibaba-damo-academy/Lingshu-Cell.git
```

从 HuggingFace [`bibona/lingshu-cell`](https://huggingface.co/bibona/lingshu-cell) 下载：

```text
/data/xiebo/models/lingshucell/
├── vcc_perturbation_85M.pth
├── gene_embeddings.npz
└── whole_dataset_stringent_clue.csv
```

仓库自带 `data/vcc/pert_counts_Test.csv`、`data/vcc/gene_id_name.csv` 和 18,080 基因词表。

### Chreode

```bash
cd /data/xiebo
git clone https://github.com/MufanQiu/Chreode.git

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

仓库自带 `artifacts/gene_vocab.parquet`（16,485 基因）。Chreode VCC adapter 复用 Lingshu 的 `gene_embeddings.npz`。

## 共用参考

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

- panel = `gene_names.csv` ∩ Chreode 原生词表 ∩ truth/control；两个模型都必须覆盖。
- `--max-cells-per-pert 400` 写入 `conditions.csv`；Chreode/Lingshu 都按这份表生成相同数量的细胞。
- 如果 test 文件没有真实对照，把 `--control-path` 换成含 `non-targeting` 的现有文件，并在报告中注明来源。
- 已有的 `evaluation/` 不会覆盖；要重建请换输出目录。

## Chreode

```bash
python -m baseline.chreode.train \
  --train-data /data/dataset/lingshucell/vcc/train/adata_Training.h5ad \
  --val-data /data/dataset/lingshucell/vcc/validation/adata_Validation.h5ad \
  --conditions /data/dataset/lingshucell/vcc/evaluation/conditions.csv \
  --release-dir /data/xiebo/models/chreode/pretrained \
  --checkpoint /data/xiebo/models/chreode/vcc/model.pt \
  --run-dir /data/xiebo/runs/chreode/vcc \
  --device cuda

python -m baseline.chreode.infer \
  --control-data /data/dataset/lingshucell/vcc/train/adata_Training.h5ad \
  --conditions /data/dataset/lingshucell/vcc/evaluation/conditions.csv \
  --checkpoint /data/xiebo/models/chreode/vcc/model.pt \
  --output-dir /data/xiebo/outputs/chreode \
  --run-dir /data/xiebo/runs/chreode/vcc_infer \
  --device cuda
```

训练时冻结官方 VAE，默认后训练 dynamics 与外接 surgery adapter，以验证集 Sinkhorn divergence + MMD 选 checkpoint；`--freeze-base` 可只训练适配头。不会读取测试扰动表达训练或选模型。推理会核对 `conditions_sha256`，训练/推理必须用同一份 `conditions.csv`。

## Lingshu

```bash
python -m baseline.lingshucell.infer \
  --assets /data/xiebo/models/lingshucell \
  --conditions /data/dataset/lingshucell/vcc/evaluation/conditions.csv \
  --output-dir /data/xiebo/outputs/lingshucell \
  --run-dir /data/xiebo/runs/lingshucell/vcc_infer
```

调用原仓库 `infer/legacy_vcc`：steps=3、CFG=2、non-targeting guidance、clue 注入；`--no-clue` 可关闭 clue 先验做消融。输出保留原始 18,080 基因 counts；评估时按共享 panel 取子集。

## 汇总

```bash
python -m baseline.compare \
  --truth-path /data/dataset/lingshucell/vcc/evaluation/truth.h5ad \
  --control-path /data/dataset/lingshucell/vcc/evaluation/control.h5ad \
  --pred lingshucell=/data/xiebo/outputs/lingshucell/prediction.h5ad \
  --pred chreode=/data/xiebo/outputs/chreode/prediction.h5ad \
  --output-dir /data/xiebo/results/comparison
```

输出 `comparison.csv`、`comparison.json`、`comparison.md`。`average_rank` 只是六项排名的描述性汇总；不要当成官方 VCC 总分。

## 旧 Norman 入口

`baseline.gears.infer` 和 `baseline.gears.export` 保留给已有 Norman 运行。Norman 与 VCC 是不同数据集、不同 split、不同扰动类型，**不参与**本次 Chreode vs Lingshu 的 VCC H1 对比。

## 限制

- Lingshu 是 generalist：H1 150 个训练靶点 + 外部细胞系 300 靶点；100 个 test 靶点在 H1 中留出，但在外部细胞系中出现过。Chreode 只用 H1 训练集。当前是 practical benchmark，不是等信息的架构对比。
- Chreode 的 VCC adapter 是本仓库新增方案，不是 Chreode 原论文已发布的 H1 实验。
- 六项指标是 vcc2026 的 lognorm 适配，不做 baseline/replicate 校准，不等同于官方 counts 成绩。

## 小规模检查

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

本地只做格式及小数据测试，完整训练和推理在服务器运行。
