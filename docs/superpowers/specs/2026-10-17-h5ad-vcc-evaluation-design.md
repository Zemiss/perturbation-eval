> 历史设计记录：当前四路径 CLI、固定参考面板和统一 lognorm 协议以根目录 README.md 为准；GEARS PKL 已移至 baseline/gears/export.py 转换入口。

# H5AD 四路径 VCC 六指标评估设计

## 目标与接口

用户提供真值 H5AD、模型预测 H5AD、独立真实对照 H5AD、输出目录，即可计算仓库现有的六项原始指标。预测格式以本地 Lingshu-Cell 的 VCC 推理输出为依据。用户于 2026-10-17 确认将对照改为第四个独立参数，不再从真值或预测中自动选择对照来源。

```bash
perturbation-eval --truth-path /data/truth.h5ad --pred-path /data/merged.h5ad --control-path /data/control.h5ad --output-dir /data/results
```

推荐在现有命令中增加 H5AD 输入支持，保留 GEARS PKL 路径及其 lognorm 算法。替代方案是完全替换 GEARS 接口，或者要求用户预先把 H5AD 转成 PKL；前者破坏现有用途，后者增加一次转换且 PKL 中的配对假设不适合生成细胞。

## 已查明的 Lingshu-Cell VCC 数据约定

- `src/modules/tokenizers/dense_gene1tok_tokenizer.py` 的 `decode` 将非负整数 counts 写入稀疏 `X`，使用 `var_names` 存储基因符号。
- 扰动列为 `obs.target_gene`，VCC 对照标签为 `non-targeting`。
- `workflows/infer/main.py` 合并推理批次为 `merged.h5ad`；VCC 条件表只列扰动，所以该文件不含真实对照。
- `scripts/run_infer_and_eval-vcc_perturbation.sh` 从独立的 `CONTROL_H5AD` 提取真实对照，追加至生成细胞，保存为 `merged_with_control.h5ad`。
- `configs/eval/vcc.yaml` 使用 `merged_with_control.h5ad` 作为评估预测输入。
- 真值由环境变量 `VCC_TEST_VALIDATION_H5AD` 指定。仓库不包含 VCC 真值或对照 H5AD，无法核实用户实际真值文件是否含对照。

## 输入及对照规则

1. H5AD 路径读取 `X`，按 counts 校验非负、有限值和整数；不自动推断归一化尺度，不把 counts 送入现有 GEARS lognorm 分支。
2. 优先读取 `obs.target_gene`，兼容明确的 `obs.condition`；缺失或冲突时给出错误。标准化已支持的对照标签，保留扰动身份。
3. 按基因名称对齐，允许基因顺序不同。优先匹配完整 `var_names`；必要时使用文件自身的 `var.gene_name` 注释。预测的每个基因都须在真值中唯一匹配；真值的多余基因不参与评估并记录数量。无法映射或匹配不唯一时退出。
4. 按预测中出现的扰动选择真值细胞，报告真值中未预测的扰动；预测有而真值没有的扰动报错。生成细胞无需与真实细胞按行配对，细胞数量可以不同。
5. `--control-path` 是 H5AD 模式的必需参数。始终从此文件中提取真实对照，忽略真值或预测中已附带的对照。默认使用 `target_gene=non-targeting` 或 `condition=ctrl`，兼容 `gene_target`、`target` 及 `non-perturbation` 标签；标签列有冲突则报错。对照专用文件可没有扰动标签列，此时全部行视为用户提供的真实对照。
6. 独立对照文件没有可用对照时退出，不能用扰动细胞或零向量代替。输入必须来自同一实验背景，提供的对照必须是实验测到的细胞；程序检查已提供的 `cell_type` 元数据，不能仅凭数值证明实验来源。
7. 初版范围为 Lingshu VCC 单靶基因扰动、单一细胞背景；不混合多种细胞背景评分。GEARS 现有组合扰动路径保持现有行为。

## 计算与输出

H5AD counts 分支使用固定版本 `cell-eval2==0.18.0` 的 `vcc2026` 配置与公共计算、聚合函数，固定 CPU/pdex，使用同一真实对照比较预测和真值。复用原库的 counts 归一化、噪声修正、靶基因排除、DE 过滤及 FDR 规则，不将现有 lognorm 表达误差公式直接套用到 counts。

结果仍为 `summary.csv`、`summary.json`，包含现有六个指标：

- `pds_cosine`
- `expr_mse_unbiased_capped_norm`
- `de_wilcoxon_direction_fidelity_yield_raw`
- `de_wilcoxon_direction_reach_raw`
- `de_wilcoxon_sig_jaccard`
- `de_wilcoxon_lfc_nmae`

JSON 保存输入路径、实际 counts/lognorm 协议、对照来源、细胞数、扰动数、基因数、基因对齐和筛选记录、依赖版本、完整配置及未定义指标状态。不生成模型预测 H5AD，不计算需要额外 baseline/replicate 数据的校准总分。

## 实施与验收

1. 新增 `h5ad.py` 负责 H5AD 加载与对齐，在 `run.py` 按预测扩展名选择加载和计算路径；验证三个真实临时 H5AD 可用四个路径运行。
2. 在 `metrics.py` 增加 counts 公共 API 路径；验证六项结果与同版本 cell-eval2 在独立构造的相同输入、相同配置下的结果一致。
3. 覆盖独立对照文件、忽略真值和预测附带的对照、基因乱序、稀疏 counts、不同细胞数量，以及缺对照、缺基因、缺扰动、负数和非整数 counts 等拒绝情形。
4. 更新 README 和命令行帮助，明确该命令输出六项原始指标，以及独立对照的来源契约。
5. 现有 GEARS 回归测试全部通过；本次检查基线为 8 项通过。只改 evaluate，不修改 Lingshu-Cell 或 cell-eval2。

验收限制：本机只有参考仓库代码，没有用户实际的 VCC 推理结果和真值；合成 H5AD 验证不能表述成已完成真实数据评分。
