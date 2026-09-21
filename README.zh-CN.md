# Crane · Python 复现

[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)](docs/EXPERIMENTS.md)
[![Paper · MICRO 2025](https://img.shields.io/badge/paper-MICRO%202025-b31b1b)](https://doi.org/10.1145/3725843.3756023)

**使用 Python 独立复现 Crane 论文中的调度流程，研究分块加速器上的 DNN 推理、训练与层间调度。**

[English](README.md) · [简体中文](README.zh-CN.md) ·
[实验指南](docs/EXPERIMENTS.md) · [复现范围](docs/REPRODUCTION.md)

> **当前状态：研究原型。** 核心调度流程已实现，并通过小规模示例和测试验证。
> 现有网络实验使用代理工作负载数据。本项目是独立复现，论文原实现使用 C++；
> 本仓库尚未证明复现了论文报告的性能加速比或能量延迟积（EDP）结果。

## 对应论文

**Crane: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture**

Yu Gong、Lingyi Huang、Haodong Chang、Rongjian Liang、Cheng Yang、Zhexiang Tang、Jiang Hu、Bo Yuan。

**MICRO 2025，1250–1263 页。**

[论文 DOI / 出版页面](https://doi.org/10.1145/3725843.3756023) ·
[论文 PDF](include/crane_paper.pdf) · [引用信息](CITATION.cff)

论文研究的核心问题是：如何在多个计算 tile 之间安排神经网络各层的执行、数据保留和传输，
并共同考虑层融合、batch 拆分以及训练中的重计算。Crane 使用分层 block 和表格表示这些决策，
再通过优化搜索调度方案。本仓库用 NumPy 和 OR-Tools 的 SCIP 后端实现相关实验流程。

## 仓库能做什么

- 描述层级 DAG、层间依赖和分层 block，并进行链式 block 合并。
- 枚举子 batch，求解 **ScT（调度表）** 与 **MeT（SRAM / DRAM 内存表）**。
- 估算计算与访存的延迟、能耗和 EDP，选择候选调度方案。
- 探索递归调度、block 结构细化，以及训练的 **FW / BW1 / BW2** 阶段。
- 根据实验入口输出 JSON、文本、CSV 或可在浏览器中查看的 HTML 调度报告。

整体流程为：**层图与工作量 → block 与依赖 → 子 batch 候选 → ScT → MeT → 成本评估 → 调度报告**。
分层搜索可进一步反馈和细化 block 结构。

这里运行的是调度模型和成本估算；不需要实际执行神经网络、训练模型权重或使用 GPU。

## 快速开始

使用 **Python 3.11–3.13**。以下示例不需要数据集、GPU、外部求解服务或商业求解器许可证。
在仓库根目录执行：

```bash
git clone https://github.com/aHappend/Crane.git
cd Crane
python -m venv .venv
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装记录的依赖版本，检查环境并运行小型示例：

```bash
python -m pip install -r requirements.txt -c constraints.txt
python tools/doctor.py
python example/quickstart.py
```

入门示例是一个**人工构造的三层链**，实际调用 ScT 和 MeT 的 SCIP 求解器，关闭启发式 fallback。
终端会显示选中的子 batch、求解器名称和估算指标，并生成：

```text
outputs/experiments/quickstart_<时间戳>/
├── summary.json     # 工作负载、完整配置、依赖版本、Git 版本和结果
└── schedule.html    # 各状态、累计 ScT 和内存表
```

用浏览器打开 `schedule.html`。正常运行时两个求解器均显示为 `ortools-scip`。
这一步证明基本流程可运行；其中的估算值不代表论文实验结果。
不同求解器版本可能在多个同等目标值的方案中选出不同的调度表。

可用 `--output-dir <新目录>` 指定输出位置。为保留已有实验，输出目录必须尚不存在。

## 实验入口与适用范围

| 入口 | 用途 | 数据及解释 |
| --- | --- | --- |
| `example/quickstart.py` | 检查环境、理解完整流程 | 三层人工工作负载 |
| `example/run_official_nns_suite.py` | 比较 12 个网络家族的代理模型，启用 block 合并 | 手工填写的工作量，与 SET 网络定义关联 |
| `example/run_official_nns_layer_level.py` | 对同一批代理模型按节点调度，关闭合并 | 一个代理节点对应一个 block，并非自动导入完整网络 |
| `example/compare_transformer_granularity.py` | 比较 stage / layer 粒度，可选论文 §7.2 硬件配置 | 手工展开的 471 节点 Transformer 链和代理成本 |
| `example/run_transformer_training_repro.py` | 探索 FW / BW1 / BW2 和 §7.3 风格硬件配置 | 代理工作量，反向与重计算成本由缩放参数构造 |

详细命令、配置、输出解释及排错见 [实验指南](docs/EXPERIMENTS.md)。
471 节点 Transformer 等大型实验可能耗时较长，多数 MILP 没有求解时限。

`official_nns` 和 `strict_paper_mode` 是保留的历史名称，不能据此认定已经精确导入原网络、
严格等价实现全部数学模型或复现论文数值。[复现范围文档](docs/REPRODUCTION.md)逐项列出了论文到代码的对应关系、
已知近似和验证缺口，尤其说明了 EDP 目标的松弛处理、代理工作负载和分析式硬件成本模型。

## 代码导航

| 目录 | 职责 |
| --- | --- |
| `model/` | 层节点、DAG 解析与拓扑排序 |
| `scheduler/` | block、ScT / MeT 求解器、硬件参数 |
| `search/` | 候选搜索、分层优化、训练阶段组织 |
| `cost_model/` | 延迟与能耗计算 |
| `example/` | 实验入口、入门示例和 HTML 报告 |
| `tests/` | 回归测试与真实求解器集成测试 |
| `tools/` | 环境检查和维护工具 |
| `src/nns/`、`include/` | SET 网络参考定义、头文件和论文 |
| `docs/` | 架构、实验与复现边界文档 |
| `outputs/` | 历史参考资料，以及被 Git 忽略的新实验输出 |

修改实现前建议阅读 [架构与单位说明](docs/ARCHITECTURE.md)。
新实验写入 `outputs/experiments/`；历史 `outputs/runs/` 仅作为存档保留。
历史结果早于本次工作量重复统计修复，不能当作当前版本的数值基线，具体见 [变更记录](CHANGELOG.md)。

## 开发和验证

```bash
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m ruff check .
python -m pytest -q
```

[CI 模板](.github/ci-template.yml)覆盖 Linux 的 Python 3.11 / 3.13 和 Windows 的 Python 3.11，
检查 SCIP、调度约束、实验入口，并上传入门示例报告。目前模板尚未启用；将其提交为
`.github/workflows/ci.yml` 需要 GitHub 的工作流权限。自动测试统一在 `tests/`；`example/*_test.py` 是历史实验脚本。

贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，后续复现里程碑见
[待完成工作](docs/REPRODUCTION.md#remaining-work)。

## 引用与来源

研究方法请引用 Crane 原论文，BibTeX 见 [英文 README](README.md#citation-and-attribution)。
使用本仓库实现时，请同时记录 `aHappend/Crane`、具体 Git commit 和依赖环境。
本仓库复现代码、Crane 论文和 SET 参考资料的作者归属分别记录。

当前仓库尚未声明统一的软件许可证；论文及第三方文件的来源和各自许可状态见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
