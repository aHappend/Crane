# Crane · 复现与调度实验平台

[![CI](https://github.com/aHappend/Crane/actions/workflows/ci.yml/badge.svg)](https://github.com/aHappend/Crane/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)](docs/EXPERIMENTS.md)

**用 Python 独立实现 Crane 的层间调度方法，并提供真实网络、数学检查、可重复实验和交互式结果演示。**

[English](README.md) · [简体中文](README.zh-CN.md) ·
[实验结果](experiments/results/reproduction_20260921/REPORT.md) ·
[复现范围](docs/REPRODUCTION.md) · [数学检查](docs/MATHEMATICAL_AUDIT.md)

> 当前已具备完整的实验运行与证据保存流程。推理实验使用真实网络定义和原版 SET 的层内成本配置；
> 训练提供经过样本覆盖及容量检查的统一 cohort 参考策略，另保留实验性的 MILP 路径。
> 完整的布局/通信成本校准、论文全部图表与性能加速比，仍未被证明复现。

## 对应论文，以及与 SET 的关系

**Crane: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture**

Yu Gong、Lingyi Huang、Haodong Chang、Rongjian Liang、Cheng Yang、Zhexiang Tang、Jiang Hu、Bo Yuan。
**MICRO 2025，1250–1263 页。**

[论文 DOI](https://doi.org/10.1145/3725843.3756023) · [论文 PDF](include/crane_paper.pdf) · [引用信息](CITATION.cff)

SET 是 ISCA 2023 的另一套层间调度框架。Crane 论文用它校验成本模型，并将其作为推理对比基线。
本仓库从固定版本的 SET 导出真实网络定义和层内映射成本，同时单独运行原版 SET 搜索作为参考。
Python Crane 调度器是独立实现；原论文实现使用 C++。

## 本轮复现更新

- **真实网络：** 编译原始 C++ 网络对象，导出 16 个网络的形状、工作量和依赖。
  ResNet-50 为 72 个节点、87 条边；Transformer 为 471 个节点、661 条边，保留残差、分支及动态权重依赖。
- **调度数学：** 修正子 batch 工作量缩放、多父节点数据需求和样本区间统计。
  对固定成本的 ScT，提供精确整数乘积 MILP，以及适用条件下等价的顶点枚举化简。
- **分层执行：** 子调用恰好处理一个父子 batch，使用实际分配的 tile 和明确的内存预算。
  不再通过不相关状态索引的重采样和放松边界来拼接默认分层结果。
- **成本参考：** 随仓库提供 ResNet-50、VGG-19、GoogLeNet、Transformer cell 的原版 SET 层内成本配置。
- **训练参考：** 显式记录 FW、BW1、重计算与 BW2 的样本区间，验证每个样本完成一次反向计算及容量约束。
- **实验与演示：** 一键实验矩阵、进程时限、原始结果、来源哈希、科学绘图和离线交互 demo。

## 快速开始

使用 Python 3.11–3.13，在仓库根目录执行：

```bash
git clone https://github.com/aHappend/Crane.git
cd Crane
python -m venv .venv
```

Linux/macOS 使用 `source .venv/bin/activate` 激活环境；Windows PowerShell 使用
`.\.venv\Scripts\Activate.ps1`。然后运行：

```bash
python -m pip install -r requirements.txt -c constraints.txt
python tools/doctor.py
python example/quickstart.py
```

入门示例用一个小型人工网络调用真实 SCIP 求解器，输出 JSON 和 HTML 调度报告。
这些是调度与成本模型实验，不需要 GPU、训练权重或下载数据集。

## 交互式 demo

克隆后直接用浏览器打开 **[docs/demo/index.html](docs/demo/index.html)**，无需联网或启动服务。
也可以运行：

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs/demo
```

在运行该命令的电脑打开 `http://127.0.0.1:8000`。如果代码在 SSH 服务器上，
可在自己的电脑运行 `ssh -L 8000:127.0.0.1:8000 <服务器别名>` 后访问同一地址。

Demo 可以选择已保存的网络、batch 和调度方案，播放 ScT 状态、查看 SRAM/DRAM 占用，
切换训练内存实验和 SET 固定种子结果。控件筛选的是已有记录，不会在浏览器中启动新的求解。

![调度实验平台预览](docs/demo/preview.png)

## 已运行的实验

[完整报告](experiments/results/reproduction_20260921/REPORT.md)包含：

- **10 组推理对照：** 分层方案与串行参考使用相同的 SET 层内配置和分析式通信模型。
- **9 组训练容量配置：** 7 组可行，2 组在当前统一 cohort 策略下不可行。
- **12 次原版 SET：** 四个网络，每个网络三个固定随机种子，记录完整参数和原始输出。

![EDP 对比](experiments/results/reproduction_20260921/figures/edp_comparison.svg)

图中是 batch 64、16 tiles 下，相对于同成本模型串行参考的 EDP。
VGG-19 的分层结果略差，这个负面结果也被保留。原版 SET 使用更完整的布局与通信评估器，
因此单独报告，不能把两种外层成本模型的比值直接称为论文加速比。

重新运行实验：

```bash
python -m experiments.run --config experiments/configs/native_core_inference.json --output-dir outputs/experiments/my-inference
python -m experiments.run --config experiments/configs/training_memory.json --output-dir outputs/experiments/my-training
```

这些命令直接读取已保存的层内成本配置，不需要 C++ 编译器。
每个任务有独立时限，记录配置、代码版本、依赖、求解状态、验证结果和完整调度表。
数据导出、原版 SET 构建、绘图和排错步骤见 [实验指南](docs/EXPERIMENTS.md)。

## 代码导航

| 目录 | 内容 |
| --- | --- |
| `workloads/` | 真实网络元数据、加载器与初始分层 |
| `model/` | 层节点与 DAG 校验 |
| `scheduler/` | ScT/MeT、目标函数、流量区间、硬件参数 |
| `search/` | 平面搜索、父子 sub-batch 组合及训练策略 |
| `cost_model/` | 分析式模型、原版 SET 成本配置适配器 |
| `experiments/` | 实验配置、运行器、参考成本、原始证据与报告 |
| `tools/` | 环境检查和固定版本的上游导出/基线工具 |
| `example/` | 入门示例与保留的历史探索脚本 |
| `docs/demo/` | 可离线打开的交互演示 |

旧 `official_nns` 示例仍使用历史代理数据。新的真实网络实验统一从 `experiments/` 入口运行。
修改算法前，建议阅读 [架构](docs/ARCHITECTURE.md)、[数学检查](docs/MATHEMATICAL_AUDIT.md)
及 [复现边界](docs/REPRODUCTION.md)。

## 开发、引用与归属

```bash
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m ruff check .
python -m pytest -q
```

CI 覆盖 Linux/Python 3.11、3.13 和 Windows/Python 3.11。大型实验单独运行并记录预算。
研究方法引用 Crane 原论文；使用 SET 网络、层内成本或基线时，同时引用
[Cai 等，ISCA 2023](https://doi.org/10.1145/3579371.3589048)。使用本实现时记录 Git commit 和实验 manifest。

贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，变更见 [CHANGELOG.md](CHANGELOG.md)。
本仓库尚未声明统一的软件许可证；第三方资料的归属与许可状态见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
