# Crane 运行与论文方法总结（20260306_195852）

## 1) 本次运行输出文件
- outputs/runs/resnet50_20260306_195536.txt
- outputs/runs/advanced_20260306_195536.txt

可直接在终端复现：
`powershell
cd "C:\Users\Happend Outerwall\OneDrive\南京大学在线\Crane"
.\.venv\Scripts\python.exe example\resnet50_test.py
.\.venv\Scripts\python.exe example\advanced_networks_test.py
`

## 2) PDF 解析产物
- outputs/notes/paper_full_text.txt（全文提取）
- outputs/notes/paper_keyword_excerpt.txt（关键词抽取）
- outputs/notes/paper_method_snippets.txt（方法相关分页摘录）

论文来源：
C:\Users\Happend Outerwall\Documents\xwechat_files\wxid_qrjemeh2xdvj22_da41\msg\file\2026-03\3725843.3756023.pdf

## 3) 论文是如何探究“最优方案”的（核心思路）

### A. 先把调度空间“结构化表示”
论文先做统一表示，再做优化。核心是：
1. **分层 block 表示（hierarchical blocks）**，把复杂 DAG 划分为嵌套 block。
2. 每个 block 用 **pipeline-derived states** 表示执行状态，状态规模是 2N-1（N 为子块数）。
3. 用两张表编码调度：
   - **ScT（Scheduling Table）**：记录每个 state、每个子块的累计处理 sub-batch 数。
   - **MeT（Memory Table）**：记录每个 state、每个子块在 SRAM/DRAM 的子批次存储边界。

这一步的意义是：把“执行顺序、融合、重计算、批拆分”统一落到可约束、可计算的变量上。

### B. 在 block 内做 MILP 优化（不是启发式采样）
论文在每个候选 sub-batch 大小下，分阶段优化：
1. 先根据 ScT 的线性约束（文中 Eq.1-6）求 state workload，优化计算侧目标（延迟/能耗相关项）。
2. 再根据 MeT 约束（文中 Eq.7-12）优化存储与数据搬运侧目标（DRAM/NoC 相关项）。
3. 通过代价模型把计算+访存统一成总成本（以 EDP 为核心）。

论文提到：因为目标可转为 MILP 结构（含线性化处理），可以用成熟求解器更快收敛，并获得每轮问题的全局最优解。

### C. 再做分层结构优化（hierarchical structure optimization）
在 block 内最优之后，继续优化 block 划分本身：
1. **Graph Partition**：先按图结构划分候选 block。
2. **Gradual Partition**：逐层比较“合并/拆分”对总成本的影响，保留低成本结构。
3. **Iterative Update**：自顶向下与自底向上交替更新，直到收敛。

这相当于“结构搜索 + 参数优化”结合，而不是固定结构下调参数。

### D. 训练场景下把 recomputation 也并入同一框架
论文把重计算拆成“前向后剩余激活 + 重算后接反向”的状态过程，仍由 ScT/MeT 追踪，
并在约束中加入 forward/backward 与重算之间的数据可用性与边界关系（文中 Eq.13-15 附近）。

### E. 为什么它能找到更优方案
从论文叙述看，核心原因是：
1. 表示能力更完整：四个核心因子（执行模式/融合/重算/批拆分）一起优化。
2. 搜索方式更系统：MILP + 分层渐进优化，少依赖启发式随机采样。
3. 通过分层与 Top-K 等机制控制搜索复杂度，兼顾质量与速度。

## 4) 与你当前代码的对应关系（简述）
你当前项目已体现：
- 2N-1 状态顺序（W/M/D）
- ScT/MeT 表结构
- MILP 分配（含多状态激活约束）

尚未完全等价论文的部分主要是：
- 论文级完整约束细节（尤其 Eq.1-15 的全量细节）
- 完整分层 block 渐进划分过程（目前是简化版）
- 训练前后向与重计算的全流程状态联立优化

