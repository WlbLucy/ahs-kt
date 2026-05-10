# AHS-KT

`AHS-KT` 是一个基于 `DIMKT` 与 `LBKT` 思路融合出来的新项目骨架，核心目标是把：

- 难度信息：`question_difficulty`、`concept_difficulty`
- 行为信息：`attempts`、`hints`、`speed`
- 相对速度：`speed_relative_student`、`speed_relative_question`
- 行为原型：`behavior_cluster_id` + `behavior_soft_membership`

统一放进一个时序知识追踪模型中。

这里的 `AHS` 表示：

- `A`：Attempts
- `H`：Hints
- `S`：Speed

同时模型还显式建模题目与知识点难度，因此可以理解为：

- **AHS-KT = Attempt-Hint-Speed + Difficulty aware Knowledge Tracing**

## 1. 当前项目包含什么

当前目录已经搭好了一个**可运行的第一版代码骨架**，包括：

- `configs/`
  - `ahskt_demo.json`：可直接跑通的 demo 配置
  - `ahskt_assist2009_template.json`：后续接真实数据时的模板配置
  - `ahskt_assist2012_template.json`：`assist2012` 的模板配置
- `src/ahskt/`
  - `config.py`：配置读取
  - `data/`：数据结构、synthetic demo、DIMKT/LBKT 字段桥接说明
  - `models/`：AHS-KT 模型与编码器
  - `training/`：训练与评估循环
- `scripts/train_ahskt.py`
  - 支持加载配置并训练
  - 支持 `--demo` 直接跑 synthetic 数据
- `scripts/build_assist2012_ahskt.py`
  - 从 `DIMKT` 的 `assist2012` 原始 CSV 构建 `AHS-KT` 所需 `.npz`
  - 自动生成 `behavior cluster` 与训练配置
- `tests/test_smoke.py`
  - 基础形状与前向 smoke test

## 2. 目录结构

```text
ahs-kt/
├── README.md
├── configs/
│   ├── ahskt_demo.json
│   ├── ahskt_assist2009_template.json
│   └── ahskt_assist2012_template.json
├── data/
│   └── README.md
├── outputs/
├── scripts/
│   ├── build_assist2012_ahskt.py
│   └── train_ahskt.py
├── src/
│   └── ahskt/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── bridge.py
│       │   ├── dataset.py
│       │   ├── assist2012.py
│       │   └── synthetic.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── ahs_kt.py
│       │   └── encoders.py
│       └── training/
│           ├── __init__.py
│           ├── engine.py
│           └── metrics.py
└── tests/
    ├── test_assist2012_bridge.py
    └── test_smoke.py
```

## 3. AHS-KT 第一版模型结构

当前第一版实现是一个**可运行的联合骨架**：

1. `question_id / concept_id / response` 进入基础嵌入层
2. `question_difficulty / concept_difficulty` 进入难度编码器
3. `attempts / hints / speed / behavior_cluster_id` 进入行为编码器
4. 行为编码器里带一个 **difficulty-aware gate**
5. 所有表示拼接后送入 `GRU`
6. 输出每个时间步的预测 logits
7. 训练时使用 `t` 时刻状态去预测 `t+1` 时刻的 correctness

当前增强版还额外做了两件事：

- **相对速度建模**：不仅输入原始 `speed`，还输入相对学生历史速度与相对题目典型速度
- **软原型 AHS**：不仅使用硬聚类 `behavior_cluster`，还使用样本对 4 类行为原型的软权重

这版实现重点是：

- 先把工程骨架跑通
- 把字段和模块接口定下来
- 为后续真正接 `DIMKT` 数据与 `LBKT` 数据做准备

## 4. 真实数据格式

项目约定真实训练数据使用 `.npz` 存储，至少包含这些字段：

- `question_ids`
- `concept_ids`
- `responses`
- `question_difficulty`
- `concept_difficulty`
- `attempts`
- `hints`
- `speed`
- `speed_relative_student`
- `speed_relative_question`
- `behavior_cluster`
- `behavior_soft_membership`
- `mask`

详细说明见：

- `data/README.md`

## 5. 怎么直接跑通 demo

进入项目目录后运行：

```bash
cd /root/autodl-tmp/ahs-kt
python scripts/train_ahskt.py --config configs/ahskt_demo.json --demo
```

运行完成后会在 `outputs/demo_run/` 下生成：

- `ahskt_demo_metrics.json`

## 6. 怎么接入 assist2012

进入项目目录后运行：

```bash
cd /root/autodl-tmp/ahs-kt
python scripts/build_assist2012_ahskt.py
python scripts/train_ahskt.py --config configs/ahskt_assist2012_v1.json --cpu-only
```

构建脚本会：

- 复用 `DIMKT/data` 里的 `problem2id`、`skill2id`、`difficult2id`、`sdifficult2id`
- 复刻 `DIMKT` 的 `train0 / valid0 / test` 用户切分逻辑
- 从原始表补出 `attempt_count`、`hint_count`、`ms_first_response`
- 生成 `attempts / hints / speed / behavior_cluster`
- 导出：
  - `data/assist2012_train_ahskt.npz`
  - `data/assist2012_valid_ahskt.npz`
  - `data/assist2012_test_ahskt.npz`
  - `data/assist2012_metadata.json`
  - `configs/ahskt_assist2012_v1.json`

## 7. 下一步怎么接真实 DIMKT / LBKT 数据

建议按下面顺序推进：

### 第一步

先把 `LBKT` 侧的行为特征接进来：

- `attempts`
- `hints`
- `speed`
- `behavior_cluster`

### 第二步

再把 `DIMKT` 侧的难度特征接进来：

- `question_difficulty`
- `concept_difficulty`

### 第三步

把两边整理成统一的 `.npz` 序列包，字段对齐后直接喂给：

- `src/ahskt/data/dataset.py`

### 第四步

跑下面几组对照：

- only difficulty
- only behavior
- difficulty + behavior
- difficulty + behavior + cluster

## 8. 这版骨架最适合干什么

这版项目不是最终论文代码，而是一个**能快速迭代方法的实验底座**。  
它最适合：

- 快速试新输入特征
- 快速试融合方式
- 快速试 `difficulty-aware behavior gate`
- 快速跑 early-stage convergence 实验

如果后面你要继续推进，我建议优先做：

1. 把真实 `assist2009 / 2012` 数据整理成 `.npz`
2. 先跑 `epoch × seed`
3. 画收敛曲线
4. 再增强 gate 和 prototype 设计
