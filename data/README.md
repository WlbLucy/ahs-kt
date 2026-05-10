# AHS-KT 数据格式说明

当前项目约定训练数据使用 `.npz` 打包，每个 split 一份文件，例如：

- `assist2009_train_ahskt.npz`
- `assist2009_valid_ahskt.npz`
- `assist2009_test_ahskt.npz`
- `assist2012_train_ahskt.npz`
- `assist2012_valid_ahskt.npz`
- `assist2012_test_ahskt.npz`

## 必备字段

每个 `.npz` 至少包含：

- `question_ids`：形状 `[N, L]`
- `concept_ids`：形状 `[N, L]`
- `responses`：形状 `[N, L]`
- `question_difficulty`：形状 `[N, L]`
- `concept_difficulty`：形状 `[N, L]`
- `attempts`：形状 `[N, L]`
- `hints`：形状 `[N, L]`
- `speed`：形状 `[N, L]`
- `behavior_cluster`：形状 `[N, L]`
- `mask`：形状 `[N, L]`

## 字段来源建议

### 来自 DIMKT

- `question_ids`
- `concept_ids`
- `responses`
- `question_difficulty`
- `concept_difficulty`

### 来自 LBKT

- `attempts`
- `hints`
- `speed`
- `behavior_cluster`
- `mask`

## 建议取值

- `responses`：`0/1`
- `question_difficulty`：建议离散到 `1..10`
- `concept_difficulty`：建议离散到 `1..10`
- `behavior_cluster`：建议 `1..K`，`0` 保留给 padding
- `mask`：有效位置为 `1`，padding 为 `0`

## 训练时的监督方式

项目默认采用：

- 用 `t` 时刻输入
- 预测 `t+1` 时刻的 correctness

因此训练时实际监督会使用：

- `responses[:, 1:]`
- `mask[:, 1:]`
