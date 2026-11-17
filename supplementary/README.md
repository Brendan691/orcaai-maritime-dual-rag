# 补充材料

对应论文《海事双源检索增强生成的效用边界与可追溯实现》的补充材料。

- **S1 同步与异步执行的延迟对比**：`S1-latency.md`；图 `figures/s1_latency.pdf`（同名 `.png`/`.svg`）；数据 `data/s1_latency.csv`。
- **S2 检索变体宏平均 NDCG@5**：`S2-retrieval-variants.md`；图 `figures/s2_variants.pdf`（同名 `.png`/`.svg`）；数据 `data/s2_variants.csv`。

S1 为单机本地测量，不含网络往返与生产并发，不代表生产吞吐；S2 为正式查询集的次要分析（31 条查询，H13 因无相关文档致指标未定义而剔除）。
