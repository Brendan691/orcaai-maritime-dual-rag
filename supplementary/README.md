# 补充材料

对应论文 *Conditional gains and traceable citations in dual-source retrieval-augmented generation for maritime question answering*（中文本稿：《海事双源检索增强生成的效用边界与可追溯实现》）的补充材料。

## 投稿用单文件

- **`Supplementary-Information.pdf`**：完整 Supplementary Information（6 页），含 Supplementary Notes A1–A6、Supplementary Figs. S1–S3、Supplementary Tables S1–S4；正文所有 Supplementary 指针均可在该文件中找到。

## 分项源文件

- **S1 同步与异步执行的延迟对比**：`S1-latency.md`；图 `figures/s1_latency.pdf`（同名 `.png`/`.svg`）；数据 `data/s1_latency.csv`。对应 SI 的 **Supplementary Fig. S2**。
- **S2 检索变体宏平均 NDCG@5**：`S2-retrieval-variants.md`；图 `figures/s2_variants.pdf`（同名 `.png`/`.svg`）；数据 `data/s2_variants.csv`。对应 SI 的 **Supplementary Fig. S3**。

S1 为单机本地测量，不含网络往返与生产并发，不代表生产吞吐；S2 为正式查询集的次要分析（31 条查询，H13 因无相关文档致指标未定义而剔除）。
