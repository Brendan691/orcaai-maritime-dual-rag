# Supplementary Information

Supplementary material for *Conditional gains and traceable citations in dual-source retrieval-augmented generation for maritime question answering*.

## Compiled submission file

- **`Supplementary-Information.pdf`**: complete Supplementary Information (6 pages), containing Supplementary Notes A1–A6, Supplementary Figs. S1–S3, and Supplementary Tables S1–S4. Every Supplementary pointer in the main text can be found in this file.

## Source components

- **S1: Synchronous versus asynchronous latency**: `S1-latency.md`; figure `figures/s1_latency.pdf` (also `.png`/`.svg`); data `data/s1_latency.csv`. This component corresponds to **Supplementary Fig. S2** in the compiled SI.
- **S2: Macro-averaged NDCG@5 of retrieval variants**: `S2-retrieval-variants.md`; figure `figures/s2_variants.pdf` (also `.png`/`.svg`); data `data/s2_variants.csv`. This component corresponds to **Supplementary Fig. S3** in the compiled SI.

S1 is a single-machine local measurement and does not include network round trips or production concurrency, so it does not represent production throughput. S2 is a secondary analysis of the formal query set (31 queries; H13 was excluded because no relevant document made the metric undefined).
