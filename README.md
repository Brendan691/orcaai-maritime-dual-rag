# Conditional gains and traceable citations in dual-source retrieval-augmented generation for maritime question answering

Companion implementation and public data for the paper.

The system fuses local-document and web evidence under fixed candidate and retained-evidence counts, generates answers with bound citations, and provides reproducible inputs for the paper's statistical results.

## Repository layout

- `code/pipeline/`: retrieval, fusion, reranking, answer generation, citation binding, and localization
- `code/analysis/`: statistical analysis and verification
- `code/configs/`, `code/prompts/`: model configuration and prompts
- `data/`: corpus manifest, queries, core checkpoints, relevance judgments, per-query scores, and checkpoint counts
- `supplementary/`: compiled Supplementary Information PDF, S1 (sync/async latency), S2 (retrieval-variant NDCG@5), figures, and data
- `tools/`: integrity checks and reproduction of key numbers

## Data

- **Public**: corpus manifest (metadata for 48 documents, no full text), 40 human-written queries (32 formal and 8 development), core checkpoints, relevance judgments, de-identified scores, checkpoint counts, code, and Supplementary S1/S2 with figures and data.
- **Restricted**: raw answers and evidence text, raw rating exports, rater–answer mappings, and blind keys. These are not included in the repository and are available from the corresponding author upon reasonable request.
- **Third party**: copyright in corpus text, web content, and model weights remains with the original rights holders; these materials are not distributed with this repository.

**Language note.** All documentation, metadata, and code comments in this repository are in English. Research materials in `data/` and `code/prompts/`, the functional Chinese tokenization/stopword rules used for retrieval, and the legacy gate-report templates in `code/analysis/` remain in their original Chinese because they are study materials, functional retrieval rules, or archived internal reports; translating them would change the study or the historical record. Absolute local filesystem paths also retain their original names.

## Reproduction

```bash
python tools/check_checksums.py
python tools/reproduce_key_numbers.py
```

The second script recomputes the primary contrast, interaction, and equivalence tests from the public per-query scores and compares them with the paper's values.

## License and citation

Code is released under the MIT License (`LICENSE-CODE`); data and documentation are released under CC BY 4.0 (`LICENSE-DATA`). Citation information is in `CITATION.cff`.

Repository: `https://github.com/Brendan691/orcaai-maritime-dual-rag`
