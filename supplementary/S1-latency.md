# Supplementary S1: Synchronous versus asynchronous latency

Latency supplementary analysis for Section 4 (engineering audit and fault-handling verification). Figure: `figures/s1_latency.pdf` (also `.png`/`.svg`; log scale, median and 95th percentile); data: `data/s1_latency.csv`.

## Environment

Apple M4 (10 logical cores: 4 performance and 6 efficiency cores), 16 GiB memory, macOS 27.0.1 (arm64), CPython 3.13.13. All execution was local: `external_api_calls=0`, `live_web_calls=0`, `generation_calls=0`. The bge-reranker-v2-m3 model was loaded once (3.679 s) and the frozen input package was loaded once (1.014 s). Asynchronous concurrency was 4, and the reranking backend used `torch.set_num_threads(4)`.

## Protocol

One warm-up round (one round per mode); 10 formal AB/BA alternating rounds (odd and even rounds swapped the sync/async order to offset order effects), each round containing 8 development queries × 2 modes = 80 runs per mode. P50/P95 were computed from the per-round measurements; single process, same machine, same model, wall_seconds = 700.2 (excluding report writing). Load before the formal test: load average 4.39/3.70/4.21; peak RSS about 5.57 GB.

## Results

| Mode | P50 | P95 |
|---|---:|---:|
| Synchronous | 3613.5 ms | 3868.3 ms |
| Asynchronous | 23341.3 ms | 36620.8 ms |

The asynchronous P50 was 6.459 times the synchronous P50. The two modes produced equivalent outputs (80/80 agreement; an additional 160/160 agreed with the main-experiment reference outputs).

## Limitations

Single-machine local measurement; it does not include network round trips or production concurrency and does not represent production throughput.
