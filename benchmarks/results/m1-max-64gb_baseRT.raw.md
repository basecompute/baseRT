# Raw BaseRT benchmark output — M1 Max 64 GB

This is the complete stdout from the second end-to-end sweep recorded in
`m1-max-64gb_baseRT.csv`. The invocation and artifact provenance are documented
in `m1-max-64gb_baseRT.md`.

```text
[1/1] Benchmarking: /Users/ariaki-m1max/Library/Caches/baseRT/models/Qwen/Qwen3.6-35B-A3B/default-q4/model.base
  pp128/tg128, reps=5, warmup=1
| model | size | test | t/s |
| --- | ---: | ---: | ---: |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | pp128 | 552.07 ± 1.29 |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | tg128 | 96.77 ± 0.72 |
  pp256/tg128, reps=5, warmup=1
| model | size | test | t/s |
| --- | ---: | ---: | ---: |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | pp256 | 692.73 ± 1.88 |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | tg128 | 96.37 ± 0.66 |
  pp512/tg128, reps=5, warmup=1
| model | size | test | t/s |
| --- | ---: | ---: | ---: |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | pp512 | 851.39 ± 0.41 |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | tg128 | 97.15 ± 0.77 |
  pp2048/tg128, reps=5, warmup=1
| model | size | test | t/s |
| --- | ---: | ---: | ---: |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | pp2048 | 1025.34 ± 10.61 |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | tg128 | 96.48 ± 0.62 |
  pp6000/tg128, reps=5, warmup=1
| model | size | test | t/s |
| --- | ---: | ---: | ---: |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20928 MiB | pp6000 | 878.60 ± 3.18 |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20928 MiB | tg128 | 95.99 ± 0.86 |
  pp43000/tg128, reps=5, warmup=1
| model | size | test | t/s |
| --- | ---: | ---: | ---: |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 21651 MiB | pp43000 | 574.93 ± 1.00 |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 21651 MiB | tg128 | 96.53 ± 0.52 |
Wrote results to /tmp/upstream-run-m1max.csv
```
