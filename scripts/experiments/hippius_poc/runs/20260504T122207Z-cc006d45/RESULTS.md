# Hippius B200 perf benchmark — 20260504T122207Z-cc006d45

## Throughput / latency

| tier | HF dl | upload | download | verify | vLLM load | TTFT | tok/s | tok/s/seq |
|---|---|---|---|---|---|---|---|---|
| tinyllama | 166.25 MB/s | 34.53 MB/s | 80.18 MB/s | 2020.70 MB/s | 15.93 s | 0.09 s | 98.41 | 12.30 |

## Volumes

| tier | HF size | HF time | upload time | download time |
|---|---|---|---|---|
| tinyllama | 2.05 GiB | 12.63 s | 60.83 s | 26.20 s |

## Pass/fail vs migration criteria

- **FAIL** tinyllama upload >= 200.0 MB/s — got 34.53 MB/s
- **FAIL** tinyllama download >= 400.0 MB/s — got 80.18 MB/s
- **PASS** tinyllama LastModified spread <= 5.0s — got 2.0s
- **PASS** tinyllama sha256 roundtrip — ok
- **PASS** tinyllama IfMatch 412 on bogus etag — ok

## Overall: NO-GO / NEEDS REVIEW
