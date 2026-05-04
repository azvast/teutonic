# Hippius B200 perf benchmark — 20260504T131357Z-94489ed4

## Throughput / latency

| tier | HF dl | upload | download | verify | vLLM load | TTFT | tok/s | tok/s/seq |
|---|---|---|---|---|---|---|---|---|
| tinyllama | 220.60 MB/s | 99.13 MB/s | 94.11 MB/s | 1962.46 MB/s | 13.20 s | 0.03 s | 1041.10 | 130.14 |

## Volumes

| tier | HF size | HF time | upload time | download time |
|---|---|---|---|---|
| tinyllama | 2.05 GiB | 9.52 s | 21.19 s | 22.32 s |

## Pass/fail vs migration criteria

- **FAIL** tinyllama upload >= 200.0 MB/s — got 99.13 MB/s
- **FAIL** tinyllama download >= 400.0 MB/s — got 94.11 MB/s
- **PASS** tinyllama LastModified spread <= 5.0s — got 3.0s
- **PASS** tinyllama sha256 roundtrip — ok
- **PASS** tinyllama IfMatch 412 on bogus etag — ok

## Overall: NO-GO / NEEDS REVIEW
