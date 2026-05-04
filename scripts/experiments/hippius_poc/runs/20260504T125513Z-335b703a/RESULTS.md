# Hippius B200 perf benchmark — 20260504T125513Z-335b703a

## Throughput / latency

| tier | HF dl | upload | download | verify | vLLM load | TTFT | tok/s | tok/s/seq |
|---|---|---|---|---|---|---|---|---|
| tinyllama | 236.81 MB/s | 32.72 MB/s | 93.93 MB/s | 1876.05 MB/s | 15.70 s | 0.03 s | 1061.43 | 132.68 |

## Volumes

| tier | HF size | HF time | upload time | download time |
|---|---|---|---|---|
| tinyllama | 2.05 GiB | 8.87 s | 64.20 s | 22.36 s |

## Pass/fail vs migration criteria

- **FAIL** tinyllama upload >= 200.0 MB/s — got 32.72 MB/s
- **FAIL** tinyllama download >= 400.0 MB/s — got 93.93 MB/s
- **PASS** tinyllama LastModified spread <= 5.0s — got 2.0s
- **PASS** tinyllama sha256 roundtrip — ok
- **PASS** tinyllama IfMatch 412 on bogus etag — ok

## Overall: NO-GO / NEEDS REVIEW
