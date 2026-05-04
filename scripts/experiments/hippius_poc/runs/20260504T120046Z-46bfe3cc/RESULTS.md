# Hippius B200 perf benchmark — 20260504T120046Z-46bfe3cc

## Throughput / latency

| tier | HF dl | upload | download | verify | vLLM load | TTFT | tok/s | tok/s/seq |
|---|---|---|---|---|---|---|---|---|
| tinyllama | 177.67 MB/s | 98.67 MB/s | 23.33 MB/s | 2065.36 MB/s | n/a | n/a | n/a | n/a |

## Volumes

| tier | HF size | HF time | upload time | download time |
|---|---|---|---|---|
| tinyllama | 2.05 GiB | 11.82 s | 21.29 s | 90.05 s |

## Pass/fail vs migration criteria

- **FAIL** tinyllama upload >= 200.0 MB/s — got 98.67 MB/s
- **FAIL** tinyllama download >= 400.0 MB/s — got 23.33 MB/s
- **PASS** tinyllama LastModified spread <= 5.0s — got 2.0s
- **PASS** tinyllama sha256 roundtrip — ok
- **PASS** tinyllama IfMatch 412 on bogus etag — ok

## Overall: NO-GO / NEEDS REVIEW
