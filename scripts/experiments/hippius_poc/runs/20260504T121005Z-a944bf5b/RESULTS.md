# Hippius B200 perf benchmark — 20260504T121005Z-a944bf5b

## Throughput / latency

| tier | HF dl | upload | download | verify | vLLM load | TTFT | tok/s | tok/s/seq |
|---|---|---|---|---|---|---|---|---|
| tinyllama | 190.65 MB/s | 63.80 MB/s | 17.19 MB/s | 1987.66 MB/s | 22.12 s | 0.19 s | 85.19 | 10.65 |

## Volumes

| tier | HF size | HF time | upload time | download time |
|---|---|---|---|---|
| tinyllama | 2.05 GiB | 11.02 s | 32.92 s | 122.22 s |

## Pass/fail vs migration criteria

- **FAIL** tinyllama upload >= 200.0 MB/s — got 63.80 MB/s
- **FAIL** tinyllama download >= 400.0 MB/s — got 17.19 MB/s
- **PASS** tinyllama LastModified spread <= 5.0s — got 2.0s
- **PASS** tinyllama sha256 roundtrip — ok
- **PASS** tinyllama IfMatch 412 on bogus etag — ok

## Overall: NO-GO / NEEDS REVIEW
