* Problem (train_bpe_tinystories): BPE Training on TinyStories

1. longest token: 7160 b' accomplishment'

2. does it make sense? Yes, a complete word tends to show up more frequently.

3. profile the code. The pre_tokenization took most of the time(19.6s out of 34s).

run `uv run snakeviz answers/ts_bpe.prof` to see the results.

* Problem tokenizer_experiments

1. for TS:

```
2026-03-29 22:57:35,406 [INFO] bytes/token: 4.101695
2026-03-29 22:57:35,407 [INFO] bytes/token: 3.843602
2026-03-29 22:57:35,408 [INFO] bytes/token: 4.126582
2026-03-29 22:57:35,409 [INFO] bytes/token: 4.388889
2026-03-29 22:57:35,409 [INFO] bytes/token: 4.219355
2026-03-29 22:57:35,410 [INFO] bytes/token: 4.082589
2026-03-29 22:57:35,411 [INFO] bytes/token: 4.242291
2026-03-29 22:57:35,412 [INFO] bytes/token: 3.591418
2026-03-29 22:57:35,413 [INFO] bytes/token: 3.924138
2026-03-29 22:57:35,414 [INFO] bytes/token: 4.050505
```

for OWT:

```
2026-03-29 23:00:17,025 [INFO] bytes/token: 4.567511
2026-03-29 23:00:17,030 [INFO] bytes/token: 4.149660
2026-03-29 23:00:17,033 [INFO] bytes/token: 4.778718
2026-03-29 23:00:17,035 [INFO] bytes/token: 3.836237
2026-03-29 23:00:17,039 [INFO] bytes/token: 4.697003
2026-03-29 23:00:17,041 [INFO] bytes/token: 4.293522
2026-03-29 23:00:17,044 [INFO] bytes/token: 4.605898
2026-03-29 23:00:17,051 [INFO] bytes/token: 4.544234
2026-03-29 23:00:17,053 [INFO] bytes/token: 4.715789
2026-03-29 23:00:17,056 [INFO] bytes/token: 4.590840
```

2. cross tokenize TS and OWT

Use ts tokenizer on owt text, the ratio is getting smaller(worse?).

```
2026-03-29 23:05:20,431 [INFO] bytes/token: 3.351393
2026-03-29 23:05:20,435 [INFO] bytes/token: 3.856691
2026-03-29 23:05:20,438 [INFO] bytes/token: 3.442509
2026-03-29 23:05:20,439 [INFO] bytes/token: 2.780303
2026-03-29 23:05:20,443 [INFO] bytes/token: 3.319216
2026-03-29 23:05:20,444 [INFO] bytes/token: 3.656897
2026-03-29 23:05:20,447 [INFO] bytes/token: 3.534979
2026-03-29 23:05:20,453 [INFO] bytes/token: 3.143943
2026-03-29 23:05:20,455 [INFO] bytes/token: 3.278049
2026-03-29 23:05:20,457 [INFO] bytes/token: 3.575505
```

if owt on ts(also worse):

```
2026-03-29 23:07:05,175 [INFO] bytes/token: 4.101695
2026-03-29 23:07:05,176 [INFO] bytes/token: 3.843602
2026-03-29 23:07:05,177 [INFO] bytes/token: 3.959514
2026-03-29 23:07:05,178 [INFO] bytes/token: 4.279167
2026-03-29 23:07:05,179 [INFO] bytes/token: 4.219355
2026-03-29 23:07:05,181 [INFO] bytes/token: 3.818372
2026-03-29 23:07:05,182 [INFO] bytes/token: 4.046218
2026-03-29 23:07:05,184 [INFO] bytes/token: 3.456014
2026-03-29 23:07:05,185 [INFO] bytes/token: 3.924138
2026-03-29 23:07:05,186 [INFO] bytes/token: 3.837321
```

3. throughput

time: ~4hr

size: 12G

throughput: 800 kb/sec

```
2026-03-30 03:54:50,686 [INFO] total time: 14766.626983 sec
2026-03-30 03:54:50,686 [INFO] filesize: 11920511059.000000 bytes
2026-03-30 03:54:50,686 [INFO] throughput: 807260.254680 bytes/sec
```

1. tokenize full text. in data/owt/full_encoding_result