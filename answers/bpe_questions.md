* Problem (train_bpe_tinystories): BPE Training on TinyStories

1. longest token: 7160 b' accomplishment'

2. does it make sense? Yes, a complete word tends to show up more frequently.

3. profile the code. The pre_tokenization took most of the time(19.6s out of 34s).

run `uv run snakeviz answers/ts_bpe.prof` to see the results.