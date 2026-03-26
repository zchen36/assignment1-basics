from constants import TINY_STORY_PATH, OWT_PATH, DATA_DIR
from bpe import train_bpe_from_file
import pickle


def main(args):
    ts_path = TINY_STORY_PATH
    owt_path = OWT_PATH

    ts_vocab_size = 10000
    owt_vocab_size = 32000

    special_token = ["<|endoftext|>"]

    if args.mode == "ts":
        vocab, merges = train_bpe_from_file(ts_path, ts_vocab_size, special_token)

        with open(DATA_DIR / "ts_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

        with open(DATA_DIR / "ts_merges.pkl", "wb") as f:
            pickle.dump(merges, f)

    elif args.mode == "owt":
        vocab, merges = train_bpe_from_file(owt_path, owt_vocab_size, special_token)

        with open(DATA_DIR / "owt_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

        with open(DATA_DIR / "owt_merges.pkl", "wb") as f:
            pickle.dump(merges, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["ts", "owt"], required=True, help="what to train on.")
    main(parser.parse_args())
