from tokenizer.constants import TINY_STORY_PATH, OWT_PATH, DATA_DIR
from tokenizer.bpe import train_bpe_from_file
import pickle
from tokenizer.util import gpt2_bytes_to_unicode
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _make_vocab_readable(vocab: dict[int, bytes], bytes_to_unicode: dict[int, str]) -> dict[int, str]:
    ret = {}
    for k, v in vocab.items():
        ret[k] = "".join(bytes_to_unicode[b] for b in v)

    return ret


def _make_merge_readable(merges: list[tuple[bytes, bytes]], bytes_to_unicode: dict[int, str]) -> list[tuple[str, str]]:
    ret = []
    for first, second in merges:
        ret.append(("".join(bytes_to_unicode[b] for b in first), "".join(bytes_to_unicode[b] for b in second)))
    return ret


def main(args):
    ts_path = TINY_STORY_PATH
    owt_path = OWT_PATH

    ts_vocab_size = 10000
    owt_vocab_size = 32000

    special_token = ["<|endoftext|>"]

    if args.mode == "ts":
        vocab, merges = train_bpe_from_file(ts_path, ts_vocab_size, special_token)

        logging.info("bpe finished. next write to file.")

        file_path = DATA_DIR / "ts" / "ts_vocab.pkl"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(DATA_DIR / "ts" / "ts_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

        with open(DATA_DIR / "ts" / "ts_merges.pkl", "wb") as f:
            pickle.dump(merges, f)

        logging.info("convert to readable.")
        readable_vocab = _make_vocab_readable(vocab, gpt2_bytes_to_unicode())
        readable_merges = _make_merge_readable(merges, gpt2_bytes_to_unicode())

        with open(DATA_DIR / "ts" / "ts_readable_vocab.json", "w") as f:
            json.dump(readable_vocab, f, indent=4, ensure_ascii=False)

        with open(DATA_DIR / "ts" / "ts_readable_merges.json", "w") as f:
            json.dump(readable_merges, f, indent=4, ensure_ascii=False)

    elif args.mode == "owt":
        vocab, merges = train_bpe_from_file(owt_path, owt_vocab_size, special_token)

        logging.info("bpe finished. next write to file.")

        file_path = DATA_DIR / "owt" / "owt_vocab.pkl"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(DATA_DIR / "owt" / "owt_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

        with open(DATA_DIR / "owt" / "owt_merges.pkl", "wb") as f:
            pickle.dump(merges, f)

        logging.info("convert to readable.")
        readable_vocab = _make_vocab_readable(vocab, gpt2_bytes_to_unicode())
        readable_merges = _make_merge_readable(merges, gpt2_bytes_to_unicode())

        with open(DATA_DIR / "owt" / "owt_readable_vocab.json", "w") as f:
            json.dump(readable_vocab, f, indent=4, ensure_ascii=False)

        with open(DATA_DIR / "owt" / "owt_readable_merges.json", "w") as f:
            json.dump(readable_merges, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["ts", "owt"], required=True, help="what to train on.")
    main(parser.parse_args())
