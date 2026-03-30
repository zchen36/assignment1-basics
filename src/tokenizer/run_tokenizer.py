from pathlib import Path
import os
from tokenizer.constants import (
    TINY_STORY_PATH,
    OWT_PATH,
    TINY_STORY_VALID_PATH,
    OWT_VALID_PATH,
    TS_VOCAB_PATH,
    TS_MERGES_PATH,
    OWT_VOCAB_PATH,
    OWT_MERGES_PATH,
    DATA_DIR,
)
from tokenizer.tokenizer import Tokenizer
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def read_first_10_documents_from_file(file_path: Path, document_delimiter: str) -> list[str]:
    with file_path.open("r") as f:
        documents = f.read().split(document_delimiter)
        return documents[:10]


def main(args):
    ts_tokenizer = Tokenizer.from_file(
        vocab_path=TS_VOCAB_PATH, merges_path=TS_MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    owt_tokenizer = Tokenizer.from_file(
        vocab_path=OWT_VOCAB_PATH, merges_path=OWT_MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    if args.mode == "ts":
        documents = read_first_10_documents_from_file(TINY_STORY_VALID_PATH, document_delimiter="<|endoftext|>")

        ts_tokenizer = Tokenizer.from_file(
            vocab_path=TS_VOCAB_PATH, merges_path=TS_MERGES_PATH, special_tokens=["<|endoftext|>"]
        )

        output_dir = DATA_DIR / "ts" / "encoding_result"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, document in enumerate(documents):
            tokens = ts_tokenizer.encode(document)
            with open(output_dir / f"document_{i}.txt", "w") as f:
                f.write(document)

            with open(output_dir / f"encoded_{i}.txt", "w") as f:
                f.write(",".join(map(str, tokens)))

            total_bytes = len(document.encode("utf-8"))
            total_tokens = len(tokens)
            logging.info("bytes/token: %f", total_bytes / total_tokens)

    elif args.mode == "owt":
        documents = read_first_10_documents_from_file(OWT_VALID_PATH, document_delimiter="<|endoftext|>")

        owt_tokenizer = Tokenizer.from_file(
            vocab_path=OWT_VOCAB_PATH, merges_path=OWT_MERGES_PATH, special_tokens=["<|endoftext|>"]
        )

        output_dir = DATA_DIR / "owt" / "encoding_result"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, document in enumerate(documents):
            tokens = owt_tokenizer.encode(document)
            with open(output_dir / f"document_{i}.txt", "w") as f:
                f.write(document)

            with open(output_dir / f"encoded_{i}.txt", "w") as f:
                f.write(",".join(map(str, tokens)))

            total_bytes = len(document.encode("utf-8"))
            total_tokens = len(tokens)
            logging.info("bytes/token: %f", total_bytes / total_tokens)

    elif args.mode == "ts_on_owt":
        # use TS tokenizer on OWT text
        ts_documents = read_first_10_documents_from_file(TINY_STORY_VALID_PATH, document_delimiter="<|endoftext|>")
        owt_documents = read_first_10_documents_from_file(OWT_VALID_PATH, document_delimiter="<|endoftext|>")
        ts_tokenizer = Tokenizer.from_file(
            vocab_path=TS_VOCAB_PATH, merges_path=TS_MERGES_PATH, special_tokens=["<|endoftext|>"]
        )
        owt_tokenizer = Tokenizer.from_file(
            vocab_path=OWT_VOCAB_PATH, merges_path=OWT_MERGES_PATH, special_tokens=["<|endoftext|>"]
        )

        output_dir = DATA_DIR / "owt_on_ts" / "encoding_result"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, document in enumerate(ts_documents):
            tokens = owt_tokenizer.encode(document)
            with open(output_dir / f"document_{i}.txt", "w") as f:
                f.write(document)

            with open(output_dir / f"encoded_{i}.txt", "w") as f:
                f.write(",".join(map(str, tokens)))

            total_bytes = len(document.encode("utf-8"))
            total_tokens = len(tokens)
            logging.info("bytes/token: %f", total_bytes / total_tokens)

    elif args.mode == "ts_full":
        data_file_path = TINY_STORY_PATH if args.dataset == "train" else TINY_STORY_VALID_PATH
        output_dir = DATA_DIR / "ts" / "full_encoding_result"

        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info("start")
        start = time.time()
        with open(output_dir / f"encoded_{args.dataset}.txt", "w") as f:
            first = True
            for t in ts_tokenizer.encode_iterable(open(data_file_path, "r")):
                if first:
                    f.write(str(t))
                    first = False
                else:
                    f.write("," + str(t))
        end = time.time()
        logging.info("start")

        logging.info("total time: %f sec", end - start)
        logging.info("filesize: %f bytes", data_file_path.stat().st_size)
        logging.info("throughput: %f bytes/sec", data_file_path.stat().st_size / (end - start))

    elif args.mode == "owt_full":
        data_file_path = OWT_PATH if args.dataset == "train" else OWT_VALID_PATH
        output_dir = DATA_DIR / "owt" / "full_encoding_result"

        output_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        logging.info("start")
        with open(output_dir / f"encoded_{args.dataset}.txt", "w") as f:
            first = True
            for t in owt_tokenizer.encode_iterable(open(data_file_path, "r")):
                if first:
                    f.write(str(t))
                    first = False
                else:
                    f.write("," + str(t))
        end = time.time()
        logging.info("end")

        logging.info("total time: %f sec", end - start)
        logging.info("filesize: %f bytes", data_file_path.stat().st_size)
        logging.info("throughput: %f bytes/sec", data_file_path.stat().st_size / (end - start))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ts", "owt", "ts_on_owt", "ts_full", "owt_full"],
        required=True,
        help="what to train on.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train", "valid"],
        required=True,
        help="what dataset to use",
    )

    main(parser.parse_args())
