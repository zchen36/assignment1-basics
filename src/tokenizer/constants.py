import pathlib

PAT: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

DESIRED_NUM_CHUNKS: int = 20

MAX_READ_TEXT_LENGTH: int = 200000000  # length 100M is roughly 100MB in memory

TINY_STORY_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data/TinyStoriesV2-GPT4-train.txt"
TINY_STORY_VALID_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data/TinyStoriesV2-GPT4-valid.txt"

OWT_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data/owt_train.txt"
OWT_VALID_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data/owt_valid.txt"


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"

TS_VOCAB_PATH = DATA_DIR / "ts" / "ts_vocab.pkl"
TS_MERGES_PATH = DATA_DIR / "ts" / "ts_merges.pkl"

OWT_VOCAB_PATH = DATA_DIR / "owt" / "owt_vocab.pkl"
OWT_MERGES_PATH = DATA_DIR / "owt" / "owt_merges.pkl"
