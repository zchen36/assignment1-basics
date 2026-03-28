import pathlib

PAT: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

DESIRED_NUM_CHUNKS: int = 60

TINY_STORY_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data/TinyStoriesV2-GPT4-train.txt"

OWT_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data/owt_train.txt"

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"
