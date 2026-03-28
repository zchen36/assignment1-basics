from tokenizer.constants import DATA_DIR
import pickle


def main():
    ts_vocab_path = DATA_DIR / "ts" / "ts_vocab.pkl"
    ts_merges_path = DATA_DIR / "ts" / "ts_merges.pkl"

    with open(ts_vocab_path, "rb") as f:
        ts_vocab: dict[int, bytes] = pickle.load(f)

    with open(ts_merges_path, "rb") as f:
        ts_merges: list[tuple[bytes, bytes]] = pickle.load(f)

    print(ts_merges[:20])

    key, value = max(ts_vocab.items(), key=lambda x: len(x[1]))
    print(key, value)  # 7160 b' accomplishment'


if __name__ == "__main__":
    main()
