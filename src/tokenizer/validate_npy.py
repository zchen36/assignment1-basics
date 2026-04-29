from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np


SEPARATORS = {",", "\n", "\r", "\t", " "}


def iter_text_token_ids(input_path: Path, chunk_size_bytes: int = 1024 * 1024):
    buffer = ""

    with input_path.open("r", encoding="utf-8") as infile:
        while True:
            chunk = infile.read(chunk_size_bytes)
            if not chunk:
                break

            text = buffer + chunk
            pieces = text.split(",")
            buffer = pieces.pop()

            for piece in pieces:
                token = piece.strip()
                if token:
                    yield int(token)

    token = buffer.strip()
    if token:
        yield int(token)


def build_positions(length: int, num_random_checks: int, seed: int) -> list[int]:
    if length == 0:
        return []

    positions = {0, length - 1, length // 2}
    positions.update(range(min(10, length)))
    positions.update(range(max(0, length - 10), length))

    remaining = [idx for idx in range(length) if idx not in positions]
    if remaining and num_random_checks > 0:
        rng = random.Random(seed)
        sample_size = min(num_random_checks, len(remaining))
        positions.update(rng.sample(remaining, sample_size))

    return sorted(positions)


def validate_pair(
    text_path: Path,
    npy_path: Path,
    num_random_checks: int,
    seed: int,
) -> None:
    npy_tokens = np.load(npy_path, mmap_mode="r")
    expected_length = len(npy_tokens)
    positions = build_positions(expected_length, num_random_checks, seed)
    remaining_positions = iter(positions)
    next_position = next(remaining_positions, None)
    checked = 0
    actual_length = 0

    for idx, text_token in enumerate(iter_text_token_ids(text_path)):
        actual_length += 1

        if idx >= expected_length:
            raise ValueError(
                f"{text_path.name} has more tokens than {npy_path.name}: "
                f"text has at least {actual_length}, npy has {expected_length}"
            )

        if next_position is not None and idx == next_position:
            npy_token = int(npy_tokens[idx])
            if text_token != npy_token:
                raise ValueError(
                    f"Mismatch at position {idx} for {text_path.name}: "
                    f"text={text_token}, npy={npy_token}"
                )
            checked += 1
            next_position = next(remaining_positions, None)

    if actual_length != expected_length:
        raise ValueError(
            f"Token count mismatch for {text_path.name}: "
            f"text has {actual_length}, npy has {expected_length}"
        )

    print(
        f"Validated {text_path.name} against {npy_path.name}: "
        f"length={actual_length}, checked_positions={checked}"
    )


def find_default_pairs(directory: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for text_path in sorted(directory.glob("*.txt")):
        npy_path = text_path.with_suffix(".npy")
        if npy_path.exists():
            pairs.append((text_path, npy_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that .npy token-id files match their comma-separated text sources."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/ts/full_encoding_result"),
        help="Directory containing matching .txt and .npy files.",
    )
    parser.add_argument(
        "--text-path", type=Path, help="Path to a specific .txt token-id file."
    )
    parser.add_argument("--npy-path", type=Path, help="Path to the matching .npy file.")
    parser.add_argument(
        "--num-random-checks",
        type=int,
        default=100,
        help="How many random positions to compare in addition to fixed positions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to choose validation positions.",
    )
    args = parser.parse_args()

    if (args.text_path is None) != (args.npy_path is None):
        raise ValueError(
            "Provide both --text-path and --npy-path together, or neither."
        )

    if args.text_path is not None:
        pairs = [(args.text_path, args.npy_path)]
    else:
        pairs = find_default_pairs(args.dir)

    if not pairs:
        raise ValueError("No matching .txt/.npy pairs found to validate.")

    for text_path, npy_path in pairs:
        validate_pair(
            text_path=text_path,
            npy_path=npy_path,
            num_random_checks=args.num_random_checks,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
