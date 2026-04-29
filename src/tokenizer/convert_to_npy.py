from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def convert_csv_token_ids_to_npy(
    input_path: Path, output_path: Path, dtype: np.dtype
) -> int:
    text = input_path.read_text(encoding="utf-8")
    values = np.fromstring(text, dtype=dtype, sep=",")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, values)
    return len(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a comma-separated token-id text file to a .npy array."
    )
    parser.add_argument(
        "input_path", type=Path, help="Path to the comma-separated token-id text file."
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Path for the output .npy file. Defaults to the input path with a .npy suffix.",
    )
    parser.add_argument(
        "--dtype",
        default="uint16",
        choices=["int16", "int32", "int64", "uint16", "uint32", "uint64"],
        help="NumPy dtype to store token ids in.",
    )
    args = parser.parse_args()

    output_path = args.output_path or args.input_path.with_suffix(".npy")
    dtype = np.dtype(args.dtype)

    token_count = convert_csv_token_ids_to_npy(
        input_path=args.input_path,
        output_path=output_path,
        dtype=dtype,
    )

    print(f"Wrote {token_count} token ids to {output_path}")
    print(f"Load normally with: np.load('{output_path}')")
    print(f"Memmap later with: np.load('{output_path}', mmap_mode='r')")


if __name__ == "__main__":
    main()
