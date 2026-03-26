"""
non multi-process version:
1) read input, by chunks
2) pre-tokenization: re.finditer
3) build each chunk
4) merge
"""

import os
from typing import BinaryIO
import regex as re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from functools import total_ordering
from sortedcontainers import SortedList
from .constants import PAT, DESIRED_NUM_CHUNKS
import logging

from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass(order=True, frozen=True)
class TokenPair:
    first: bytes
    second: bytes


@total_ordering
@dataclass
class TokenPairDetails:
    bytes_pair: TokenPair
    count: int
    words_with_pair: set[str]

    def __lt__(self, other):
        if self.count != other.count:
            return self.count < other.count
        else:
            return self.bytes_pair < other.bytes_pair

    def __eq__(self, other):
        return self.count == other.count and self.bytes_pair == other.bytes_pair


@dataclass
class WordStatus:
    count: int
    token_form: list[bytes]


@dataclass
class MergeStatus:
    vocab: list[bytes]
    merges: list[TokenPair]
    word_count: dict[str, WordStatus]
    # These 2 work together as a heap with random access
    index: dict[TokenPair, TokenPairDetails]
    CountHeap: SortedList[TokenPairDetails]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_one_chunk(chunk_str: str, split_special_token: bytes, pre_tokenizer_regexp: str) -> dict[str, int]:
    special_token_str = split_special_token.decode("utf-8")
    chunk_str_list = chunk_str.split(special_token_str)

    ret: dict[str, int] = {}
    for clean_chunk_str in chunk_str_list:
        for match in re.finditer(pre_tokenizer_regexp, clean_chunk_str):
            token = match.group(0)
            ret[token] = ret.get(token, 0) + 1
    return ret


def pre_tokenize_file(
    file_path: Path,
    desired_num_chunks: int,
    split_special_token: bytes,
    pre_tokenizer_regexp: str,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    with file_path.open("rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, split_special_token)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk: str = f.read(end - start).decode("utf-8", errors="ignore")
                futures.append(
                    executor.submit(pre_tokenize_one_chunk, chunk, split_special_token, pre_tokenizer_regexp)
                )

            for future in as_completed(futures):
                counter.update(future.result())
    return dict(counter)


def train_bpe_from_word_count(
    word_count: dict[str, int], split_special_token: bytes, vocab_size: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Run bpe and return vocab and merges.

    Args:
        word_count: a dic for words count.
        split_special_token
        vocab_size： max vocab size.

    Returns:
        dict: as vocab
        list: as merges
    """

    """
    1. init vocab and merges
    2. build status
    3. pop the next token
    4. check vocab size and update status
    """

    vocab: list[bytes] = [split_special_token]
    vocab.extend([bytes([i]) for i in range(256)])

    merges: list[TokenPair] = []
    word_count_and_bytes_form: dict[str, WordStatus] = {}
    index: dict[TokenPair, TokenPairDetails] = {}
    CountHeap: SortedList[TokenPairDetails] = SortedList()

    bpe_status: MergeStatus = MergeStatus(vocab, merges, word_count_and_bytes_form, index, CountHeap)

    # init the status
    for word, count in word_count.items():
        word_bytes = word.encode("utf-8")
        bpe_status.word_count[word] = WordStatus(count, [bytes([b]) for b in word_bytes])

        for i in range(len(word_bytes) - 1):
            pair = TokenPair(bytes([word_bytes[i]]), bytes([word_bytes[i + 1]]))

            if pair not in bpe_status.index:
                bpe_status.index[pair] = TokenPairDetails(pair, count, {word})
                bpe_status.CountHeap.add(bpe_status.index[pair])

            else:
                original_details = bpe_status.index[pair]
                # delete from heap
                bpe_status.CountHeap.remove(original_details)
                # update node
                original_details.count += count
                original_details.words_with_pair.add(word)
                # update index
                bpe_status.index[pair] = original_details
                # add back to heap
                bpe_status.CountHeap.add(bpe_status.index[pair])

    # do one merge
    while len(bpe_status.vocab) < vocab_size and bpe_status.CountHeap:
        most_popular_pair_detail = bpe_status.CountHeap.pop()
        del bpe_status.index[most_popular_pair_detail.bytes_pair]

        # 1. update vocab and merges
        bpe_status.vocab.append(most_popular_pair_detail.bytes_pair.first + most_popular_pair_detail.bytes_pair.second)
        bpe_status.merges.append(most_popular_pair_detail.bytes_pair)

        # 2. update status

        for word in most_popular_pair_detail.words_with_pair:
            # 2.1 update word token_form
            word_status = bpe_status.word_count[word]
            old_token_form = word_status.token_form.copy()
            new_token_form: list[bytes] = []
            i = 0
            while i < len(word_status.token_form):
                if i == len(word_status.token_form) - 1:
                    new_token_form.append(word_status.token_form[i])
                    break

                first, second = word_status.token_form[i], word_status.token_form[i + 1]
                if (
                    first == most_popular_pair_detail.bytes_pair.first
                    and second == most_popular_pair_detail.bytes_pair.second
                ):
                    new_token_form.append(bpe_status.vocab[-1])
                    i += 2
                else:
                    new_token_form.append(first)
                    i += 1
            word_status.token_form = new_token_form

            def _get_token_pair_to_count(word_token_form: list[bytes]) -> dict[TokenPair, int]:
                ret: dict[TokenPair, int] = {}
                for i in range(len(word_token_form) - 1):
                    p = TokenPair(word_token_form[i], word_token_form[i + 1])
                    ret[p] = ret.get(p, 0) + 1
                return ret

            def _get_token_pair_delta(
                old_count: dict[TokenPair, int], new_count: dict[TokenPair, int]
            ) -> dict[TokenPair, int]:
                ret: dict[TokenPair, int] = new_count.copy()
                for t, c in old_count.items():
                    if t in new_count:
                        ret[t] -= c
                    else:
                        ret[t] = -c
                return ret

            old_token_pair_count = _get_token_pair_to_count(old_token_form)
            new_token_pair_count = _get_token_pair_to_count(word_status.token_form)
            delta_count = _get_token_pair_delta(old_token_pair_count, new_token_pair_count)

            for t, c in delta_count.items():
                if t == most_popular_pair_detail.bytes_pair:
                    continue

                if c == 0:
                    continue

                if t in bpe_status.index:
                    # could > 0 if in the next word when it's already added as a new before
                    # assert c < 0
                    original_details = bpe_status.index[t]
                    # delete
                    bpe_status.CountHeap.remove(original_details)
                    # update
                    original_details.count += c * word_status.count
                    original_details.words_with_pair.add(word)
                    if original_details.count == 0:
                        del bpe_status.index[t]
                    else:
                        # update index
                        bpe_status.index[t] = original_details
                        # add back
                        bpe_status.CountHeap.add(bpe_status.index[t])
                else:
                    assert c > 0
                    bpe_status.index[t] = TokenPairDetails(t, c * word_status.count, {word})
                    bpe_status.CountHeap.add(bpe_status.index[t])

    return dict(enumerate(bpe_status.vocab)), [(pair.first, pair.second) for pair in bpe_status.merges]


def train_bpe_from_file(
    file_path: Path,
    # desired_num_chunks: int,
    vocab_size: int,
    split_special_token: list[str],
    # pre_tokenizer_regexp: str,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    word_count = pre_tokenize_file(file_path, DESIRED_NUM_CHUNKS, split_special_token[0].encode("utf-8"), PAT)
    return train_bpe_from_word_count(word_count, split_special_token[0].encode("utf-8"), vocab_size)
