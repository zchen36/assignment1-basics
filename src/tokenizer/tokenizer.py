import regex as re
from tokenizer.constants import PAT
import logging
import os
from collections.abc import Iterable, Iterator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ):
        self._vocab = vocab
        self._merges: list[tuple[bytes, bytes]] = merges
        self._special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []

        self._vocab_to_id = {v: k for k, v in vocab.items()}

        for special_token in self._special_tokens:
            if special_token.encode("utf-8") not in self._vocab_to_id:
                logging.info("adding special_token to vocab: %s", special_token)
                self._vocab[len(self._vocab)] = special_token.encode("utf-8")
                self._vocab_to_id[special_token.encode("utf-8")] = len(self._vocab) - 1

    @classmethod
    def from_file(
        cls, vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str] | None = None
    ):
        """vocab and merges files should be in pickle format."""

        with open(vocab_path, "rb") as f:
            vocab = p.load(f)

        with open(merges_path, "rb") as f:
            merges = p.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """encode the input text to tokens."""

        """
        1. break into [paragraph_1]<special_token>[paragraph_2]<special_token>[paragraph_3]
        2. for each paragraph: pre-tokenization -> [words]
        3. create set [words]
        4. dict word_to_bytes_form: [word, list[bytes]], bytes_pair_to_word [bytes, bytes] -> [word]
        5. iterate merges, for each merge pair
            5.1 find all words impacted
            5.2 merge bytes in those words, update word_bytes_form
            5.3 update pair_to_word
        6. get word -> list[id] mapping
        7. encode paragraph
        8. concat paragraphs + special_token
        """

        # 1
        if self._special_tokens:
            pattern = re.compile("|".join(map(re.escape, self._special_tokens)))
            logging.info("regexp used: %s", pattern.pattern)

            paragraphs = pattern.split(text)
            existing_special_tokens = pattern.findall(text)
            logging.info("special tokens found: %s", existing_special_tokens)
            special_token_ids = [self._vocab_to_id[token.encode("utf-8")] for token in existing_special_tokens]
        else:
            paragraphs = [text]
            special_token_ids = []

        # 2
        def _pre_tokenize(paragraph: str, pre_tokenizer_regexp: str) -> list[str]:
            ret: list[str] = []
            for match in re.finditer(pre_tokenizer_regexp, paragraph):
                ret.append(match.group(0))
            return ret

        word_list_list = [_pre_tokenize(paragraph, PAT) for paragraph in paragraphs]

        # 3
        word_set = set()

        for word_list in word_list_list:
            word_set.update(word_list)

        # 4
        word_to_bytes_form: dict[str, list[bytes]] = {}
        bytes_pair_to_word: dict[tuple[bytes, bytes], set[str]] = {}

        for word in word_set:
            word_bytes = word.encode("utf-8")
            word_to_bytes_form[word] = [bytes([b]) for b in word_bytes]

            for i in range(len(word_bytes) - 1):
                pair = (bytes([word_bytes[i]]), bytes([word_bytes[i + 1]]))
                if pair not in bytes_pair_to_word:
                    bytes_pair_to_word[pair] = {word}
                else:
                    bytes_pair_to_word[pair].add(word)

        for bytes_pair in self._merges:
            if bytes_pair not in bytes_pair_to_word:
                continue

            # 5.1
            words_impacted = bytes_pair_to_word[bytes_pair]

            # 5.2
            for word in words_impacted:
                word_bytes_form = word_to_bytes_form[word]
                new_word_bytes_form: list[bytes] = []
                i = 0
                while i < len(word_bytes_form):
                    if i == len(word_bytes_form) - 1:
                        new_word_bytes_form.append(word_bytes_form[i])
                        break

                    first, second = word_bytes_form[i], word_bytes_form[i + 1]
                    if first == bytes_pair[0] and second == bytes_pair[1]:
                        new_word_bytes_form.append(first + second)
                        i += 2
                    else:
                        new_word_bytes_form.append(first)
                        i += 1
                word_to_bytes_form[word] = new_word_bytes_form

                # 5.3
                for i in range(len(new_word_bytes_form) - 1):
                    pair = (new_word_bytes_form[i], new_word_bytes_form[i + 1])
                    if pair not in bytes_pair_to_word:
                        bytes_pair_to_word[pair] = {word}
                    else:
                        bytes_pair_to_word[pair].add(word)

            # 5.3
            del bytes_pair_to_word[bytes_pair]

        # 6
        word_to_ids = {
            word: [self._vocab_to_id[b] for b in bytes_list] for word, bytes_list in word_to_bytes_form.items()
        }

        # 7
        id_list_list_list = [[word_to_ids[word] for word in word_list] for word_list in word_list_list]

        # 8
        concat_paragraph_special_token = [
            id for pair in zip(id_list_list_list, special_token_ids) for id in pair
        ] + id_list_list_list[-1]

        def _flatten(nested) -> list[int]:
            ret = []
            for item in nested:
                if isinstance(item, list):
                    ret.extend(_flatten(item))
                else:
                    ret.append(item)
            return ret

        return _flatten(concat_paragraph_special_token)

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self._vocab[id] for id in ids]
        bytes_form = b"".join(bytes_list)
        return bytes_form.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
