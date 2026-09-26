"""Admission for the runtime's pinned byte-level BPE tokenizer profile."""

from typing import Any

from .errors import ConversionError
from .jsonio import json_bytes

SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
BOS = "<|startoftext|>"


def bytelevel_alphabet() -> set[str]:
    selected = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    return {chr(value) for value in selected} | {
        chr(256 + index) for index in range(256 - len(selected))
    }


def profile_components() -> tuple[dict, dict, dict]:
    pre = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": SPLIT},
                "behavior": "Isolated",
                "invert": False,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": False,
            },
        ],
    }
    start = {"SpecialToken": {"id": BOS, "type_id": 0}}
    post = {
        "type": "Sequence",
        "processors": [
            {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": False,
                "use_regex": True,
            },
            {
                "type": "TemplateProcessing",
                "single": [start, {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [
                    start,
                    {"Sequence": {"id": "A", "type_id": 0}},
                    start,
                    {"Sequence": {"id": "B", "type_id": 0}},
                ],
                "special_tokens": {BOS: {"id": BOS, "ids": [1], "tokens": [BOS]}},
            },
        ],
    }
    decoder = {
        "type": "Sequence",
        "decoders": [
            {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True,
            },
        ],
    }
    return pre, post, decoder


def _id(value: Any, vocab_size: int) -> int:
    if type(value) is not int or not 0 <= value < min(64402, vocab_size):
        raise ConversionError("token ID is outside mapped runtime/model vocabulary")
    return value


def validate_tokenizer(value: Any, vocab_size: int) -> None:
    if not isinstance(value, dict):
        raise ConversionError("tokenizer root must be an object")
    expected = {
        "version",
        "truncation",
        "padding",
        "added_tokens",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
        "model",
    }
    if set(value) != expected or value["version"] != "1.0":
        raise ConversionError("unsupported tokenizer root fields or version")
    if any(value[name] is not None for name in ("truncation", "padding", "normalizer")):
        raise ConversionError(
            "tokenizer truncation, padding and normalizer must be null"
        )
    for name, expected in zip(
        ("pre_tokenizer", "post_processor", "decoder"),
        profile_components(),
        strict=True,
    ):
        if json_bytes(value[name]) != json_bytes(expected):
            raise ConversionError(f"unsupported tokenizer {name} profile")
    model = value["model"]
    fields = {
        "type",
        "dropout",
        "unk_token",
        "continuing_subword_prefix",
        "end_of_word_suffix",
        "fuse_unk",
        "byte_fallback",
        "ignore_merges",
        "vocab",
        "merges",
    }
    if not isinstance(model, dict) or set(model) != fields or model["type"] != "BPE":
        raise ConversionError("unsupported BPE model fields")
    if any(
        model[name] is not None
        for name in (
            "dropout",
            "unk_token",
            "continuing_subword_prefix",
            "end_of_word_suffix",
        )
    ):
        raise ConversionError("unsupported BPE mode")
    if any(
        model[name] is not False
        for name in ("fuse_unk", "byte_fallback", "ignore_merges")
    ):
        raise ConversionError("unsupported BPE mode")
    vocab = model["vocab"]
    if not isinstance(vocab, dict) or not vocab:
        raise ConversionError("tokenizer vocabulary must be a non-empty object")
    alphabet, ids = bytelevel_alphabet(), set()
    for content, token_id in vocab.items():
        token_id = _id(token_id, vocab_size)
        if not content or not set(content) <= alphabet or token_id in ids:
            raise ConversionError("empty/undecodable vocabulary symbol or duplicate ID")
        ids.add(token_id)
    if vocab.get(BOS) != 1:
        raise ConversionError("BOS vocabulary entry must have ID 1")
    merges, pairs = model["merges"], set()
    if not isinstance(merges, list):
        raise ConversionError("BPE merges must be an array of pairs")
    for pair in merges:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(x) is not str for x in pair)
        ):
            raise ConversionError("BPE merges must be pairs of strings")
        left, right = pair
        if left not in vocab or right not in vocab or left + right not in vocab:
            raise ConversionError("merge symbol/result is missing from vocabulary")
        if tuple(pair) in pairs:
            raise ConversionError("duplicate BPE merge")
        pairs.add(tuple(pair))
    tokens = value["added_tokens"]
    if not isinstance(tokens, list):
        raise ConversionError("added_tokens must be an array")
    seen_ids, contents, has_bos = set(), set(), False
    fields = {
        "id",
        "content",
        "single_word",
        "lstrip",
        "rstrip",
        "normalized",
        "special",
    }
    for token in tokens:
        if not isinstance(token, dict) or set(token) != fields:
            raise ConversionError("unsupported added-token fields")
        token_id, content = _id(token["id"], vocab_size), token["content"]
        if type(content) is not str or not content:
            raise ConversionError("added-token content must be non-empty text")
        for flag in fields - {"id", "content"}:
            if type(token[flag]) is not bool:
                raise ConversionError("added-token flags must be Boolean")
        if any(token[flag] for flag in ("single_word", "lstrip", "rstrip")):
            raise ConversionError("added-token matching flags are unsupported")
        if token_id in seen_ids or content in contents:
            raise ConversionError("duplicate added-token ID or content")
        if content in vocab and vocab[content] != token_id:
            raise ConversionError("added-token ID contradicts vocabulary")
        if token_id in ids and vocab.get(content) != token_id:
            raise ConversionError("added-token ID conflicts with a byte-level token")
        seen_ids.add(token_id)
        contents.add(content)
        has_bos |= token_id == 1 and content == BOS and token["special"]
    if not has_bos:
        raise ConversionError("BOS must have a matching special added-token definition")
