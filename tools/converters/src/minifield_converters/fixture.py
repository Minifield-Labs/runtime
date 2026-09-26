"""Deterministic synthetic LFM2 input generator for loader and converter tests."""

from pathlib import Path

import numpy as np

from .config import validate_config
from .container import Tensor, write_tensors
from .jsonio import json_bytes
from .paths import MANIFEST, SCHEMA, staged_destination
from .tokenizer import BOS, profile_components


def fixture_config() -> dict:
    return {
        "model_type": "lfm2",
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "conv_L_cache": 3,
        "vocab_size": 32,
        "num_hidden_layers": 2,
        "layer_types": ["conv", "full_attention"],
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
        "norm_eps": 1e-5,
        "block_auto_adjust_ff_dim": False,
        "conv_bias": False,
        "dtype": "float32",
        "bos_token_id": 1,
    }


def fixture_tokenizer() -> dict:
    pre, post, decoder = profile_components()
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "normalizer": None,
        "pre_tokenizer": pre,
        "post_processor": post,
        "decoder": decoder,
        "added_tokens": [
            {
                "id": 1,
                "content": BOS,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {"a": 0, BOS: 1, "b": 2, "ab": 3, "x": 4},
            "merges": [["a", "b"]],
        },
    }


def write_fixture(
    output: Path, *, overwrite: bool = False, classes: int | None = None
) -> None:
    """Write a tiny synthetic two-layer model. It isn't a trained model."""
    config = fixture_config()
    tensors = {}
    for index, (name, shape) in enumerate(
        sorted(validate_config(config).inventory(classes).items())
    ):
        if len(shape) == 1:
            values = np.ones(shape, dtype=np.float32)
        else:
            # Exactly representable ternary values, with multiple row-local groups.
            values = (((np.arange(np.prod(shape)) + index) % 3) - 1).astype(np.float32)
            values = (values * np.float32(0.125)).reshape(shape)
        tensors[name] = Tensor.from_array(values)
    with staged_destination(output, None, overwrite) as pending:
        (pending / "tokenizer").mkdir()
        (pending / "config.json").write_bytes(json_bytes(config))
        (pending / "tokenizer/tokenizer.json").write_bytes(
            json_bytes(fixture_tokenizer())
        )
        write_tensors(pending / "model.safetensors", tensors, {"format": "pt"})
        # This marks a generated source directory so explicit fixture replacement
        # can share the same narrow destination guard. It's not a validated bundle.
        (pending / MANIFEST).write_bytes(
            json_bytes({"schema": SCHEMA, "purpose": "synthetic_source_fixture"})
        )
