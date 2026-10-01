"""The supported LFM2 numerical profile and exact physical tensor inventory."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ConversionError


def _integer(value: Any, name: str, maximum: int = 0xFFFFFFFF) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ConversionError(f"{name} must be a positive integer <= {maximum}")
    return value


def _positive_f32(value: Any, name: str) -> np.float32:
    if type(value) not in (int, float):
        raise ConversionError(f"{name} must be numeric")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.float32(value)
    except (OverflowError, ValueError) as error:
        raise ConversionError(f"{name} must be positive finite FP32") from error
    if not np.isfinite(converted) or converted <= 0:
        raise ConversionError(f"{name} must be positive finite FP32")
    return converted


def _alias(config: dict[str, Any], primary: str, alias: str) -> Any:
    if primary in config and alias in config:
        if type(config[primary]) is not type(config[alias]):
            raise ConversionError(f"conflicting aliases: {primary}, {alias}")
        if config[primary] != config[alias]:
            raise ConversionError(f"conflicting aliases: {primary}, {alias}")
    return config.get(primary, config.get(alias))


@dataclass(frozen=True)
class Lfm2Config:
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    conv_width: int
    vocab: int
    layers: tuple[str, ...]
    dtype: str

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def inventory(self, classes: int | None = None) -> dict[str, tuple[int, ...]]:
        hidden, ff = self.hidden, self.intermediate
        tensors = {
            "model.embed_tokens.weight": (self.vocab, hidden),
            "model.embedding_norm.weight": (hidden,),
        }
        for index, layer in enumerate(self.layers):
            prefix = f"model.layers.{index}"
            if layer == "conv":
                tensors.update(
                    {
                        f"{prefix}.conv.conv.weight": (hidden, 1, self.conv_width),
                        f"{prefix}.conv.in_proj.weight": (3 * hidden, hidden),
                        f"{prefix}.conv.out_proj.weight": (hidden, hidden),
                    }
                )
            else:
                for name in ("q_layernorm", "k_layernorm"):
                    tensors[f"{prefix}.self_attn.{name}.weight"] = (self.head_dim,)
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    rows = (
                        self.kv_heads * self.head_dim
                        if name in ("k_proj", "v_proj")
                        else hidden
                    )
                    tensors[f"{prefix}.self_attn.{name}.weight"] = (rows, hidden)
            for name, shape in (
                ("w1", (ff, hidden)),
                ("w2", (hidden, ff)),
                ("w3", (ff, hidden)),
            ):
                tensors[f"{prefix}.feed_forward.{name}.weight"] = shape
            for name in ("ffn_norm", "operator_norm"):
                tensors[f"{prefix}.{name}.weight"] = (hidden,)
        if classes is not None:
            tensors["classification_head.weight"] = (
                _integer(classes, "classes", 65536),
                hidden,
            )
        return tensors


def is_quantized_role(name: str, shape: tuple[int, ...]) -> bool:
    """The classifier head is always dense, including packed backbone bundles."""
    return len(shape) == 2 and name != "classification_head.weight"


def validate_config(value: Any, *, allow_f16: bool = False) -> Lfm2Config:
    if not isinstance(value, dict) or value.get("model_type") != "lfm2":
        raise ConversionError("only config model_type=lfm2 is supported")
    config = value
    for field in ("conv_bias", "block_auto_adjust_ff_dim", "block_use_swiglu"):
        if field in config and type(config[field]) is not bool:
            raise ConversionError(f"{field} must be Boolean")
    if config.get("conv_bias", False) or not config.get("block_use_swiglu", True):
        raise ConversionError(
            "biased convolution and non-SwiGLU blocks are unsupported"
        )
    if config.get("rope_scaling") is not None:
        raise ConversionError("scaled RoPE is unsupported")
    hidden = _integer(config.get("hidden_size"), "hidden_size")
    raw_ff = _integer(config.get("intermediate_size"), "intermediate_size")
    for alias, expected in (
        ("block_dim", hidden),
        ("conv_dim", hidden),
        ("block_ff_dim", raw_ff),
    ):
        if alias in config and _integer(config[alias], alias) != expected:
            raise ConversionError(f"conflicting dimension alias: {alias}")
    intermediate = raw_ff
    if config.get("block_auto_adjust_ff_dim", False):
        multiplier = config.get("block_ffn_dim_multiplier")
        if type(multiplier) not in (int, float) or not math.isfinite(multiplier):
            raise ConversionError("block_ffn_dim_multiplier must be finite")
        if multiplier <= 0 or raw_ff * 2 > 0xFFFFFFFF:
            raise ConversionError("FF dimension adjustment is invalid")
        multiple = _integer(config.get("block_multiple_of"), "block_multiple_of")
        scaled = (raw_ff * 2 // 3) * multiplier
        if not math.isfinite(scaled) or scaled > 0xFFFFFFFF:
            raise ConversionError("effective FF dimension exceeds u32")
        intermediate = _integer(
            ((int(scaled) + multiple - 1) // multiple) * multiple,
            "effective FF dimension",
        )
    heads = _integer(_alias(config, "num_attention_heads", "num_heads"), "heads")
    kv_heads = _integer(config.get("num_key_value_heads"), "num_key_value_heads")
    if hidden % heads or heads % kv_heads or (hidden // heads) % 2:
        raise ConversionError(
            "invalid attention dimensions or odd split-half RoPE width"
        )
    tied = _alias(config, "tie_embedding", "tie_word_embeddings")
    if tied is not True:
        raise ConversionError(
            "only explicitly tied embedding/LM-head models are supported"
        )
    count = _integer(config.get("num_hidden_layers"), "num_hidden_layers")
    layers = config.get("layer_types")
    if not isinstance(layers, list) or len(layers) != count:
        raise ConversionError("layer_types must match num_hidden_layers")
    if any(layer not in ("conv", "full_attention") for layer in layers):
        raise ConversionError("unsupported LFM2 layer type")
    norm = _positive_f32(config.get("norm_eps"), "norm_eps")
    _positive_f32(config.get("block_norm_eps", norm.item()), "block_norm_eps")
    rope = config.get("rope_parameters")
    legacy_theta = config.get("rope_theta")
    if rope is not None:
        if not isinstance(rope, dict) or rope.get("rope_type", "default") != "default":
            raise ConversionError("only default rope_parameters are supported")
        if "factor" in rope or "rope_scaling" in rope:
            raise ConversionError("scaled RoPE is unsupported")
        theta = _positive_f32(rope.get("rope_theta"), "rope_parameters.rope_theta")
        if legacy_theta is not None and theta != _positive_f32(
            legacy_theta, "rope_theta"
        ):
            raise ConversionError("conflicting RoPE aliases")
    else:
        _positive_f32(legacy_theta, "rope_theta")
    if "max_position_embeddings" in config:
        _integer(
            config["max_position_embeddings"], "max_position_embeddings", 2**64 - 1
        )
    dtype = config.get("dtype", "float32")
    dtypes = {"float32": "F32", "f32": "F32", "bfloat16": "BF16", "bf16": "BF16"}
    if allow_f16:
        dtypes.update({"float16": "F16", "f16": "F16"})
    if dtype not in dtypes:
        raise ConversionError("only dense F32 and BF16 source storage is supported")
    return Lfm2Config(
        hidden,
        intermediate,
        heads,
        kv_heads,
        _integer(config.get("conv_L_cache"), "conv_L_cache"),
        _integer(config.get("vocab_size"), "vocab_size", 65536),
        tuple(layers),
        dtypes[dtype],
    )
