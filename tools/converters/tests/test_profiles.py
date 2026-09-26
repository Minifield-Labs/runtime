import copy

import pytest

from minifield_converters.config import validate_config
from minifield_converters.errors import ConversionError
from minifield_converters.fixture import fixture_config, fixture_tokenizer
from minifield_converters.jsonio import parse_json
from minifield_converters.tokenizer import validate_tokenizer


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "llama"),
        ("conv_bias", True),
        ("tie_word_embeddings", False),
        ("num_attention_heads", 3),
        ("num_key_value_heads", 3),
        ("norm_eps", 0),
        ("rope_scaling", {"factor": 2}),
        ("dtype", "float16"),
        ("layer_types", ["conv", "attention"]),
        ("hidden_size", True),
        ("block_dim", 64),
        ("block_use_swiglu", False),
    ],
)
def test_unsupported_config_profiles(field, value):
    config = fixture_config()
    config[field] = value
    with pytest.raises(ConversionError):
        validate_config(config)


def test_aliases_and_effective_ff_dimensions_match_runtime():
    config = fixture_config()
    config["num_heads"] = config.pop("num_attention_heads")
    config["tie_embedding"] = config.pop("tie_word_embeddings")
    config["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": config.pop("rope_theta"),
    }
    config.update(
        {
            "block_auto_adjust_ff_dim": True,
            "intermediate_size": 192,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 128,
        }
    )
    result = validate_config(config)
    assert result.intermediate == 128
    assert result.inventory()["model.layers.1.feed_forward.w2.weight"] == (128, 128)
    config["num_attention_heads"] = 2
    with pytest.raises(ConversionError, match="conflicting"):
        validate_config(config)


def test_profile_rejects_numeric_boolean_substitutes_and_duplicate_json_keys():
    tokenizer = fixture_tokenizer()
    validate_tokenizer(tokenizer, 32)
    malformed = copy.deepcopy(tokenizer)
    malformed["pre_tokenizer"]["pretokenizers"][0]["invert"] = 0
    with pytest.raises(ConversionError, match="profile"):
        validate_tokenizer(malformed, 32)
    with pytest.raises(ConversionError, match="duplicate"):
        parse_json(b'{"model_type":"lfm2","model_type":"other"}')
    with pytest.raises(ConversionError, match="non-finite"):
        parse_json(b'{"norm_eps":NaN}')


def test_tokenizer_mappings_merges_and_bos_are_validated():
    for mutate in (
        lambda value: value["model"]["vocab"].update({"x": 65535}),
        lambda value: value["model"]["vocab"].update({"x": 0}),
        lambda value: value["model"]["merges"].append(["a", "x"]),
        lambda value: value["added_tokens"].clear(),
        lambda value: value.update({"normalizer": {"type": "Lowercase"}}),
    ):
        tokenizer = fixture_tokenizer()
        mutate(tokenizer)
        with pytest.raises(ConversionError):
            validate_tokenizer(tokenizer, 32)


@pytest.mark.parametrize(
    "source",
    [b'"\\ud800"', b'{"\\udfff":1}', b"1e400", b"[" * 130 + b"null" + b"]" * 130],
)
def test_json_admission_rejects_invalid_unicode_nonfinite_numbers_and_deep_data(source):
    with pytest.raises(ConversionError):
        parse_json(source)
