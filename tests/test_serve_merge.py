from filinglens.serve.merge import (
    config_dtype,
    expected_changed,
    is_tokenizer_file,
    module_kind,
    only_base_ok,
)


def test_module_kind_strips_layer_index():
    assert module_kind("model.layers.3.self_attn.q_proj.weight") == "self_attn.q_proj"
    assert module_kind("model.layers.27.mlp.down_proj.weight") == "mlp.down_proj"
    assert module_kind("model.embed_tokens.weight") == "model.embed_tokens.weight"


def test_expected_changed_list_vs_pattern():
    seven = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert expected_changed(seven, 28) == 196
    assert expected_changed([*seven, "q_proj"], 28) == 196  # duplicates don't count twice
    assert expected_changed("all-linear", 28) is None


def test_config_dtype_both_spellings():
    assert config_dtype({"dtype": "bfloat16"}) == "bfloat16"
    assert config_dtype({"torch_dtype": "float32"}) == "float32"
    assert config_dtype({}) is None


def test_tokenizer_files_but_not_weights_or_config():
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "generation_config.json",
    ]:
        assert is_tokenizer_file(name), name
    for name in ["config.json", "model.safetensors", "README.md", "LICENSE"]:
        assert not is_tokenizer_file(name), name


def test_only_base_ok_accepts_only_a_verified_tied_head():
    assert only_base_ok([], tie_word_embeddings=False, head_equal=False)
    assert only_base_ok(["lm_head.weight"], tie_word_embeddings=True, head_equal=True)
    assert not only_base_ok(["lm_head.weight"], tie_word_embeddings=False, head_equal=True)
    assert not only_base_ok(["lm_head.weight"], tie_word_embeddings=True, head_equal=False)
    two = ["lm_head.weight", "model.norm.weight"]
    assert not only_base_ok(two, tie_word_embeddings=True, head_equal=True)
