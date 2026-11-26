from filinglens.serve.llama_client import chat_request, content_of, sum_timings


def test_chat_request_uses_the_eval_settings():
    conv = [{"role": "user", "content": "x"}]
    body = chat_request(conv, {"type": "object"})
    assert body["messages"] is conv
    assert body["temperature"] == 0.0 and body["max_tokens"] == 400 and body["seed"] == 0
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"schema": {"type": "object"}},
    }
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_content_of_handles_missing_parts():
    assert content_of({"choices": [{"message": {"content": '{"a": 1}'}}]}) == '{"a": 1}'
    assert content_of({"choices": []}) == ""
    assert content_of({}) == ""
    assert content_of({"choices": [{"message": {"content": None}}]}) == ""


def test_sum_timings_totals_and_rates():
    timings = [
        {"prompt_n": 1000, "prompt_ms": 2000, "predicted_n": 100, "predicted_ms": 4000},
        {"prompt_n": 500, "prompt_ms": 1000, "predicted_n": 50, "predicted_ms": 2000},
    ]
    assert sum_timings(timings) == {
        "requests_with_timings": 2,
        "prompt_tokens": 1500,
        "generated_tokens": 150,
        "prompt_tok_per_s": 500.0,
        "generate_tok_per_s": 25.0,
    }


def test_sum_timings_empty():
    result = sum_timings([])
    assert result["requests_with_timings"] == 0
    assert result["prompt_tok_per_s"] is None and result["generate_tok_per_s"] is None
