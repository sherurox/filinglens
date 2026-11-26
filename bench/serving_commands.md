# Commands that produced bench/serving.csv and bench/serving.png

Rule (committed before the test run): `bench/m5_serving_rule.json`

- load: `bash scripts/fl_m5_serving.sh load "<power mode>"`
- test: `bash scripts/fl_m5_serving.sh test "<power mode>"`
- table: `bash scripts/fl_m5_serving.sh table`

## Server per build (copied from bench/load_<build>.json)

The load settings are the script's (levels 1/4/16, 64 requests, 4 warm-up)

- merged-bf16: server `python -m filinglens.serve.vllm_server --build merged-bf16`
- fp8-dynamic: server `python -m filinglens.serve.vllm_server --build fp8-dynamic`
- gptq-w4a16: server `python -m filinglens.serve.vllm_server --build gptq-w4a16`
- awq-w4a16: server `python -m filinglens.serve.vllm_server --build awq-w4a16`
- gguf-q4_k_m: server `llama-server -m data/interim/serve/gguf/student-q4_k_m.gguf -np 16 -c 49152 -t 12 -tb 24 --jinja --seed 0`
