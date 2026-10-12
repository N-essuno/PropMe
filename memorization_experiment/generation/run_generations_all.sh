#!/usr/bin/env bash
# Generate completions for every model with vLLM: the generic and specific prompt sets,
# the prefix prompts and the two prompt-free settings.
#
# For each model this starts `vllm serve` (the `vllm` executable on PATH), sends all
# of the model's runs to it at once so the GPU never idles between them, then stops
# the server. Runs whose output exists are skipped, so a failed or interrupted model
# can simply be rerun.
#
#   model        HF id @ revision                                      prompt sets         prefix prompts            minimal cues
#   dfm-stage1   danish-foundation-models/dfm-decoder-open-v0-7b-pt @ stage1  *_dw, *_cp   dynaword2, commonpile   da, en
#   dfm-stage2   ... @ stage2                                           *_dw, *_cp         dynaword2, commonpile     da, en
#   dfm-main     ... @ main                                             *_dw, *_cp         dynaword2, commonpile     da, en
#   comma-2t     common-pile/comma-v0.1-2t @ main                       *_cp               commonpile                en
#   olmo3-32b    allenai/Olmo-3-1125-32B @ main                         *_d3               dolma3                    en
#
# with *_x = generic_x (Tatoeba) and specific_x (unseen-source sentences), both built in
# 00_prepare_data/propensity_settings,
# plus one unconditional run per model.
#
# Usage, from anywhere (all models, or the ones named):
#   bash memorization_experiment/generation/run_generations_all.sh
#   bash memorization_experiment/generation/run_generations_all.sh dfm-main comma-2t
#
# Run it in a Python environment with vLLM installed (and CUDA_HOME / nvcc available
# for FlashInfer's kernel compilation).
#
# Environment overrides: NUM_GENERATIONS (per prompt, default 10), FREE_GENERATIONS
# (per prompt-free run, default 10000: as many as the 1000 prompts x 10 of a prompted setting), PORT (default 8000), OVERWRITE=1.
#
# Outputs, in memorization_experiment/data next to the prompts (<dataset>: dynaword2, commonpile, dolma3):
#   <dataset>/<setting>/generations/<model>/<setting>_generations.json   (generic, specific)
#   <dataset>/prefix/generations/<model>/<corpus>_prefix_generations.json
#   prompt_free/<run>/generations/<model>/<run>_generations.json         (unconditional, minimal_cue_<lang>)
# plus vllm_server.log and one <run>.log per run in logs/generation/<model>.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NUM_GENERATIONS="${NUM_GENERATIONS:-10}"
FREE_GENERATIONS="${FREE_GENERATIONS:-10000}"
PORT="${PORT:-8000}"
OVERWRITE="${OVERWRITE:-0}"
API_BASE="http://127.0.0.1:${PORT}/v1"
SEED=42
TEMPERATURE=0.7
TOP_P=0.95
MAX_NEW_TOKENS=256

DATA_DIR=memorization_experiment/data
LOGS_DIR=logs/generation
# Prompt-set corpus -> dataset folder in DATA_DIR, which holds <setting>/<setting>_prompts.jsonl.
declare -A DATASET_DIR=([dw]=dynaword2 [cp]=commonpile [d3]=dolma3)

DFM=danish-foundation-models/dfm-decoder-open-v0-7b-pt
declare -A HF_ID=(
    [dfm-stage1]=$DFM [dfm-stage2]=$DFM [dfm-main]=$DFM
    [comma-2t]=common-pile/comma-v0.1-2t
    [olmo3-32b]=allenai/Olmo-3-1125-32B
)
declare -A REVISION=(
    [dfm-stage1]=stage1 [dfm-stage2]=stage2 [dfm-main]=main
    [comma-2t]=main [olmo3-32b]=main
)
# Prompt-set corpora (cp, d3, dw), prefix prompt files (<corpus dir>/<stem>) and cue languages.
declare -A CORPORA=(
    [dfm-stage1]="dw cp" [dfm-stage2]="dw cp" [dfm-main]="dw cp"
    [comma-2t]="cp" [olmo3-32b]="d3"
)
declare -A PREFIXES=(
    [dfm-stage1]="dynaword2/dynaword commonpile/commonpile"
    [dfm-stage2]="dynaword2/dynaword commonpile/commonpile"
    [dfm-main]="dynaword2/dynaword commonpile/commonpile"
    [comma-2t]="commonpile/commonpile"
    [olmo3-32b]="dolma3/dolma3"
)
declare -A CUE_LANGS=(
    [dfm-stage1]="da en" [dfm-stage2]="da en" [dfm-main]="da en"
    [comma-2t]="en" [olmo3-32b]="en"
)
ALL_MODELS=(dfm-stage1 dfm-stage2 dfm-main comma-2t olmo3-32b)

# vLLM 0.25's Olmo 3 code reads a flat config.rope_parameters, but transformers 5 nests it
# by layer type and the model fails to load (KeyError: 'rope_theta'). The flat override is
# the full-attention entry: vLLM uses it on the full-attention layers and only its
# rope_theta (plain RoPE) on the sliding-window layers, as in the original config.
OLMO3_ROPE='{"rope_parameters": {"rope_type": "yarn", "rope_theta": 500000, "factor": 8.0, "original_max_position_embeddings": 8192, "attention_factor": 1.2079441541679836, "beta_fast": 32.0, "beta_slow": 1.0}}'
declare -A SERVE_ARGS=([olmo3-32b]="--hf-overrides")
declare -A SERVE_VALUE=([olmo3-32b]="$OLMO3_ROPE")

SERVER_PID=""

stop_server() {
    [[ -z "$SERVER_PID" ]] && return
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping vLLM server"
        # On SIGINT the API server shuts down its engine process too.
        kill -INT "$SERVER_PID" 2>/dev/null
        for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            pkill -KILL -P "$SERVER_PID" 2>/dev/null
            kill -KILL "$SERVER_PID" 2>/dev/null
        fi
        wait "$SERVER_PID" 2>/dev/null
    fi
    SERVER_PID=""
}
trap 'stop_server; exit 130' INT TERM
trap stop_server EXIT

start_server() {
    local model=$1 log=$2
    echo "Starting vLLM server for $model (log: $log)"
    # Default scheduler limits (1024 sequences in flight): on the B200 this keeps the KV
    # cache below half full; raising them caused preemptions and lowered throughput.
    local extra=()
    [[ -n "${SERVE_ARGS[$model]:-}" ]] && extra=("${SERVE_ARGS[$model]}" "${SERVE_VALUE[$model]}")
    vllm serve "${HF_ID[$model]}" \
        --revision "${REVISION[$model]}" \
        --served-model-name "$model" \
        --host 127.0.0.1 --port "$PORT" \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.92 \
        --seed "$SEED" "${extra[@]}" >>"$log" 2>&1 &
    SERVER_PID=$!
    # The first start of a model also downloads its weights.
    for _ in $(seq 720); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM server for $model exited during startup; see $log" >&2
            SERVER_PID=""
            return 1
        fi
        curl -sf "$API_BASE/models" >/dev/null && return 0
        sleep 5
    done
    echo "vLLM server for $model not ready after 1 hour" >&2
    return 1
}

# Every run of a model, one "<run name>|<output file>|<command>" per line.
model_runs() {
    local model=$1 corpus setting name folder entry lang run
    local prompted=(
        python memorization_experiment/generation/generate_vllm.py
        --model "$model" --api_base "$API_BASE"
        --do_sample --temperature "$TEMPERATURE" --top_p "$TOP_P"
        --num_generations "$NUM_GENERATIONS" --max_new_tokens "$MAX_NEW_TOKENS"
        --max_input_tokens 256 --batch_size 128 --seed "$SEED"
        # Requests queue behind the other runs' requests on the shared server.
        --request_timeout_s 14400
    )
    local free=(
        python memorization_experiment/generation/generate_vllm_free.py
        --model "$model" --tokenizer "${HF_ID[$model]}" --revision "${REVISION[$model]}"
        --num_generations "$FREE_GENERATIONS" --api_base "$API_BASE"
        --temperature "$TEMPERATURE" --top_p "$TOP_P" --max_new_tokens "$MAX_NEW_TOKENS"
        --seed "$SEED" --request_timeout_s 14400
    )
    for corpus in ${CORPORA[$model]}; do
        for setting in generic specific; do
            # Run named after the prompt set: generic_dw, specific_dw.
            name=${setting}_$corpus
            folder=$DATA_DIR/${DATASET_DIR[$corpus]}/$setting
            echo "$name|$folder/generations/$model/${setting}_generations.json|${prompted[*]} --input_jsonl $folder/${setting}_prompts.jsonl"
        done
    done
    for entry in ${PREFIXES[$model]}; do
        local corpus_dir=${entry%/*} stem=${entry#*/}
        echo "${stem}_prefix|$DATA_DIR/$corpus_dir/prefix/generations/$model/${stem}_prefix_generations.json|${prompted[*]} --input_jsonl $DATA_DIR/$corpus_dir/prefix/${stem}_prefix_prompts.jsonl"
    done
    echo "unconditional|$DATA_DIR/prompt_free/unconditional/generations/$model/unconditional_generations.json|${free[*]} --setting unconditional"
    for lang in ${CUE_LANGS[$model]}; do
        run=minimal_cue_$lang
        echo "$run|$DATA_DIR/prompt_free/$run/generations/$model/${run}_generations.json|${free[*]} --setting minimal_cue --lang $lang"
    done
}

run_model() {
    local model=$1 log_dir=$LOGS_DIR/$1 line rest output cmd
    local -a todo=()
    echo
    echo "=== $model: ${HF_ID[$model]} @ ${REVISION[$model]} ==="
    while IFS= read -r line; do
        rest=${line#*|}
        output=${rest%%|*}
        if [[ "$OVERWRITE" != 1 && -s "$output" ]]; then
            echo "  skip $output"
        else
            echo "  run  $output"
            todo+=("$line")
        fi
    done < <(model_runs "$model")
    [[ ${#todo[@]} -eq 0 ]] && return 0

    mkdir -p "$log_dir"
    start_server "$model" "$log_dir/vllm_server.log" || { stop_server; return 1; }

    # All runs at once: the server batches across them, so no run's tail leaves the GPU idle.
    local -a pids=() names=()
    local failed=0 i name
    for line in "${todo[@]}"; do
        name=${line%%|*}
        rest=${line#*|}
        output=${rest%%|*}
        cmd=${rest#*|}
        mkdir -p "$(dirname "$output")"
        $cmd --output_file "$output" >"$log_dir/$name.log" 2>&1 &
        pids+=($!)
        names+=("$name")
    done
    echo "  ${#pids[@]} runs in flight; logs in $log_dir/<run>.log"
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "  done   ${names[$i]}"
        else
            echo "  FAILED ${names[$i]} (see $log_dir/${names[$i]}.log)" >&2
            failed=1
        fi
    done
    stop_server
    return $failed
}

models=("$@")
[[ ${#models[@]} -eq 0 ]] && models=("${ALL_MODELS[@]}")
for model in "${models[@]}"; do
    [[ -n "${HF_ID[$model]:-}" ]] || { echo "Unknown model: $model (choose from ${ALL_MODELS[*]})" >&2; exit 2; }
done

status=0
for model in "${models[@]}"; do
    run_model "$model" || status=1
done
exit $status
