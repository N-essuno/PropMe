#!/usr/bin/env python3
"""Generate completions for the generic and specific prompt sets with vLLM.

For each model this starts a vLLM OpenAI-compatible server, runs
memorization_experiment/generation/generate_vllm.py on every prompt set matched to the
model's training corpus, then stops the server:

    olmo3-32b                          allenai/Olmo-3-1125-32B              d3 (Dolma3)
    dfm-stage1, dfm-stage2, dfm-main   dfm-decoder-open-v0-7b-pt @ revision dw (Dynaword), cp (Common Pile)
    comma-2t                           common-pile/comma-v0.1-2t            cp (Common Pile)

Prompt sets per corpus x (in memorization_experiment/data): generic_x (Tatoeba,
build_prompt_sets.py) and specific_x (unseen-source sentences,
build_unseen_prompt_sets.py).

Two prompt-free settings (memorization_experiment/generation/generate_vllm_free.py) need
the number of generations given with --free-generations:

    unconditional       start-of-document token only; one run per model
    minimal_cue_<lang>  start-of-document token + one common word, sampled per
                        generation from wordfreq; one run per language of the
                        model's corpora (cp, d3 -> en, dw -> da)

All models sample with temperature 0.7 and top_p 0.95 (see MODELS). Run from the repository root;
the server is started with the `vllm` executable on PATH (--vllm-bin to change):

    python 00_prepare_data/propensity_settings/run_generations.py --dry-run --free-generations 1000
    python 00_prepare_data/propensity_settings/run_generations.py --models dfm-main --num-generations 5 --free-generations 1000
    python 00_prepare_data/propensity_settings/run_generations.py --settings prompted --num-generations 5

Writes each model's generations next to the prompts (run_tracing.generations_path:
memorization_experiment/data/<dataset>/<setting>/generations/<model>/, or
prompt_free/<run>/generations/<model>/ for the prompt-free runs), one record per
generation, readable by simple_trace.py --is-generation-json, plus run_config.json
and vllm_server.log per model in logs/generation/<model>. Runs whose output exists are
skipped unless --overwrite.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import run_tracing as rt


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parents[1]
GENERATE_VLLM_PATH = REPO_ROOT / "memorization_experiment" / "generation" / "generate_vllm.py"
GENERATE_FREE_PATH = REPO_ROOT / "memorization_experiment" / "generation" / "generate_vllm_free.py"
DEFAULT_VLLM_BIN = Path(shutil.which("vllm") or "vllm")

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
PROMPT_SET_KINDS = ("generic_{x}", "specific_{x}")
SETTINGS = ("prompted", "unconditional", "minimal_cue")
FREE_SETTINGS = ("unconditional", "minimal_cue")
# wordfreq language of each corpus, for the minimal-cue words.
CORPUS_LANG = {"cp": "en", "d3": "en", "dw": "da"}


@dataclass(frozen=True)
class Model:
    hf_id: str
    revision: str
    corpora: tuple[str, ...]  # prompt-set suffixes: cp, d3, dw
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    sampling_source: str = "default: no recommendation in the model card or generation_config.json"


DFM = "danish-foundation-models/dfm-decoder-open-v0-7b-pt"
MODELS = {
    "olmo3-32b": Model(
        "allenai/Olmo-3-1125-32B",
        "main",
        ("d3",),
        sampling_source=(
            "default, so all models sample alike (the model card's usage example uses "
            "temperature=1.0, top_p=0.7)"
        ),
    ),
    "dfm-stage1": Model(DFM, "stage1", ("dw", "cp")),
    "dfm-stage2": Model(DFM, "stage2", ("dw", "cp")),
    "dfm-main": Model(DFM, "main", ("dw", "cp")),
    "comma-2t": Model("common-pile/comma-v0.1-2t", "main", ("cp",)),
}


def prompt_sets_for(model: Model, kinds: list[str]) -> list[str]:
    return [kind.format(x=corpus) for corpus in model.corpora for kind in kinds]


def cue_languages(model: Model) -> list[str]:
    return list(dict.fromkeys(CORPUS_LANG[corpus] for corpus in model.corpora))


def output_path(model_name: str, run_name: str) -> Path:
    return rt.generations_path(run_name, model_name)


def model_jobs(model_name: str, model: Model, args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """(run name, generation command) for every run of a model in the selected settings."""
    jobs = []
    if "prompted" in args.settings:
        jobs += [(p, generate_command(model_name, model, p, args)) for p in prompt_sets_for(model, args.set_kinds)]
    if "unconditional" in args.settings:
        jobs.append(("unconditional", free_command(model_name, model, "unconditional", None, args)))
    if "minimal_cue" in args.settings:
        jobs += [
            (f"minimal_cue_{lang}", free_command(model_name, model, "minimal_cue", lang, args))
            for lang in cue_languages(model)
        ]
    return jobs


def server_command(model_name: str, model: Model, args: argparse.Namespace) -> list[str]:
    return [
        str(args.vllm_bin), "serve", model.hf_id,
        "--revision", model.revision,
        "--served-model-name", model_name,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--seed", str(args.seed),
        *shlex.split(args.vllm_args),
    ]


def generate_command(model_name: str, model: Model, prompt_set: str, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(GENERATE_VLLM_PATH),
        "--model", model_name,
        "--api_base", api_base(args),
        "--input_jsonl", str(rt.prompt_set_path(prompt_set)),
        "--output_file", str(output_path(model_name, prompt_set)),
        "--do_sample",
        "--temperature", str(model.temperature),
        "--top_p", str(model.top_p),
        "--num_generations", str(args.num_generations),
        "--max_new_tokens", str(args.max_new_tokens),
        "--max_input_tokens", str(args.max_input_tokens),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
    ]


def free_command(
    model_name: str, model: Model, setting: str, lang: str | None, args: argparse.Namespace
) -> list[str]:
    run_name = setting if lang is None else f"{setting}_{lang}"
    return [
        sys.executable, str(GENERATE_FREE_PATH),
        "--model", model_name,
        "--tokenizer", model.hf_id,
        "--revision", model.revision,
        "--setting", setting,
        *(["--lang", lang] if lang else []),
        "--num_generations", str(args.free_generations),
        "--cue_top_n", str(args.cue_top_n),
        "--api_base", api_base(args),
        "--output_file", str(output_path(model_name, run_name)),
        "--temperature", str(model.temperature),
        "--top_p", str(model.top_p),
        "--max_new_tokens", str(args.max_new_tokens),
        "--seed", str(args.seed),
    ]


def api_base(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}/v1"


def wait_for_server(server: subprocess.Popen | None, args: argparse.Namespace) -> None:
    """Poll /v1/models until the server answers; fail if it exits or times out."""
    deadline = time.time() + args.server_timeout
    while time.time() < deadline:
        if server is not None and server.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {server.returncode}; see its log")
        try:
            with urllib.request.urlopen(f"{api_base(args)}/models", timeout=5):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(5)
    raise TimeoutError(f"vLLM server not ready after {args.server_timeout}s")


def stop_server(server: subprocess.Popen) -> None:
    """Stop the server and its worker processes (it runs in its own process group)."""
    if server.poll() is not None:
        return
    pgid = server.pid
    for sig, wait_s in ((signal.SIGINT, 60), (signal.SIGTERM, 30), (signal.SIGKILL, 30)):
        try:
            os.killpg(pgid, sig)
            server.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            return


def run_model(model_name: str, model: Model, args: argparse.Namespace) -> None:
    jobs = model_jobs(model_name, model, args)
    todo = [(name, cmd) for name, cmd in jobs if args.overwrite or not output_path(model_name, name).exists()]
    print(f"\n=== {model_name}: {model.hf_id}@{model.revision} ===")
    print(f"Sampling: temperature={model.temperature}, top_p={model.top_p} ({model.sampling_source})")
    for name, _ in jobs:
        print(f"  {'run ' if any(name == t for t, _ in todo) else 'skip'} {name}")
    if not todo:
        return

    serve_cmd = server_command(model_name, model, args)
    if args.dry_run:
        if not args.no_serve:
            print("$ " + shlex.join(serve_cmd))
        for _, cmd in todo:
            print("$ " + shlex.join(cmd))
        return

    out_dir = rt.GENERATION_LOGS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": asdict(model),
        "served_model_name": model_name,
        "num_generations": args.num_generations,
        "free_generations": args.free_generations,
        "cue_top_n": args.cue_top_n,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "seed": args.seed,
        "vllm_command": None if args.no_serve else shlex.join(serve_cmd),
        "runs": [name for name, _ in todo],
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=4), encoding="utf-8")

    server = None
    if not args.no_serve:
        log = open(out_dir / "vllm_server.log", "a", encoding="utf-8")
        print(f"Starting vLLM server (log: {out_dir / 'vllm_server.log'})", flush=True)
        server = subprocess.Popen(serve_cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_for_server(server, args)
        for run_name, cmd in todo:
            print(f"\n--- {model_name} / {run_name} ---", flush=True)
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    finally:
        if server is not None:
            print(f"Stopping vLLM server for {model_name}", flush=True)
            stop_server(server)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument(
        "--set-kinds",
        nargs="+",
        choices=PROMPT_SET_KINDS,
        default=list(PROMPT_SET_KINDS),
        help="Prompt-set kinds to generate for each model corpus ({x} = cp, d3 or dw).",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=SETTINGS,
        default=list(SETTINGS),
        help="prompted: the prompt sets; unconditional and minimal_cue: prompt-free generation.",
    )
    parser.add_argument("--num-generations", type=int, default=1, help="Completions per prompt (prompted setting).")
    parser.add_argument(
        "--free-generations",
        type=int,
        default=None,
        help="Generations per unconditional / minimal_cue run; required when those settings are selected.",
    )
    parser.add_argument("--cue-top-n", type=int, default=100, help="Minimal cues are drawn from this many most common words.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64, help="Prompts per API request.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-bin", type=Path, default=DEFAULT_VLLM_BIN)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="GPUs per server. Olmo-3 32B in bf16 needs ~65 GB of weights: 1 x 80 GB GPU or 2 smaller ones.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Context the server reserves KV cache for; prompts plus completions stay far below this.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-args", default="", help="Extra arguments for `vllm serve`, as one string.")
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=3600,
        help="Seconds to wait for the server; the first start also downloads the weights.",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Use a server already running at --host/--port (serving one model, under its model name here).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running anything.")
    args = parser.parse_args()

    if args.num_generations < 1:
        parser.error("--num-generations must be >= 1")
    free = [s for s in args.settings if s in FREE_SETTINGS]
    if free and args.free_generations is None:
        parser.error(
            f"--free-generations is required for {', '.join(free)} "
            "(or leave them out with --settings prompted)"
        )
    if args.free_generations is not None and args.free_generations < 1:
        parser.error("--free-generations must be >= 1")
    if args.no_serve and len(args.models) > 1:
        parser.error("--no-serve uses one running server, so pass a single model with --models")
    if not args.dry_run and not args.no_serve and not args.vllm_bin.exists():
        parser.error(f"vLLM not found at {args.vllm_bin}; pass --vllm-bin")
    missing = [
        p for name in args.models for p in prompt_sets_for(MODELS[name], args.set_kinds)
        if "prompted" in args.settings and not rt.prompt_set_path(p).exists()
    ]
    if missing:
        parser.error(
            "Missing prompt sets (run build_prompt_sets.py / build_unseen_prompt_sets.py): "
            + ", ".join(sorted(set(missing)))
        )

    for name in args.models:
        run_model(name, MODELS[name], args)


if __name__ == "__main__":
    main()
