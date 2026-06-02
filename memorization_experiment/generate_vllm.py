"""
Prompt inference via a running vLLM OpenAI-compatible server.

Workflow for Apple Silicon:
    1) Install vLLM in base environment (Apple Silicon uses CPU build path).
    2) Start server:
        vllm serve <HF_MODEL_ID> --host 127.0.0.1 --port 8000
    3) Run this script to batch prompts through /v1/completions.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, TypeVar

from tqdm import tqdm

T = TypeVar("T")


@dataclass
class RunConfig:
    model: str
    api_base: str
    batch_size: int
    max_input_tokens: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    num_beams: int
    repetition_penalty: float
    seed: int
    input_json: str | None
    input_jsonl: str | None
    text_field: str
    domain_field: str
    default_domain: str
    output_file: str
    request_timeout_s: float


def validate_prompt_dict(data: Any, source_name: str) -> dict[str, list[str]]:
    if not isinstance(data, dict):
        raise ValueError(f"{source_name} must be a dict[str, list[str]].")

    cleaned: dict[str, list[str]] = {}
    for domain, prompts in data.items():
        if not isinstance(domain, str):
            raise ValueError(f"{source_name}: all keys must be strings. Found {type(domain)}.")
        if not isinstance(prompts, list):
            raise ValueError(f"{source_name}: value for domain '{domain}' must be a list.")
        normalized = [str(p) for p in prompts if str(p).strip()]
        if normalized:
            cleaned[domain] = normalized

    if not cleaned:
        raise ValueError(f"{source_name} contains no usable prompts.")
    return cleaned


def load_prompt_jsonl(
    path: str,
    *,
    text_field: str,
    domain_field: str,
    default_domain: str,
) -> dict[str, list[str]]:
    prompt_dict: dict[str, list[str]] = {}
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    for line_num, line in enumerate(raw, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_num} in {path}: {exc}") from exc

        if not isinstance(item, dict):
            raise ValueError(f"Line {line_num} in {path} must be a JSON object.")
        text = item.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        domain = item.get(domain_field)
        if not isinstance(domain, str) or not domain.strip():
            domain = default_domain
        prompt_dict.setdefault(domain, []).append(text)

    if not prompt_dict:
        raise ValueError(f"No usable prompts found in {path}.")
    return prompt_dict


def load_prompt_sets(
    *,
    input_json: str | None,
    input_jsonl: str | None,
    text_field: str,
    domain_field: str,
    default_domain: str,
) -> dict[str, dict[str, list[str]]]:
    if input_json:
        raw = json.loads(Path(input_json).read_text(encoding="utf-8"))
        return {"custom": validate_prompt_dict(raw, "custom_input_json")}

    if input_jsonl:
        prompt_dict = load_prompt_jsonl(
            input_jsonl,
            text_field=text_field,
            domain_field=domain_field,
            default_domain=default_domain,
        )
        set_name = Path(input_jsonl).stem
        return {set_name: prompt_dict}

    raise ValueError("Either --input_json or --input_jsonl must be provided.")


def batched(items: list[T], batch_size: int) -> Iterator[list[T]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _api_url(api_base: str, path: str) -> str:
    return f"{api_base.rstrip('/')}/{path.lstrip('/')}"


def _http_json_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
    timeout_s: float,
) -> dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not connect to {url}: {e}") from e


def check_server(api_base: str, api_key: str, timeout_s: float) -> None:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    _http_json_request(
        "GET",
        _api_url(api_base, "/models"),
        headers=headers,
        payload=None,
        timeout_s=timeout_s,
    )


def generate_batch_vllm_api(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompts: list[str],
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_beams: int,
    repetition_penalty: float,
    seed: int,
    request_timeout_s: float,
) -> list[dict[str, str]]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompts,
        "max_tokens": max_new_tokens,
        "truncate_prompt_tokens": max_input_tokens,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
        "n": 1,
    }

    if do_sample:
        payload["temperature"] = temperature
        payload["top_p"] = top_p
    else:
        payload["temperature"] = 0.0

    if num_beams > 1:
        payload["use_beam_search"] = True
        payload["best_of"] = num_beams

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = _http_json_request(
        "POST",
        _api_url(api_base, "/completions"),
        headers=headers,
        payload=payload,
        timeout_s=request_timeout_s,
    )

    choices = response.get("choices")
    if not isinstance(choices, list):
        raise RuntimeError(f"Unexpected completion response format: {response}")

    # Use None as sentinel so an empty-string completion is treated as valid.
    completions: list[str | None] = [None] * len(prompts)
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        idx = int(choice.get("index", 0))
        text = str(choice.get("text", ""))
        if 0 <= idx < len(completions) and completions[idx] is None:
            completions[idx] = text

    missing = [i for i, text in enumerate(completions) if text is None]
    if missing:
        raise RuntimeError(
            f"Missing completions for prompt indices: {missing}. "
            f"choices={choices}"
        )

    return [
        {
            "prompt": prompt,
            "completion": completion.strip(),
            "full_text": f"{prompt}{completion}",
        }
        for prompt, completion in zip(prompts, completions)
    ]


def run_prompt_set(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt_dict: dict[str, list[str]],
    set_name: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_beams: int,
    repetition_penalty: float,
    seed: int,
    request_timeout_s: float,
) -> dict[str, list[dict[str, str]]]:
    print(f"\n[*] Running prompt set: {set_name}")
    results_by_domain: dict[str, list[dict[str, str]]] = {}

    # Flatten across all domains for maximal batching throughput.
    flat_items: list[tuple[str, int, str]] = []
    for domain, prompts in prompt_dict.items():
        print(f"  - {domain}: {len(prompts)} prompts")
        results_by_domain[domain] = [{} for _ in prompts]
        for idx, prompt in enumerate(prompts):
            flat_items.append((domain, idx, prompt))

    total_batches = (len(flat_items) + batch_size - 1) // batch_size
    batch_iterator = tqdm(
        batched(flat_items, batch_size),
        total=total_batches,
        desc=f"{set_name}/all_domains",
        unit="batch",
        leave=False,
    )
    for item_batch in batch_iterator:
        prompt_batch = [item[2] for item in item_batch]
        generated = generate_batch_vllm_api(
            api_base=api_base,
            api_key=api_key,
            model=model,
            prompts=prompt_batch,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            seed=seed,
            request_timeout_s=request_timeout_s,
        )
        for (domain, idx, _), result in zip(item_batch, generated):
            results_by_domain[domain][idx] = result

    return {domain: list(items) for domain, items in results_by_domain.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference through a running vLLM OpenAI-compatible server."
    )
    parser.add_argument("--model", required=True, help="HF model id loaded by vLLM server.")
    parser.add_argument(
        "--api_base",
        default="http://127.0.0.1:8000/v1",
        help="vLLM OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api_key",
        default="EMPTY",
        help="Bearer token for API calls (vLLM commonly uses 'EMPTY').",
    )
    parser.add_argument(
        "--skip_server_check",
        action="store_true",
        help="Skip initial /v1/models connectivity check.",
    )
    parser.add_argument(
        "--input_json",
        default=None,
        help="Optional custom JSON dict[str, list[str]].",
    )
    parser.add_argument(
        "--input_jsonl",
        default=None,
        help="Path to JSONL prompts file (e.g. specific_prompts.jsonl).",
    )
    parser.add_argument(
        "--text_field",
        default="text",
        help="Field name containing prompt text in JSONL input.",
    )
    parser.add_argument(
        "--domain_field",
        default="domain",
        help="Field name containing domain/category in JSONL input.",
    )
    parser.add_argument(
        "--default_domain",
        default="default",
        help="Fallback domain name when JSONL rows omit domain.",
    )
    parser.add_argument("--output_file", default="generations_vllm.json", help="Output JSON path.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for API calls.")
    parser.add_argument("--max_input_tokens", type=int, default=256, help="Prompt token cap passed to API.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam count (1 = greedy/sample).")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--request_timeout_s",
        type=float,
        default=1000.0,
        help="HTTP timeout (seconds) per API request.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_server_check:
        print(f"[*] Checking vLLM server: {args.api_base}")
        check_server(args.api_base, args.api_key, args.request_timeout_s)

    if bool(args.input_json) == bool(args.input_jsonl):
        raise ValueError("Provide exactly one of --input_json or --input_jsonl")

    prompt_sets = load_prompt_sets(
        input_json=args.input_json,
        input_jsonl=args.input_jsonl,
        text_field=args.text_field,
        domain_field=args.domain_field,
        default_domain=args.default_domain,
    )
    run_cfg = RunConfig(
        model=args.model,
        api_base=args.api_base,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        input_json=args.input_json,
        input_jsonl=args.input_jsonl,
        text_field=args.text_field,
        domain_field=args.domain_field,
        default_domain=args.default_domain,
        output_file=args.output_file,
        request_timeout_s=args.request_timeout_s,
    )

    started_at = time.time()
    payload: dict[str, Any] = {
        "config": asdict(run_cfg),
        "model_is_encoder_decoder": False,
        "results": {},
    }

    for set_name, prompt_dict in prompt_sets.items():
        payload["results"][set_name] = run_prompt_set(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            prompt_dict=prompt_dict,
            set_name=set_name,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            request_timeout_s=args.request_timeout_s,
        )

    payload["runtime_seconds"] = round(time.time() - started_at, 3)
    payload["generated_items"] = sum(
        len(items)
        for set_data in payload["results"].values()
        for items in set_data.values()
    )

    out_path = Path(args.output_file)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[*] API base: {args.api_base}")
    print(f"[*] Saved results to: {out_path}")
    print(f"[*] Total generated items: {payload['generated_items']}")
    print(f"[*] Runtime (s): {payload['runtime_seconds']}")


if __name__ == "__main__":
    main()
