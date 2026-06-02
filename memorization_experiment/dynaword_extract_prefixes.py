from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL_ID = "meta-llama/Llama-2-7b-hf"
PREFIX_TOKENS = 50
DEFAULT_DOMAIN = "unknown"

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "memorization_experiment" / "data" / "dynaword" / "dynaword_sample_docs.jsonl"
OUTPUT_PATH = REPO_ROOT / "memorization_experiment" / "data" / "dynaword" / "prefix" / "dynaword_prefix_prompts.jsonl"


def extract_prefixes() -> None:
	tokenizer = AutoTokenizer.from_pretrained(
		MODEL_ID,
		add_bos_token=False,
		add_eos_token=False,
	)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

	with INPUT_PATH.open("r", encoding="utf-8") as in_f, OUTPUT_PATH.open(
		"w", encoding="utf-8"
	) as out_f:
		for line_num, line in enumerate(in_f, start=1):
			if not line.strip():
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(
					f"Invalid JSON on line {line_num} in {INPUT_PATH}: {exc}"
				) from exc

			text = record.get("text", "")
			if not isinstance(text, str) or not text.strip():
				continue

			token_ids = tokenizer.encode(text, add_special_tokens=False)
			prefix_ids = token_ids[:PREFIX_TOKENS]
			prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)

			metadata = record.get("metadata", {})
			domain = DEFAULT_DOMAIN
			if isinstance(metadata, dict):
				source = metadata.get("source")
				if isinstance(source, str) and source.strip():
					domain = source

			out_f.write(
				json.dumps({"text": prefix_text, "domain": domain}, ensure_ascii=False)
				+ "\n"
			)


if __name__ == "__main__":
	extract_prefixes()
