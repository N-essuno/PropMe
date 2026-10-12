"""Models, corpora, settings and file layout of the PropMe generation runs.

The generations come from run_generations_all.sh. This module says where each
one lives and where its SimpleTrace outputs, propensity reports and plots go, so
that the presets of run_memorization_experiments.py, compute_propensity_metrics.py,
plot_memorization_results.py and plot_comparison_overviews.py stay in step.

A run is one model traced against one of its training corpora's index. Each run
has five settings: the generic (Tatoeba) and specific (unseen-source sentences,
00_prepare_data/propensity_settings) prompt sets, the two prompt-free settings (unconditional,
minimal_cue) and the prefix capability setting. Minimal cues are in the corpus language, so a
DFM model's minimal_cue is minimal_cue_da on Dynaword and minimal_cue_en on
Common Pile; its unconditional generations are traced against both.

Prompt sets with more than SAMPLE_SIZE prompts are traced on a random sample of
SAMPLE_SIZE of them (seed 42, from subsample_prompt_sets.py): their generations are
read from <stem>_generations_<SAMPLE_SIZE>.json and their traces written to
st_<setting>_<SAMPLE_SIZE>_{results,summary}.json, next to any full-set traces.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Corpus:
    key: str  # prompt-set suffix: generic_<key>, specific_<key>, ...
    name: str  # index key in run_memorization_experiments.py, and the tag/group name
    label: str
    lang: str  # wordfreq language of the minimal cues
    data_dir: str  # memorization_experiment/data/<data_dir>: the corpus's prompts, by setting
    prefix_stem: str  # <prefix_stem>_prefix_prompts.jsonl


@dataclass(frozen=True)
class Model:
    key: str  # served model name, and the folder of its generations
    label: str
    family: str
    corpora: tuple[str, ...]


DYNAWORD = Corpus("dw", "dynaword", "Dynaword", "da", "dynaword2", "dynaword")
COMMONPILE = Corpus("cp", "commonpile", "Common Pile", "en", "commonpile", "commonpile")
DOLMA3 = Corpus("d3", "dolma3", "Dolma 3", "en", "dolma3", "dolma3")
CORPORA = {corpus.name: corpus for corpus in (DYNAWORD, COMMONPILE, DOLMA3)}

MODELS = (
    Model("dfm-stage1", "DFM Stage 1", "dfm", ("dynaword", "commonpile")),
    Model("dfm-stage2", "DFM Stage 2", "dfm", ("dynaword", "commonpile")),
    Model("dfm-main", "DFM", "dfm", ("dynaword", "commonpile")),
    Model("comma-2t", "Comma", "comma", ("commonpile",)),
    Model("olmo3-32b", "Olmo 3 32B", "olmo3", ("dolma3",)),
)
MODELS_BY_KEY = {model.key: model for model in MODELS}

SETTINGS = ("generic", "specific", "unconditional", "minimal_cue", "prefix")
NON_PREFIX_SETTINGS = SETTINGS[:-1]
SETTING_LABELS = {
    "generic": "Generic",
    "specific": "Specific",
    "unconditional": "Unconditional",
    "minimal_cue": "Minimal Cue",
    "prefix": "Prefix",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = "memorization_experiment/data"
# Prompt sets larger than this are traced on a sample of this many prompts (see
# subsample_prompt_sets.py); None traces every set in full.
SAMPLE_SIZE: int | None = 1000
# Prompt-free runs (unconditional, minimal_cue_<lang>) belong to no corpus.
PROMPT_FREE_DIR = f"{DATA_DIR}/prompt_free"
OUTPUT_DIR = "memorization_experiment/data/propme"


def runs() -> list[tuple[Model, Corpus]]:
    return [(model, CORPORA[name]) for model in MODELS for name in model.corpora]


def run_name(model: Model, corpus: Corpus) -> str:
    return f"{model.key}-{corpus.name}"


def setting_tag(setting: str) -> str:
    return setting.replace("_", "-")


@functools.cache
def _prompt_count(path: str) -> int:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    return sum(1 for line in text.split("\n") if line.strip())


def sample_suffix(corpus: Corpus, setting: str) -> str:
    """"_1000" for a prompted setting whose prompt set is larger than SAMPLE_SIZE, else ""."""
    if SAMPLE_SIZE is None or setting in ("unconditional", "minimal_cue"):
        return ""
    return f"_{SAMPLE_SIZE}" if _prompt_count(prompts_path(corpus, setting)) > SAMPLE_SIZE else ""


def generation_path(model: Model, corpus: Corpus, setting: str) -> str:
    """A model's generations for a setting, next to the setting's prompts:
    dynaword2/generic/generations/dfm-main/generic_generations_1000.json (sampled, see
    sample_suffix), or for the prompt-free settings
    prompt_free/minimal_cue_da/generations/dfm-main/minimal_cue_da_generations.json."""
    if setting in ("unconditional", "minimal_cue"):
        run = "unconditional" if setting == "unconditional" else f"minimal_cue_{corpus.lang}"
        return f"{PROMPT_FREE_DIR}/{run}/generations/{model.key}/{run}_generations.json"
    folder, name = prompts_path(corpus, setting).rsplit("/", 1)
    stem = name.removesuffix("_prompts.jsonl")
    return f"{folder}/generations/{model.key}/{stem}_generations{sample_suffix(corpus, setting)}.json"


def prompts_path(corpus: Corpus, setting: str) -> str:
    """Prompts of a prompted setting, e.g. dynaword2/generic/generic_prompts.jsonl."""
    stem = f"{corpus.prefix_stem}_prefix" if setting == "prefix" else setting
    return f"{DATA_DIR}/{corpus.data_dir}/{setting}/{stem}_prompts.jsonl"


def run_dir(model: Model, corpus: Corpus) -> str:
    return f"{OUTPUT_DIR}/{model.key}/{corpus.name}"


def results_path(model: Model, corpus: Corpus, setting: str) -> str:
    return f"{run_dir(model, corpus)}/st_{setting}{sample_suffix(corpus, setting)}_results.json"


def summary_path(model: Model, corpus: Corpus, setting: str) -> str:
    return f"{run_dir(model, corpus)}/st_{setting}{sample_suffix(corpus, setting)}_summary.json"


def propensity_path(model: Model, corpus: Corpus) -> str:
    return f"{run_dir(model, corpus)}/propensity/propensity_metrics.json"


def run_plots_dir(model: Model, corpus: Corpus) -> str:
    return f"{run_dir(model, corpus)}/plots"


@dataclass(frozen=True)
class Series:
    label: str
    model: Model
    corpus: Corpus


@dataclass(frozen=True)
class Comparison:
    """Runs compared setting by setting.

    With a reference, each series also gets a propensity report against the
    reference in place of the prefix setting (as the earlier stage and
    cross-model comparisons did).
    """

    name: str
    title: str
    series: tuple[Series, ...]
    reference: Series | None = None

    @property
    def all_series(self) -> tuple[Series, ...]:
        return self.series + ((self.reference,) if self.reference else ())


def _series(model_key: str, corpus: Corpus, label: str | None = None) -> Series:
    model = MODELS_BY_KEY[model_key]
    return Series(label or model.label, model, corpus)


COMPARISONS = (
    Comparison(
        "dfm-stages-dynaword",
        "DFM Stages on Dynaword",
        (_series("dfm-stage1", DYNAWORD), _series("dfm-stage2", DYNAWORD)),
        _series("dfm-main", DYNAWORD),
    ),
    Comparison(
        "dfm-stages-commonpile",
        "DFM Stages on Common Pile",
        (_series("dfm-stage1", COMMONPILE), _series("dfm-stage2", COMMONPILE)),
        _series("dfm-main", COMMONPILE),
    ),
    Comparison(
        "dfm-dynaword-vs-commonpile",
        "DFM: Dynaword vs Common Pile",
        (_series("dfm-main", DYNAWORD, "Dynaword"),),
        _series("dfm-main", COMMONPILE, "Common Pile"),
    ),
    Comparison(
        "commonpile-dfm-vs-comma",
        "Common Pile: DFM vs Comma",
        (_series("dfm-main", COMMONPILE),),
        _series("comma-2t", COMMONPILE),
    ),
    Comparison(
        "dfm-stages-dynaword-commonpile",
        "DFM Stages on Dynaword and Common Pile",
        tuple(
            _series(key, corpus, f"{corpus.label}{stage}")
            for key, stage in (("dfm-stage1", " Stage 1"), ("dfm-stage2", " Stage 2"), ("dfm-main", ""))
            for corpus in (COMMONPILE, DYNAWORD)
        ),
    ),
)


def comparison_dir(comparison: Comparison) -> str:
    return f"{OUTPUT_DIR}/comparisons/{comparison.name}"
