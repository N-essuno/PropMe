#!/usr/bin/env python3
"""Sample Wikipedia articles created after a date (the content date of rule (a)).

MediaWiki assigns page ids at creation, in increasing order (moves keep the
id, restored pages get their old one back). So:

1. The threshold id for a date is the median page id of the first 50 page
   creations on or after it (creation log, any namespace; one API request).
   Every page with a higher id was created after the date. The median is
   robust to log entries whose page was later deleted and recreated under a
   new id, and is only minutes later than the first creation.
2. The pages-articles-multistream dump is ordered by page id, so only its
   parts covering ids above the lowest threshold are downloaded (the whole
   dump for small wikis such as dawiki).
3. Main-namespace pages with an id above the threshold that are not
   redirects or disambiguation pages are kept. Wikitext becomes plain text
   with mwparserfromhell (templates, tags and file links are dropped).
4. --num-docs of them are drawn at random (seeded).

The API is used only for the thresholds: Wikimedia rate-limits anonymous API
clients hard, so the article text comes from the dumps.

Dates per corpus are in sources.WIKI_CREATED_AFTER. Output:
RAW_DIR/wikipedia/<corpus>_docs.jsonl with id (page id), title, text, url,
lang, created_after, id_threshold, dump.

Run from the repository root:

    python 00_prepare_data/propensity_settings/wikipedia_new.py --corpora dw cp d3
"""

from __future__ import annotations

import argparse
import bz2
import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mwparserfromhell

import sources as src

USER_AGENT = "PropMe-unseen-prompts/1.0 (research script; python-urllib)"
DUMPS = {"da": ("dawiki", "20261001"), "en": ("enwiki", "20260901")}
DEFAULT_NUM_DOCS = {"dw": 400, "cp": 1000, "d3": 2500}
WIKI_DIR = src.RAW_DIR / "wikipedia"
DISAMBIGUATION = re.compile(r"\{\{\s*(disambig|dab\b|hndis|set index|flertydig|geodis)", re.I)
# Leftovers of image/category links and table syntax that strip_code keeps.
NOISE_LINE = re.compile(r"^\s*(thumb\||\||!|\{\||\|\}|Kategori:|Category:|Fil:|File:|Billede:|Image:)")


def get(url: str, attempts: int = 8) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except Exception as error:
            retry_after = getattr(error, "headers", None) and error.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else min(120, 5 * 2 ** attempt)
            print(f"[retry {attempt + 1}] {url}: {error}; sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Request failed: {url}")


def threshold_id(lang: str, date: str) -> int:
    params = {"action": "query", "list": "logevents", "letype": "create", "lestart": f"{date}T00:00:00Z",
              "ledir": "newer", "lelimit": "50", "leprop": "ids|timestamp", "format": "json", "formatversion": "2"}
    data = json.loads(get(f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)))
    ids = sorted(e["pageid"] for e in data["query"]["logevents"] if e.get("pageid"))
    if not ids:
        raise RuntimeError(f"No page creations on {lang}.wikipedia after {date}")
    return ids[len(ids) // 2]


def dump_parts(lang: str, min_id: int) -> list[tuple[str, int]]:
    """(file name, size) of the multistream dump parts holding page ids >= min_id."""
    wiki, date = DUMPS[lang]
    status = json.loads(get(f"https://dumps.wikimedia.org/{wiki}/{date}/dumpstatus.json"))["jobs"]
    files = {**status["articlesmultistreamdump"]["files"], **status.get("articlesmultistreamdumprecombine", {}).get("files", {})}
    part_range = re.compile(r"\.xml-p\d+p(\d+)\.bz2$")
    parts = [(name, v["size"]) for name, v in files.items() if part_range.search(name)]
    if parts:
        def last_id(name: str) -> int:
            return int(part_range.search(name).group(1))
        return sorted((p for p in parts if last_id(p[0]) >= min_id), key=lambda p: last_id(p[0]))
    whole = f"{wiki}-{date}-pages-articles-multistream.xml.bz2"
    return [(whole, files[whole]["size"])]


def download_parts(lang: str, min_id: int) -> list[Path]:
    wiki, date = DUMPS[lang]
    out_dir = WIKI_DIR / "dumps"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, size in dump_parts(lang, min_id):
        path = out_dir / name
        if not path.exists() or path.stat().st_size != size:
            print(f"  downloading {name} ({size / 1e9:.2f} GB)", flush=True)
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(urllib.request.Request(
                    f"https://dumps.wikimedia.org/{wiki}/{date}/{name}", headers={"User-Agent": USER_AGENT}),
                    timeout=600) as response, open(tmp, "wb") as f:
                while chunk := response.read(1 << 22):
                    f.write(chunk)
            tmp.rename(path)
        paths.append(path)
    return paths


def plain_text(wikitext: str) -> str:
    text = mwparserfromhell.parse(wikitext).strip_code(normalize=True, collapse=True)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line and not NOISE_LINE.match(line))


def parse_part(args: tuple[Path, int]) -> list[dict]:
    """Main-namespace articles with page id >= min_id in one dump part."""
    path, min_id = args
    articles = []
    with bz2.open(path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if not elem.tag.endswith("}page"):
                continue
            ns = elem.findtext("{*}ns")
            page_id = int(elem.findtext("{*}id"))
            if ns == "0" and page_id >= min_id and elem.find("{*}redirect") is None:
                wikitext = elem.findtext("{*}revision/{*}text") or ""
                if not DISAMBIGUATION.search(wikitext):
                    text = plain_text(wikitext)
                    if text:
                        articles.append({"id": page_id, "title": elem.findtext("{*}title"), "text": text})
            elem.clear()
    return articles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="+", choices=list(src.WIKI_CREATED_AFTER), default=list(src.WIKI_CREATED_AFTER))
    parser.add_argument("--num-docs", type=int, default=None,
                        help="Articles per corpus (default: dw 400, cp 1000, d3 2500).")
    parser.add_argument("--workers", type=int, default=8, help="Dump parts parsed in parallel.")
    args = parser.parse_args()
    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    thresholds_path = WIKI_DIR / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {}
    for corpus in args.corpora:
        lang, date = src.WIKI_CREATED_AFTER[corpus]
        key = f"{lang}:{date}"
        if key not in thresholds:
            thresholds[key] = threshold_id(lang, date)
            thresholds_path.write_text(json.dumps(thresholds, indent=2))
        print(f"{corpus}: {lang}.wikipedia pages created after {date} have id >= {thresholds[key]:,}", flush=True)

    for lang in sorted({src.WIKI_CREATED_AFTER[c][0] for c in args.corpora}):
        corpora = [c for c in args.corpora if src.WIKI_CREATED_AFTER[c][0] == lang]
        min_id = min(thresholds[":".join(src.WIKI_CREATED_AFTER[c])] for c in corpora)
        parts = download_parts(lang, min_id)
        with ProcessPoolExecutor(min(args.workers, len(parts))) as pool:
            articles = [a for found in pool.map(parse_part, [(p, min_id) for p in parts]) for a in found]
        print(f"{lang}: {len(articles):,} articles with id >= {min_id:,} in {len(parts)} dump parts", flush=True)
        for corpus in corpora:
            date = src.WIKI_CREATED_AFTER[corpus][1]
            threshold = thresholds[f"{lang}:{date}"]
            eligible = [a for a in articles if a["id"] >= threshold]
            num_docs = args.num_docs or DEFAULT_NUM_DOCS[corpus]
            picked = random.Random(f"{src.SEED}-wikipedia-{corpus}").sample(eligible, min(num_docs, len(eligible)))
            out_path = WIKI_DIR / f"{corpus}_docs.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for a in picked:
                    f.write(json.dumps({
                        **a, "url": f"https://{lang}.wikipedia.org/?curid={a['id']}", "lang": lang,
                        "created_after": date, "id_threshold": threshold, "dump": "-".join(DUMPS[lang]),
                    }, ensure_ascii=False) + "\n")
            print(f"{corpus}: {len(picked)} of {len(eligible):,} eligible articles -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
