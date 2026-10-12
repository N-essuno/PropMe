"""Run one tool's CLI on one input file and record resource usage.

Both tools get the same input file, index, unigram table, --docs-per-span,
--num-workers and --find-threads. The process tree is polled every 0.25 s
for private memory (RssAnon), page tables (VmPTE) and bytes read from storage.

python olmotrace/comparison/run_bench.py --tool ot --tag dolma3_verbatim_s0 \
    --inputs olmotrace/comparison/inputs/dolma3_verbatim.jsonl \
    --index-dir $PROPME_DATA_ROOT/indexes/dolma3_index_link \
    --unigram-probs-path 02_unigram_probs/unigram_probs_dolma3_link.json
"""

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time

import psutil

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "olmotrace", "comparison", "outputs", "e2e")
LOG_DIR = os.path.join(REPO, "logs", "olmotrace_comparison", "e2e")
TOOLS = {
    "ot": ["olmotrace/olmo_trace.py"],
    "st_text": ["03_tracing/simple_trace.py", "--match-mode", "text"],
    "st_mixed": ["03_tracing/simple_trace.py", "--match-mode", "mixed"],
}
BUCKETS = "1-3,4-6,7-10,11-20,21-30,31-40,41-50,51-100,101-inf"


def _status_kb(pid: int, field: str) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(field):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _cgroup_unreclaimable_gb() -> float:
    """Unreclaimable memory of this container's cgroup: anon, page tables, unreclaimable slab, stacks."""
    try:
        with open("/sys/fs/cgroup/memory.stat") as f:
            stat = dict(line.split() for line in f)
        return sum(int(stat.get(k, 0)) for k in ("anon", "pagetables", "slab_unreclaimable", "kernel_stack", "percpu")) / 1e9
    except OSError:
        return 0.0


def _progress(log_text: str) -> str | None:
    m = re.findall(r"(\d+)/(\d+) \[", log_text)
    return f"{m[-1][0]}/{m[-1][1]}" if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", choices=TOOLS, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--index-dir", nargs="+", required=True)
    p.add_argument("--unigram-probs-path", required=True)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--find-threads", type=int, default=4)
    p.add_argument("--docs-per-span", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-unreclaimable-gb", type=float, default=70.0,
                   help="Kill the run when the container's anon + page-table + kernel memory exceeds this "
                        "(the cgroup limit is 96 GB; an OOM kills the whole job, not just the run).")
    a = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    stem = os.path.join(OUT, f"{a.tag}__{a.tool}")
    log_path = os.path.join(LOG_DIR, f"{a.tag}__{a.tool}.log")
    cmd = [sys.executable, *TOOLS[a.tool],
           "--dataset", a.inputs, "--is-jsonl", "--text-field", "text",
           "--index-dir", *a.index_dir, "--unigram-probs-path", a.unigram_probs_path,
           "--num-workers", str(a.num_workers), "--find-threads", str(a.find_threads),
           "--docs-per-span", str(a.docs_per_span), "--length-buckets", BUCKETS,
           "--results-output", f"{stem}.results.jsonl", "--summary-output", f"{stem}.summary.json"]
    if a.limit:
        cmd += ["--limit", str(a.limit)]

    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
        root = psutil.Process(proc.pid)
        peak_anon = peak_pte = peak_unrecl = 0.0
        aborted = False
        read_bytes: dict[int, int] = {}
        while proc.poll() is None:
            unrecl = _cgroup_unreclaimable_gb()
            peak_unrecl = max(peak_unrecl, unrecl)
            if unrecl > a.max_unreclaimable_gb:
                aborted = True
                try:
                    victims = root.children(recursive=True) + [root]
                except psutil.NoSuchProcess:
                    victims = []
                for pr in victims:
                    try:
                        pr.kill()
                    except psutil.Error:
                        pass
                proc.wait()
                break
            try:
                procs = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            anon = pte = 0
            for pr in procs:
                anon += _status_kb(pr.pid, "RssAnon:")
                pte += _status_kb(pr.pid, "VmPTE:")
                try:
                    read_bytes[pr.pid] = max(read_bytes.get(pr.pid, 0), pr.io_counters().read_bytes)
                except (psutil.Error, OSError):
                    pass
            peak_anon, peak_pte = max(peak_anon, anon), max(peak_pte, pte)
            time.sleep(0.2)
    wall = time.perf_counter() - t0
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)

    with open(log_path) as f:
        log_text = f.read()
    m = re.search(r"Elapsed Time: ([0-9.]+) seconds", log_text)
    rec = {
        "tag": a.tag, "tool": a.tool, "inputs": a.inputs, "index_dir": a.index_dir,
        "num_workers": a.num_workers, "find_threads": a.find_threads, "docs_per_span": a.docs_per_span,
        "returncode": proc.returncode,
        "aborted_memory_cap": aborted,
        "max_unreclaimable_gb": a.max_unreclaimable_gb,
        "peak_cgroup_unreclaimable_gb": peak_unrecl,
        "progress_at_end": _progress(log_text),
        "wall_s": wall,
        "trace_elapsed_s": float(m.group(1)) if m else None,
        "cpu_s": (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime),
        "user_s": ru1.ru_utime - ru0.ru_utime,
        "sys_s": ru1.ru_stime - ru0.ru_stime,
        "peak_rss_anon_gb": peak_anon / 1e6,
        "peak_page_tables_gb": peak_pte / 1e6,
        "read_gb": sum(read_bytes.values()) / 1e9,
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - wall)),
    }
    with open(os.path.join(OUT, "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    if proc.returncode != 0:
        print(log_text[-3000:], file=sys.stderr)
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
