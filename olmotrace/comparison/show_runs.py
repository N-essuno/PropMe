"""Print one line per finished run from outputs/e2e/runs.jsonl."""
import json, os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "e2e", "runs.jsonl")
for l in open(p):
    r = json.loads(l)
    if r["tag"].startswith("pilot"):
        continue
    print(f"{r['tag']:42s} {r['tool']:9s} wall {r['wall_s']:7.1f}  trace {r['trace_elapsed_s']}  cpu {r['cpu_s']:8.1f}  "
          f"anon {r['peak_rss_anon_gb']:.2f}GB  pte {r['peak_page_tables_gb']:.2f}GB  read {r['read_gb']:.2f}GB")
