"""Markdown table from vram_matrix.py results: python bench/vram_table.py research/vram-scaling/results"""
import json
import sys
from pathlib import Path

CARD = [(26, "32 GB"), (19, "24 GB"), (12.5, "16 GB"), (8.5, "12 GB"), (0, "8 GB")]


def card_class(usable):
    return next(name for floor, name in CARD if usable >= floor)


def fmt(v, spec="{:.1f}", none="-"):
    return none if v is None else spec.format(v)


def main(d):
    rows = [json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))]
    rows.sort(key=lambda r: (r["quant"], -r["usable_gb"], r["context"]))
    print("| Card | Quant | VRAM for the model | Context | Prefill chunk | Cached experts | Expert picks VRAM / RAM / NVMe "
          "| Decode, median (min) | Prefill | First token | Startup |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        e, plan, long = r.get("engine") or {}, r.get("plan") or {}, r.get("long") or {}
        chunk = "1,536, no staging" if r.get("nostage") or plan.get("staging") is False else f"{plan.get('chunk') or 0:,}"
        head = (f"| {card_class(r['usable_gb'])} | {r['quant']} | {r['usable_gb']:g} GB | {r['context'] // 1024}K | "
                f"{chunk if plan.get('ok') or r.get('nostage') else '-'} |")
        if not r.get("decode_tps_median"):
            why = str(r.get("error") or "no result").splitlines()[0][:120]
            print(f"{head} - | - | **does not run**: {why} | - | - | {fmt(r.get('startup_s'), '{:.0f} s')} |")
            continue
        picks = (f"{e['picks_vram_pct']:.0f} / {e['picks_ram_pct']:.0f} / {e['picks_nvme_pct']:.0f} %"
                 if e.get("picks_vram_pct") is not None else "-")
        prefill = (f"{long['prefill_tps']:,.0f} tok/s ({long['prompt_tokens'] // 1000}K prompt"
                   f"{'' if long.get('needle_found') else ', needle missed'})" if long.get("prefill_tps") else "-")
        print(f"{head} {e.get('slots', '-'):,} | {picks} | **{fmt(r['decode_tps_median'])}** "
              f"({fmt(r.get('decode_tps_min'))}) tok/s | {prefill} | {fmt(r.get('short_ttft_median_s'), '{:.2f} s')} | "
              f"{fmt(r.get('startup_s'), '{:.0f} s')} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "research/vram-scaling/results")
