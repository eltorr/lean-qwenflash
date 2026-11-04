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
    rows = []
    for p in sorted(Path(d).glob("*.json")):
        r = json.loads(p.read_text())
        rows.append(r)
    rows.sort(key=lambda r: (r["quant"], -r["usable_gb_target"], r["context"]))
    print("| Card class | Quant | VRAM for the model | Context | Prefill chunk | Cached experts | Expert picks from VRAM / RAM / NVMe "
          "| Decode, median (min) | Prefill (prompt) | First token, short | Startup |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        e, plan, long = r.get("engine") or {}, r.get("plan") or {}, r.get("long") or {}
        if r.get("error") and not r.get("decode_tps_median"):
            print(f"| {card_class(r['usable_gb_target'])} | {r['quant']} | {r['free_at_start_gb']} GB | "
                  f"{r['context'] // 1024}K | {plan.get('chunk') or ('1536 (no staging)' if r.get('nostage') else '-')} | - | - | "
                  f"**does not run**: {str(r['error']).splitlines()[0][:110]} | - | - | {fmt(r.get('startup_s'), '{:.0f} s')} |")
            continue
        picks = (f"{e.get('picks_vram_pct')} / {e.get('picks_ram_pct')} / {e.get('picks_nvme_pct')} %"
                 if e.get("picks_vram_pct") is not None else "-")
        chunk = "1536 (no staging)" if r.get("nostage") else plan.get("chunk")
        print(f"| {card_class(r['usable_gb_target'])} | {r['quant']} | {r['free_at_start_gb']} GB "
              f"(used {fmt(r.get('model_vram_after_gb'))}) | {r['context'] // 1024}K | {chunk} | {e.get('slots')} | {picks} | "
              f"**{fmt(r.get('decode_tps_median'))}** ({fmt(r.get('decode_tps_min'))}) tok/s | "
              f"{fmt(long.get('prefill_tps'), '{:,.0f}')} tok/s ({long.get('prompt_tokens', '-'):,} tok"
              f"{', needle ok' if long.get('needle_found') else ', NEEDLE MISSED' if long else ''}) | "
              f"{fmt(r.get('short_ttft_median_s'), '{:.2f} s')} | {fmt(r.get('startup_s'), '{:.0f} s')} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "research/vram-scaling/results")
