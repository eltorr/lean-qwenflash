"""Tables from vram_matrix.py results.

  python bench/vram_table.py research/vram-scaling/results            markdown table for the report
  python bench/vram_table.py --sizing research/vram-scaling/results   rows for sizing.MEASURED (README presets)
"""
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
        if not r.get("decode_tps_median") or r.get("error"):
            tail = r.get("log_tail") or ""
            why = ("ran out of VRAM" + (" on the long prompt" if r.get("decode_tps_median") else " on the first request")
                   if "VRAM LIMIT EXCEEDED" in tail else str(r.get("error") or "no result").splitlines()[0][:100])
            dec = f" (decode {fmt(r['decode_tps_median'])} tok/s before that)" if r.get("decode_tps_median") else ""
            print(f"{head} - | - | **fails**: {why}{dec} | - | - | {fmt(r.get('startup_s'), '{:.0f} s')} |")
            continue
        picks = (f"{e['picks_vram_pct']:.0f} / {e['picks_ram_pct']:.0f} / {e['picks_nvme_pct']:.0f} %"
                 if e.get("picks_vram_pct") is not None else "-")
        prefill = (f"{long['prefill_tps']:,.0f} tok/s ({long['prompt_tokens'] // 1000}K prompt"
                   f"{'' if long.get('needle_found') else ', needle missed'})" if long.get("prefill_tps") else "-")
        print(f"{head} {e.get('slots') or 0:,} | {picks} | **{fmt(r['decode_tps_median'])}** "
              f"({fmt(r.get('decode_tps_min'))}) tok/s | {prefill} | {fmt(r.get('short_ttft_median_s'), '{:.2f} s')} | "
              f"{fmt(r.get('startup_s'), '{:.0f} s')} |")


def sizing_rows(d):
    """One sizing.MEASURED entry per completed run (edit gpu/ram if the defaults don't describe the machine)."""
    for p in sorted(Path(d).glob("*.json")):
        r = json.loads(p.read_text())
        long = r.get("long") or {}
        if r.get("error") or not r.get("decode_tps_median") or not long.get("prefill_tps"):
            continue
        h, plan = r.get("host") or {}, r.get("plan") or {}
        uncapped = r.get("method") == "uncapped card"
        card = round(r["card_total_gb"] * 1e9 / 2**30) if uncapped else int(card_class(r["usable_gb"]).split()[0])
        display = h.get("display") if uncapped else card > 8 or r["usable_gb"] < 7.5
        ram = round(h["ram_total_gb"] * 1e9 / 2**30) if h.get("ram_total_gb") else None
        gpu = (h.get("gpu") or "?").replace("NVIDIA GeForce ", "") + ("" if uncapped else ", capped")
        extra = ("" if display is not False else "display=False, ") + \
                ("" if plan.get("staging", True) and not r.get("nostage") else "staging=False, ")
        print(f'    dict(card={card}, {extra}quant="{r["quant"]}", context={r["context"]}, '
              f'ram_cache_gib={r.get("ram_gib")}, decode={r["decode_tps_median"]:.1f}, '
              f'prefill={long["prefill_tps"]:.0f}, ram={ram}, '
              f'platform="{h.get("platform", "linux")}", gpu="{gpu}", source="{d}"),')


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--sizing"]
    (sizing_rows if "--sizing" in sys.argv else main)(args[0] if args else "research/vram-scaling/results")
