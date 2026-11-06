# Run records

Every benchmark behind the numbers in this repo, newest first. Each folder has a report and the raw results.

| Date | By | Machine | Stack | What it measured | Record |
|---|---|---|---|---|---|
| 2026-11-05 | this fork | RTX 5090, 64 GB RAM, WSL2 | 2.11 / 12.8 vs 2.15 / 13.2 | The stack upgrade: prefill 47-72% faster, decode the same | [torch-2.15](torch-2.15/REPORT.md) |
| 2026-11-04 | this fork | RTX 5090, capped to 8-32 GB | PyTorch 2.11, CUDA 12.8 | VRAM scaling: the README's tables. Decode answers were capped at 700 tokens | [vram-scaling](vram-scaling/REPORT.md) |
| 2026-11-03 | this fork | RTX 5090, 64 GB RAM, WSL2 | PyTorch 2.11, CUDA 12.8 | 3.05 vs 4.05 bpw on the full card. Decode answers were capped at 700 tokens | [rtx5090-wsl](rtx5090-wsl/REPORT.md) |
| 2026-10-24 | MaxDevv | RTX 4090 held to 16 GB VRAM, 8 GiB RAM | PyTorch 2.11, CUDA 12.8 | The original README's numbers | [4090-final](4090-final/REPORT.md) |
| 2026-10-24 | MaxDevv | RTX 4090, 8 GiB RAM | not recorded | Long context, up to 262K tokens | [4090-long](4090-long/RESUMED-REPORT.md) |
| 2026-10-24 | MaxDevv | RTX 3080 Laptop (16 GB) | not recorded | 8 vs 16 GiB expert RAM cache | [local-expert-ram-compare](local-expert-ram-compare/REPORT.md) |

## Adding a run

1. **Run it** with [`qwenflash bench`](../docs/manager.md#benchmarks), or from the dashboard's Benchmark tab. The
   manager frees the GPU and puts the model back afterwards. Every answer runs to its natural end.

   ```
   qwenflash bench matrix --out research/NAME/results      the README's table: 8-32 GB cards on a 32 GB one
   qwenflash bench quants --out research/NAME/results      3.05 and 4.05 bpw on this card
   ```

   Each result records the machine and its stack (GPU, driver, CUDA, PyTorch, commit).
2. **Tables:** `qwenflash bench --table research/NAME/results` prints the report's tables, and `--sizing` prints the
   rows for `sizing.MEASURED`. Then `python sizing.py presets --write README.md` regenerates the README's tables.
3. **Record:** write the folder's `REPORT.md`, and add a row here with the date, machine and stack.
