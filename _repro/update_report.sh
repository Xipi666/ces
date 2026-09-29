#!/usr/bin/env bash
# Regenerate the comparison table in REPRODUCTION_REPORT.md from summary.tsv files
set -euo pipefail
cd "$(dirname "$0")/.."

/g/Anaconda3/envs/wrj/python.exe - << 'PYEOF'
import io, os

ORIG_MSE = {
    ('96','24'): 0.070, ('96','96'): 0.129, ('336','192'): 0.173,
    ('720','336'): 0.193, ('720','720'): 0.270,
    ('168','24'): 0.031, ('168','96'): 0.061, ('168','192'): 0.080, ('168','336'): 0.103,
}
ORIG_WTR = {
    ('96','24'): 69.640, ('96','96'): 57.146, ('336','192'): 49.314,
    ('720','336'): 44.424, ('720','720'): 38.842,
}

rows = []
for seed in (2024, 2025, 2026):
    tsv = f'_repro/logs/etth2_matrix_seed{seed}/summary.tsv'
    if not os.path.exists(tsv):
        continue
    with io.open(tsv, encoding='utf-8', errors='replace') as fh:
        header = fh.readline().rstrip('\n').split('\t')
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 15:
                continue
            d = dict(zip(header, parts))
            sl, pl = d['seq_len'], d['pred_len']
            om = ORIG_MSE.get((sl, pl), '-')
            ow = ORIG_WTR.get((sl, pl), '-')
            tm = d.get('test_mse', '?')
            tw = d.get('weighted_wtr', '?')
            rows.append(f"| {d['run_id']} | {sl} | {pl} | {d['master_seed']} | {om} | {ow} | {tm} | {tw} | {d['status']} |")

if not rows:
    rows = ['| （尚无完成的组） | | | | | | | | |']

report = '_repro/REPRODUCTION_REPORT.md'
text = io.open(report, encoding='utf-8').read()
marker = '<!-- REPRO_ROWS -->'
idx = text.index(marker)
text = text[:idx + len(marker)] + '\n' + '\n'.join(rows) + text[idx + len(marker):]
io.open(report, 'w', encoding='utf-8').write(text)
print(f'report table updated: {len(rows)} rows')
PYEOF
