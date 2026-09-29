#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'artifacts' / 'experiments'
DOC = ROOT / 'docs' / 'RESULTS.md'

RUNS = [
    ('bart_baseline_500', 'BART answer-aware baseline', None, None),
    ('qg_only', 'Restored QG only', 0.0, 0.0),
    ('qg_qa', 'Restored QG + QA', 0.8, 0.0),
    ('qg_uw', 'Restored QG + KD', 0.0, 0.15),
    ('full', 'Restored QG + QA + KD', 0.8, 0.15),
]


def load(name):
    path = EXP / f'{name}_combined.json'
    return json.loads(path.read_text()) if path.exists() else None


def f(x):
    if x is None: return '—'
    return f'{x:.4f}' if isinstance(x, float) else str(x)

rows = []
for key, label, alpha, beta in RUNS:
    d = load(key)
    if not d: continue
    rows.append((label, alpha, beta, d))

lines = [
    '# Experimental Results', '',
    'These are results from the restored portfolio implementation. They are **not an exact reproduction** of Wang et al. (EMNLP 2022): the repository uses BART-based components, vanilla Hugging Face SQuAD validation data, and a documented restoration of the original course implementation.', '',
    '## In-domain SQuAD evaluation', '',
    '| Model | α | β | N | BLEU | ROUGE-L F1 | QA EM | QA F1 | Lexical novelty | Uncommon-word rate |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for label, alpha, beta, d in rows:
    lines.append('| ' + ' | '.join([
        label, f(alpha), f(beta), f(d.get('num_examples')), f(d.get('bleu')),
        f(d.get('rougeL_f1')), f(d.get('answer_consistency_em')),
        f(d.get('answer_consistency_f1')), f(d.get('lexical_novelty')),
        f(d.get('mean_uncommon_word_rate'))
    ]) + ' |')

lines += ['', '### Interpretation guardrails', '',
          '- The independent-QA EM/F1 columns measure whether an external QA model recovers the intended answer from the generated question and passage.',
          '- Lexical novelty is the fraction of generated question tokens absent from the passage/answer under the repository metric.',
          '- The uncommon-word rate is a portfolio analysis metric based on low corpus frequency; it is **not** the paper\'s AGS metric.',
          '- Compare the restored ablations against one another; do not treat the EMNLP paper\'s published BLEU as directly comparable because the data processing/splits and implementation differ.', '']

for domain in ['bart_hotpot', 'full_hotpot']:
    d = load(domain)
    if d:
        label = 'BART baseline' if domain.startswith('bart') else 'Restored full model'
        lines += [f'## Cross-domain HotpotQA — {label}', '',
                  f'- Examples: **{d.get("num_examples")}**',
                  f'- BLEU: **{d.get("bleu", 0):.4f}**',
                  f'- ROUGE-L F1: **{d.get("rougeL_f1", 0):.4f}**',
                  f'- QA consistency EM/F1: **{d.get("answer_consistency_em", 0):.4f} / {d.get("answer_consistency_f1", 0):.4f}**',
                  f'- Lexical novelty: **{d.get("lexical_novelty", 0):.4f}**',
                  f'- Uncommon-word rate: **{d.get("mean_uncommon_word_rate", 0):.4f}**', '']

lines += ['## Reproducibility', '',
          'The corresponding JSON metrics, configs, histories, and raw prediction files are produced under `artifacts/experiments/`. Large checkpoints and raw prediction JSONL files are intentionally ignored by Git.', '']

DOC.write_text('\n'.join(lines) + '\n')
print(DOC.read_text())

# Insert/update a compact README results section.
readme = ROOT / 'README.md'
text = readme.read_text()
start = '<!-- AUTO_RESULTS_START -->'
end = '<!-- AUTO_RESULTS_END -->'
compact = [start, '## Restored experimental results', '',
           'The corrected repository now includes a clean BART baseline plus four controlled loss ablations. Full tables and interpretation notes are in [`docs/RESULTS.md`](docs/RESULTS.md).', '']
if rows:
    compact += ['| Model | BLEU | QA F1 | Novelty |', '| --- | ---: | ---: | ---: |']
    for label, _, _, d in rows:
        compact.append(f'| {label} | {f(d.get("bleu"))} | {f(d.get("answer_consistency_f1"))} | {f(d.get("lexical_novelty"))} |')
compact += ['', '> These numbers are from the restored portfolio pipeline and are not presented as an exact reproduction of the EMNLP paper.', end]
block = '\n'.join(compact)
if start in text and end in text:
    before = text.split(start, 1)[0].rstrip()
    after = text.split(end, 1)[1].lstrip()
    text = before + '\n\n' + block + '\n\n' + after
else:
    anchor = '\n## Historical result\n'
    if anchor in text:
        text = text.replace(anchor, '\n' + block + '\n\n## Historical result\n', 1)
    else:
        text = text.rstrip() + '\n\n' + block + '\n'
readme.write_text(text)
