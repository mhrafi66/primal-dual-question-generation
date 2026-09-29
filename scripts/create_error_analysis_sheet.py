#!/usr/bin/env python3
"""Create a CSV worksheet for systematic manual error analysis."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('predictions'); p.add_argument('--output',default='artifacts/error_analysis.csv'); p.add_argument('--limit',type=int,default=200); a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.predictions).read_text().splitlines() if x.strip()][:a.limit]
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    fields=['context','answer','reference_question','generated_question','qa_f1','qa_confidence','error_type','grammaticality_1_5','relevance_1_5','answerability_1_5','naturalness_1_5','novelty_1_5','notes']
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,'') for k in fields})
    print(f'wrote {len(rows)} rows to {out}')
if __name__=='__main__': main()
