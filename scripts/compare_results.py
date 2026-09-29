#!/usr/bin/env python3
"""Combine metric JSON files into a Markdown comparison table."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('metrics',nargs='+'); p.add_argument('--output',default='artifacts/RESULTS.md'); a=p.parse_args()
    rows=[]
    for fn in a.metrics:
        d=json.loads(Path(fn).read_text()); d['run']=d.get('run',Path(fn).stem.replace('_metrics','')); rows.append(d)
    cols=['run','num_examples','bleu','rougeL_f1','answer_consistency_em','answer_consistency_f1','lexical_novelty','mean_uncommon_word_rate']
    header='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']+['---:']*(len(cols)-1))+' |\n'
    lines=[]
    for r in rows:
        vals=[]
        for c in cols:
            v=r.get(c,'—')
            vals.append(f'{v:.4f}' if isinstance(v,float) else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(header+'\n'.join(lines)+'\n')
    print(out.read_text())
if __name__=='__main__': main()
