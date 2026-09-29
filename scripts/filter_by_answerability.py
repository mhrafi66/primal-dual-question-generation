#!/usr/bin/env python3
"""Filter generated questions using an independent extractive QA model."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from transformers import pipeline
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from primal_dual_qg.metrics import token_f1_score, exact_match_score

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('predictions')
    p.add_argument('--accepted', default='artifacts/experiments/accepted.jsonl')
    p.add_argument('--rejected', default='artifacts/experiments/rejected.jsonl')
    p.add_argument('--qa-model', default='distilbert/distilbert-base-cased-distilled-squad')
    p.add_argument('--min-f1', type=float, default=0.5)
    p.add_argument('--min-confidence', type=float, default=0.0)
    return p.parse_args()

def main():
    a=parse_args(); qa=pipeline('question-answering', model=a.qa_model, device=0 if torch.cuda.is_available() else -1)
    rows=[json.loads(x) for x in Path(a.predictions).read_text().splitlines() if x.strip()]
    good=[]; bad=[]
    for r in rows:
        q=qa(question=r['generated_question'],context=r['context'])
        r=dict(r)
        r['qa_predicted_answer']=q['answer']; r['qa_confidence']=float(q['score'])
        r['qa_em']=exact_match_score(q['answer'],r['answer']); r['qa_f1']=token_f1_score(q['answer'],r['answer'])
        (good if r['qa_f1']>=a.min_f1 and r['qa_confidence']>=a.min_confidence else bad).append(r)
    for path, data in [(Path(a.accepted),good),(Path(a.rejected),bad)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as f:
            for r in data: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps({'total':len(rows),'accepted':len(good),'rejected':len(bad)},indent=2))
if __name__=='__main__': main()
