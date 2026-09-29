#!/usr/bin/env python3
"""Prepare a small HotpotQA JSONL set for cross-domain generation evaluation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from datasets import load_dataset

def flatten_context(value):
    if isinstance(value, dict):
        sentences=value.get('sentences',[])
        return ' '.join(s for group in sentences for s in (group if isinstance(group,list) else [group]))
    if isinstance(value, list):
        parts=[]
        for item in value:
            if isinstance(item,(list,tuple)) and len(item)>=2:
                sent=item[1]; parts.extend(sent if isinstance(sent,list) else [sent])
            else: parts.append(str(item))
        return ' '.join(parts)
    return str(value)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='artifacts/hotpotqa_eval.jsonl'); p.add_argument('--max-samples',type=int,default=500); p.add_argument('--split',default='validation'); a=p.parse_args()
    ds=load_dataset('hotpot_qa','distractor',split=a.split,trust_remote_code=True)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); count=0
    with out.open('w') as f:
        for r in ds:
            context=flatten_context(r['context']); answer=str(r['answer'])
            if answer.lower() not in context.lower(): continue
            f.write(json.dumps({'context':context,'answer':answer,'reference_question':r['question']},ensure_ascii=False)+'\n'); count+=1
            if count>=a.max_samples: break
    print(f'wrote {count} examples to {out}')
if __name__=='__main__': main()
