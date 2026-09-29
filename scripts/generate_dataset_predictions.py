#!/usr/bin/env python3
"""Generate JSONL questions for SQuAD or a prepared JSONL evaluation set."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm.auto import tqdm

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dataset', choices=['squad','jsonl'], default='squad')
    p.add_argument('--input-jsonl')
    p.add_argument('--split', default='validation')
    p.add_argument('--max-samples', type=int, default=500)
    p.add_argument('--num-beams', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=48)
    return p.parse_args()

def load_rows(args):
    if args.dataset=='squad':
        ds=load_dataset('squad', split=args.split)
        if args.max_samples:
            ds=ds.select(range(min(args.max_samples,len(ds))))
        return [
            {'context':r['context'], 'answer':r['answers']['text'][0], 'reference_question':r['question']}
            for r in ds
        ]
    if not args.input_jsonl:
        raise SystemExit('--input-jsonl is required for --dataset jsonl')
    rows=[json.loads(x) for x in Path(args.input_jsonl).read_text().splitlines() if x.strip()]
    return rows[:args.max_samples] if args.max_samples else rows

def main():
    args=parse_args(); rows=load_rows(args)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForSeq2SeqLM.from_pretrained(args.model)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        for row in tqdm(rows, desc='generate'):
            prompt=f"answer: {row['answer']} context: {row['context']}"
            inp=tok(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                ids=model.generate(**inp, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, early_stopping=True)
            record=dict(row)
            record['generated_question']=tok.decode(ids[0], skip_special_tokens=True)
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
    print(f'wrote {len(rows)} predictions to {out}')
if __name__=='__main__': main()
