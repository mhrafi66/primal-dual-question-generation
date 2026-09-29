#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from primal_dual_qg.restored_model import MODEL_NAME, PrimalDualQuestionGenerator, build_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dataset', choices=['squad', 'jsonl'], default='squad')
    p.add_argument('--input-jsonl')
    p.add_argument('--split', default='validation')
    p.add_argument('--max-samples', type=int, default=500)
    p.add_argument('--num-beams', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=48)
    return p.parse_args()


def rows_from_args(args):
    if args.dataset == 'squad':
        ds = load_dataset('squad', split=args.split)
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        return [
            {
                'context': row['context'],
                'answer': row['answers']['text'][0],
                'reference_question': row['question'],
            }
            for row in ds
        ]
    if not args.input_jsonl:
        raise SystemExit('--input-jsonl required for --dataset jsonl')
    rows = [json.loads(x) for x in Path(args.input_jsonl).read_text().splitlines() if x.strip()]
    return rows[:args.max_samples] if args.max_samples else rows


def qg_inputs(tokenizer, context: str, answer: str, device):
    enc = tokenizer(context, answer, truncation='only_first', max_length=512, return_tensors='pt')
    seq_ids = enc.sequence_ids(0)
    segment_ids = []
    last_real = 0
    for seq_id in seq_ids:
        if seq_id is None:
            segment_ids.append(last_real)
        else:
            last_real = int(seq_id)
            segment_ids.append(last_real)
    input_ids = enc['input_ids'].to(device)
    task_ids = torch.zeros_like(input_ids)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)
    return input_ids, task_ids, segment_ids


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = build_tokenizer(MODEL_NAME)
    model = PrimalDualQuestionGenerator(MODEL_NAME, with_teacher=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing, unexpected = model.load_state_dict(checkpoint['model_param'], strict=False)
    missing = [k for k in missing if not k.startswith('teacher_')]
    if missing or unexpected:
        print('checkpoint load notes:', {'missing': missing, 'unexpected': unexpected})
    model.to(device).eval()

    rows = rows_from_args(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as f:
        for row in tqdm(rows, desc=f"generate {Path(args.checkpoint).stem}"):
            ids, task, segment = qg_inputs(tokenizer, row['context'], row['answer'], device)
            generated = model.generate_questions(
                ids, task, segment,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
            record = dict(row)
            record['generated_question'] = tokenizer.decode(generated[0], skip_special_tokens=True)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'wrote {len(rows)} predictions to {output}')


if __name__ == '__main__':
    main()
