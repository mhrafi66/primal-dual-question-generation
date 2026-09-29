#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--metrics', required=True)
p.add_argument('--uncommon', required=True)
p.add_argument('--run', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()

m = json.loads(Path(a.metrics).read_text())
u = json.loads(Path(a.uncommon).read_text())
m['run'] = a.run
m['mean_uncommon_word_rate'] = u.get('mean_uncommon_word_rate')
Path(a.output).write_text(json.dumps(m, indent=2) + '\n')
print(Path(a.output).read_text())
