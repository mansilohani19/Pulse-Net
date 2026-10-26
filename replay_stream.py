"""Replays NSL-KDD test rows into the API so the dashboard looks live.

  python replay_stream.py --mode normal    # only normal flows
  python replay_stream.py --mode attack    # only attack flows
  python replay_stream.py --mode mixed     # ~90% normal, ~10% attack
"""
import argparse
import time

import pandas as pd
import requests

from src.features import COLS, FEATURES

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["normal", "attack", "mixed"], default="mixed")
p.add_argument("--n", type=int, default=300, help="how many flows to send")
p.add_argument("--delay", type=float, default=0.15, help="seconds between flows")
p.add_argument("--url", default="http://localhost:8000/predict")
p.add_argument("--seed", type=int, default=None)
args = p.parse_args()

df = pd.read_csv("data/raw/KDDTest+.txt", names=COLS)
is_attack = df["label"] != "normal"
normal, attack = df[~is_attack], df[is_attack]

if args.mode == "normal":
    rows = normal.sample(args.n, random_state=args.seed)
elif args.mode == "attack":
    rows = attack.sample(args.n, random_state=args.seed)
else:
    k = int(args.n * 0.1)
    rows = pd.concat([normal.sample(args.n - k, random_state=args.seed),
                      attack.sample(k, random_state=args.seed)]).sample(frac=1, random_state=args.seed)

session = requests.Session()
for i, (flow, label) in enumerate(zip(rows[FEATURES].to_dict("records"), rows["label"]), 1):
    r = session.post(args.url, json={"flow": flow, "true_label": label}, timeout=10)
    r.raise_for_status()
    ev = r.json()
    print(f"{i:4d}  {label:<14} score={ev['score']:.4f}  {'ALERT' if ev['is_anomaly'] else 'ok'}")
    time.sleep(args.delay)
