"""Target-first GPT-2 member: a generative LM over rows written as text, scored as a Bayes classifier.

Method (Chris Deotte's S6E10 first-place recipe, from his write-up; this is our own code): each row becomes

    <bos> fraud yes, amount 12.34, hours 5.2, ..., used vpn 0<eos>

with the label first and the feature fragments in a random order (a new order every time a row is seen, as in
GReaT, Borisov et al. 2023). GPT-2 is fine-tuned on the training fold's rows as plain next-token prediction.
A held-out row is scored twice, once with "yes" and once with "no"; the difference of the two sequence
log-likelihoods is the posterior log-odds log p(yes, x) - log p(no, x), averaged over several fixed fragment
orders. Nothing is tuned or stopped on the validation fold: training runs for a fixed time or
`--epochs` passes.

Several GPUs: the folds are shared out to one worker process per GPU (`--workers`), each writing
runs/<name>/fold<k>.npz; the parent merges them into oof.npy, test.npy and meta.json. Each fold scores the
test set with one order (fold k uses order k), so the 5-fold test mean covers five orders.

Usage (Kaggle GPU kernel; needs `transformers`, preinstalled there):
  python train_lm.py --name r14_gpt2 --train-min 95 --oof-orders 3
Smoke test (CPU, tiny random model, saves nothing):
  python train_lm.py --name smoke --model sshleifer/tiny-gpt2 --rows 3000 --train-min 1 --folds 0
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import BOOL, CAT_STR, NUM, RUNS, TARGET, _find, folds

# Short, mostly single-token names keep the sequences near 120 tokens.
SHORT = {
    "amount_usd": "amount",
    "hours_since_last_txn": "hours",
    "txn_count_last_24h": "txns",
    "distance_from_home_km": "distance",
    "card_age_months": "card age",
    "customer_age": "age",
    "account_balance_usd": "balance",
    "cvv_retry_count": "cvv retries",
    "velocity_score": "velocity",
    "time_of_day_hour": "hour",
    "day_of_week": "weekday",
    "merchant_risk_score": "risk",
    "prior_disputes": "disputes",
    "merchant_category": "category",
    "card_type": "card",
    "auth_method": "auth",
    "channel": "channel",
    "device_type": "device",
    "is_foreign_transaction": "foreign",
    "is_new_merchant": "new merchant",
    "used_vpn": "vpn",
    "ip_country_mismatch": "ip mismatch",
    "billing_shipping_mismatch": "address mismatch",
    "is_ai_generated_scam_attempt": "ai scam",
}
COLS = NUM + CAT_STR + BOOL


def read_text_frames(rows: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """train.csv and test.csv as strings exactly as written in the files, plus the label array.

    `rows` > 0 keeps a stratified subsample of the training rows (smoke tests) and 1,000 test rows.
    """
    tr = pd.read_csv(_find("train.csv"), dtype=str, keep_default_na=False)
    te = pd.read_csv(_find("test.csv"), dtype=str, keep_default_na=False)
    for df in (tr, te):
        for c in BOOL:
            df[c] = df[c].str.lower().map({"true": "1", "false": "0"}).fillna(df[c])
    lab = {"1": 1, "0": 0, "true": 1, "false": 0, "1.0": 1, "0.0": 0}
    y = tr[TARGET].str.lower().map(lab).astype(int).to_numpy()
    if rows:
        from sklearn.model_selection import train_test_split

        keep, _ = train_test_split(np.arange(len(tr)), train_size=rows, stratify=y, random_state=0)
        keep = np.sort(keep)
        tr, y, te = tr.iloc[keep].reset_index(drop=True), y[keep], te.iloc[:1000].reset_index(drop=True)
    return tr, te, y


def fragments(tok, df: pd.DataFrame) -> list[list[list[int]]]:
    """Token ids of every " name value," fragment: result[col][row] is a list of ids.

    Each fragment starts with a space, so concatenating fragment ids in any order tokenizes the same as the
    joined text would (GPT-2's BPE does not merge across a leading space).
    """
    out = []
    for c in COLS:
        texts = [f" {SHORT[c]} {v}," for v in df[c].tolist()]
        out.append(tok(texts, add_special_tokens=False)["input_ids"])
    return out


def build(
    frag: list, rows: np.ndarray, perms: list[np.ndarray], prefix: list[int], eos: int
) -> list[list[int]]:
    """One id sequence per row: prefix (bos + label), the fragments in that row's order, eos."""
    seqs = []
    for r, p in zip(rows, perms, strict=True):
        s = list(prefix)
        for j in p:
            s.extend(frag[j][r])
        s.append(eos)
        seqs.append(s)
    return seqs


def pad(seqs: list[list[int]], pad_id: int):
    """Right-padded id tensor and attention mask."""
    import torch

    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    return ids, att


def seq_loglik(model, ids, att):
    """Sum of next-token log-probabilities of each sequence (positions after the first), in float32."""
    import torch

    with torch.autocast(ids.device.type, dtype=torch.float16, enabled=ids.device.type == "cuda"):
        logits = model(input_ids=ids, attention_mask=att).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:]
    ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * att[:, 1:]
    return ll.sum(1)


def run_fold(a: argparse.Namespace, k: int) -> None:
    """Fine-tune on fold k's training rows, score its validation rows and the test set; write fold<k>.npz."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed + k)
    rng = np.random.default_rng(a.seed + k)
    tr, te, y = read_text_frames(a.rows)
    i_tr, i_va = folds(y)[k]
    tok = AutoTokenizer.from_pretrained(a.model)
    bos = eos = tok.eos_token_id
    pre = {lab: [bos, *tok(f"fraud {lab},", add_special_tokens=False)["input_ids"]] for lab in ("yes", "no")}
    ftr, fte = fragments(tok, tr), fragments(tok, te)
    print(f"[fold {k}] tokenized in {time.time() - t0:.0f}s on {dev}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    n_col = len(COLS)
    steps_epoch = math.ceil(len(i_tr) / a.bs)
    total = math.ceil(a.epochs * steps_epoch)
    orders = [rng.permutation(i_tr)]  # a new shuffle per epoch, drawn when the epoch starts
    budget = a.train_min * 60
    t1, step, seen_tok = time.time(), 0, 0
    model.train()
    while step < total:
        frac = max((time.time() - t1) / budget, step / total)
        if frac >= 1:
            break
        warm = min(1.0, (step + 1) / a.warmup)
        for g in opt.param_groups:
            g["lr"] = a.lr * warm * 0.5 * (1 + math.cos(math.pi * frac))
        ep, at = divmod(step, steps_epoch)
        if ep == len(orders):
            orders.append(rng.permutation(i_tr))
        rows = orders[ep][at * a.bs : (at + 1) * a.bs]
        perms = [rng.permutation(n_col) for _ in rows]
        seqs = []
        for r, p in zip(rows, perms, strict=True):
            lab = "yes" if y[r] == 1 else "no"
            seqs.extend(build(ftr, np.array([r]), [p], pre[lab], eos))
        ids, att = pad(seqs, eos)
        ids, att = ids.to(dev), att.to(dev)
        labels = ids.masked_fill(att == 0, -100)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            loss = model(input_ids=ids, attention_mask=att, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        seen_tok += int(att.sum())
        step += 1
        if step % 500 == 0 or step == 1:
            el = time.time() - t1
            print(
                f"[fold {k}] step {step}/{total} loss {loss.item():.4f} "
                f"{seen_tok / el:.0f} tok/s {el / 60:.1f} min",
                flush=True,
            )
    train_rows = min(step * a.bs, int(a.epochs * len(i_tr)))
    print(
        f"[fold {k}] trained {step} steps ({train_rows} rows) in {(time.time() - t1) / 60:.1f} min",
        flush=True,
    )

    model.eval()
    fixed = [np.arange(n_col)] + [np.random.default_rng(1000 + o).permutation(n_col) for o in range(1, 8)]

    @torch.no_grad()
    def score(frag: list, rows: np.ndarray, orders: list[np.ndarray]) -> np.ndarray:
        """Log-odds of each row, averaged over the given fragment orders."""
        out = np.zeros(len(rows))
        for p in orders:
            for s in range(0, len(rows), a.eval_bs):
                rr = rows[s : s + a.eval_bs]
                ll = {}
                for lab in ("yes", "no"):
                    ids, att = pad(build(frag, rr, [p] * len(rr), pre[lab], eos), eos)
                    ll[lab] = seq_loglik(model, ids.to(dev), att.to(dev)).cpu().numpy()
                out[s : s + len(rr)] += (ll["yes"] - ll["no"]) / len(orders)
        return out

    t2 = time.time()
    oof = score(ftr, i_va, fixed[: a.oof_orders])
    auc = roc_auc_score(y[i_va], oof)
    print(
        f"[fold {k}] val auc {auc:.5f} ({a.oof_orders} orders, {(time.time() - t2) / 60:.1f} min)", flush=True
    )
    t3 = time.time()
    test = score(fte, np.arange(len(te)), [fixed[(k + j) % len(fixed)] for j in range(a.test_orders)])
    print(
        f"[fold {k}] test scored in {(time.time() - t3) / 60:.1f} min; "
        f"fold total {(time.time() - t0) / 60:.1f} min"
    )
    d = RUNS / a.name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"fold{k}.npz", idx=i_va, oof=oof, test=test, auc=auc, steps=step, rows=train_rows)


def merge(a: argparse.Namespace, ks: list[int]) -> None:
    """Combine fold<k>.npz files into the standard run files (oof.npy, test.npy, meta.json)."""
    from common import save_run

    _, te, y = read_text_frames(a.rows)
    d = RUNS / a.name
    oof, test, scores, steps = np.full(len(y), np.nan), np.zeros(len(te)), [], []
    for k in ks:
        z = np.load(d / f"fold{k}.npz", allow_pickle=False)
        oof[z["idx"]] = z["oof"]
        test += z["test"] / len(ks)
        scores.append(float(z["auc"]))
        steps.append(int(z["steps"]))
    m = ~np.isnan(oof)
    cv = roc_auc_score(y[m], oof[m])
    print(f"{a.name}: CV {cv:.5f}  folds {np.round(scores, 5).tolist()}")
    if len(ks) == 5 and not a.rows:
        save_run(a.name, oof, test, dict(vars(a), cv=cv, folds=scores, steps=steps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="gpt2", help="Hugging Face causal LM id")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=0, help="parallel worker processes; 0 = one per GPU")
    ap.add_argument("--worker", default="", help=argparse.SUPPRESS)
    ap.add_argument("--train-min", type=float, default=95, help="training time budget per fold (minutes)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--oof-orders", type=int, default=3)
    ap.add_argument("--test-orders", type=int, default=1, help="test orders per fold, from fixed[k]")
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the fold's training rows")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--merge-only", action="store_true", help="combine fold files from pods")
    ap.add_argument("--rows", type=int, default=0, help="smoke test on a stratified subsample; saves no run")
    a = ap.parse_args()
    ks = [int(k) for k in a.folds.split(",")]
    if a.merge_only:
        merge(a, ks)
        return
    if a.worker:
        for k in (int(x) for x in a.worker.split(",")):
            run_fold(a, k)
        return
    n = a.workers
    if n == 0:
        import torch

        n = max(1, torch.cuda.device_count())
    groups = [ks[i::n] for i in range(n)]
    procs = []
    for g, part in enumerate(groups):
        if not part:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)) if n > 1 else dict(os.environ)
        cmd = [sys.executable, __file__, *sys.argv[1:], "--worker", ",".join(map(str, part))]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"worker exit codes {codes}")
    merge(a, ks)


if __name__ == "__main__":
    main()
