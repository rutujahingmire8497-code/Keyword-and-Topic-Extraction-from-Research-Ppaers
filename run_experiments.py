"""Run baseline (TextRank) vs advanced (BART/T5) and compare with ROUGE.

Examples:
    python run_experiments.py --dataset cnn_dailymail --n 100
    python run_experiments.py --dataset arxiv --n 50 --models bart t5
"""
import argparse
import json
import time
from pathlib import Path

from src.abstractive import AbstractiveSummarizer
from src.data import load_samples
from src.evaluate import rouge_corpus
from src.textrank import textrank_summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cnn_dailymail", choices=["cnn_dailymail", "arxiv"])
    ap.add_argument("--n", type=int, default=100, help="number of test documents")
    ap.add_argument("--models", nargs="+", default=["bart", "t5"], choices=["bart", "t5"])
    ap.add_argument("--sentences", type=int, default=3, help="TextRank summary length")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    docs, refs = load_samples(args.dataset, n=args.n)
    results, outputs = {}, {"reference": refs}

    t0 = time.time()
    preds = [textrank_summarize(d, args.sentences) for d in docs]
    results["TextRank (baseline)"] = {**rouge_corpus(refs, preds), "seconds": round(time.time() - t0, 1)}
    outputs["textrank"] = preds

    for key in args.models:
        model = AbstractiveSummarizer(key)
        t0 = time.time()
        preds = model.summarize_many(docs)
        results[f"{key.upper()} ({model.model_name})"] = {
            **rouge_corpus(refs, preds),
            "seconds": round(time.time() - t0, 1),
        }
        outputs[key] = preds

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / f"{args.dataset}_scores.json").write_text(json.dumps(results, indent=2))
    (out / f"{args.dataset}_outputs.json").write_text(json.dumps(outputs, indent=2))

    print(f"\nDataset: {args.dataset}  |  documents: {len(docs)}")
    print(f"{'Model':45s} {'R-1':>6s} {'R-2':>6s} {'R-L':>6s} {'sec':>7s}")
    for name, r in results.items():
        print(f"{name:45s} {r['rouge1']:6.2f} {r['rouge2']:6.2f} {r['rougeL']:6.2f} {r['seconds']:7.1f}")


if __name__ == "__main__":
    main()
