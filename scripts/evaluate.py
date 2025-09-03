"""Evaluation harness for TechTrans RAG.

Usage:
  python scripts/evaluate.py --gold data/eval/gold.jsonl --top-k 5 --limit 0

Metrics:
  * retrieval_recall@K: whether any retrieved chunk belongs to the gold doc_id
  * first_hit_rank: position (1-based) of first chunk from gold doc (or 0 if none)
  * mrr: reciprocal rank of first hit
  * answer_contains_overlap: heuristic string containment overlap between system answer and gold answer
  * answer_cites_required: if gold provides must_cite list, check those tokens present

Environment:
  Requires OPENAI_API_KEY for answer generation (already used by app). If absent, answer metrics are skipped.

The harness uses Flask test client so the running server is not required.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any

from app import create_app


def load_gold(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                continue
    return items


def jaccard(a: str, b: str) -> float:
    ta = set(w.strip('.,;:"').lower() for w in a.split())
    tb = set(w.strip('.,;:"').lower() for w in b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def evaluate(gold_path: str, top_k: int, limit: int = 0, include_answer: bool = True):
    gold = load_gold(gold_path)
    if limit > 0:
        gold = gold[:limit]
    app = create_app()
    client = app.test_client()

    results: List[Dict[str, Any]] = []
    for item in gold:
        qid = item.get('qid')
        query = item.get('query')
        gold_doc = item.get('doc_id')
        gold_answer = item.get('answer') or ''
        must_cite = item.get('must_cite') or []

        resp = client.post('/api/query', json={'query': query, 'top_k': top_k})
        js = resp.get_json() or {}
        matches = js.get('matches') or []
        answer = js.get('answer') or ''

        hit_ranks = [i+1 for i, m in enumerate(matches) if m.get('doc_id') == gold_doc]
        first_hit = hit_ranks[0] if hit_ranks else 0
        recall = 1.0 if first_hit else 0.0
        rr = 1.0 / first_hit if first_hit else 0.0
        overlap = jaccard(answer, gold_answer) if answer and gold_answer else 0.0
        cite_ok = True
        if must_cite and answer:
            for token in must_cite:
                if token not in answer:
                    cite_ok = False
                    break
        results.append({
            'qid': qid,
            'first_hit_rank': first_hit,
            'recall@k': recall,
            'rr': rr,
            'answer_jaccard': overlap,
            'cite_ok': cite_ok,
            'n_matches': len(matches)
        })

    # Aggregate
    agg = {
        'questions': len(results),
        'retrieval_recall@k': sum(r['recall@k'] for r in results) / len(results) if results else 0.0,
        'mrr': sum(r['rr'] for r in results) / len(results) if results else 0.0,
        'answer_jaccard_mean': statistics.mean([r['answer_jaccard'] for r in results]) if results else 0.0,
        'answer_jaccard_median': statistics.median([r['answer_jaccard'] for r in results]) if results else 0.0,
        'cite_accuracy': sum(1 for r in results if r['cite_ok']) / len(results) if results else 0.0,
    }

    return {'per_question': results, 'aggregate': agg}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gold', default='data/eval/gold.jsonl')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--limit', type=int, default=0, help='Limit number of gold examples (0 = all)')
    p.add_argument('--no-answer', action='store_true', help='Skip answer metrics (retrieval only)')
    p.add_argument('--out', default='data/eval/report.json')
    args = p.parse_args()

    report = evaluate(args.gold, args.top_k, limit=args.limit, include_answer=not args.no_answer)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report['aggregate'], indent=2))
    print(f'Wrote detailed report to {out_path}')


if __name__ == '__main__':
    main()
