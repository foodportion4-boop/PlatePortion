"""Run 300 plates; calibrate on 150 only. GT never enters pipeline.run."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from plateclock.io.loaders import load_record
from plateclock.io.class_names import CLASS_NAMES
from plateclock.pipeline import run, load_config
from plateclock.plate.rectify import warp
from plateclock.plate.group_regions import group_regions
from plateclock.plate.evaluation import match_regions, threshold_table

def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/raw/foodseg103')
    p.add_argument('--subset', default='data/processed/plate_subset.csv')
    p.add_argument('--config', default='configs/week2.yaml')
    p.add_argument('--out', default='results/week2/evaluation')
    a = p.parse_args()
    cfg = load_config(a.config)
    from plateclock.plate.segment_food import FoodSegmenter
    model = FoodSegmenter(cfg['model']['checkpoint'])
    records = sorted(csv.DictReader(Path(a.subset).open()), key=lambda r:r['id'])
    if len(records) != 300 or len({r['id'] for r in records}) != 300:
        raise ValueError('Require exactly 300 unique validation records; run handoff check first')
    # Persist the split. Re-running the same sorted IDs gives the same assignment.
    rng = np.random.default_rng(42)
    calibration_indices = set(map(int, rng.permutation(300)[:150]))
    split = {r['id']:('calibration' if i in calibration_indices else 'holdout') for i,r in enumerate(records)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/'split.json').write_text(json.dumps(split, indent=2))
    candidates, review, heldout = [], [], []
    for index, row in enumerate(records):
        sample = load_record(row, a.root)
        if sample.class_names != CLASS_NAMES:
            raise ValueError('Class mapping mismatch')
        result = run(sample.image, model, cfg)
        review.append(dict(id=sample.sample_id, split=split[sample.sample_id], status=result['status'],
                           items=len(result['items']), text=result['text'], reason=result['reason'],
                           human_review='pending', failure_type='pending'))
        if index < 20:
            Image.fromarray(result['overlay']).save(out/f'overlay_{index:03d}.png')
        if result['status'] != 'plate_failure':
            # Warp GT ONLY for scoring, using the predicted transform/plate frame.
            gt = warp(sample.mask, result['affine'], is_mask=True)
            valid = result['plate'] & (gt != 255)
            gold = group_regions(gt, np.ones(gt.shape, np.float32), valid, CLASS_NAMES, cfg)
            matches = match_regions(result['regions'], gold, valid)
            for number, (region, decision, correct) in enumerate(zip(result['regions'], result['all_decisions'], matches)):
                target = candidates if split[sample.sample_id] == 'calibration' else heldout
                target.append(dict(id=sample.sample_id, region=number, label=region.label,
                                       score=decision.score, correct=correct))
        print(index+1, '/', len(records), sample.sample_id, result['status'])
    write_csv(out/'review.csv', review, ['id','split','status','items','text','reason','human_review','failure_type'])
    write_csv(out/'candidates.csv', candidates, ['id','region','label','score','correct'])
    write_csv(out/'holdout_candidates.csv', heldout, ['id','region','label','score','correct'])
    frozen = {'thresholds': cfg['confidence'], 'note': 'Config thresholds only. Candidate precision conditional on predicted plate frame; not food recall or geometry accuracy.'}
    for key in ['omit_below', 'keep_at']:
        kept = [r for r in heldout if r['score'] >= cfg['confidence'][key]]
        frozen[key] = {'retained':len(kept), 'candidates':len(heldout),
                       'candidate_coverage':len(kept)/len(heldout) if heldout else 0.,
                       'precision':sum(r['correct'] for r in kept)/len(kept) if kept else None}
    (out/'holdout_metrics.json').write_text(json.dumps(frozen, indent=2))
    stats = dict(images=len(review), calibration_images=150, holdout_images=150,
                 plate_failures=sum(r['status']=='plate_failure' for r in review),
                 no_confident_food=sum(r['status']=='no_confident_food' for r in review),
                 median_spoken_items=float(np.median([r['items'] for r in review])),
                 calibration_candidates=len(candidates))
    (out/'summary.json').write_text(json.dumps(stats, indent=2))
    if not candidates:
        raise RuntimeError('No calibration candidates. Tables saved; fix model/detector before calibration.')
    scores = np.array([r['score'] for r in candidates]); correct = np.array([r['correct'] for r in candidates])
    curve = threshold_table(scores, correct)
    write_csv(out/'precision_threshold.csv', curve, ['threshold','retained','coverage','precision'])
    bins = []
    for i in range(10):
        mask = (scores >= i/10) & ((scores < (i+1)/10) if i < 9 else (scores <= 1))
        if mask.any():
            bins.append(dict(mean_score=float(scores[mask].mean()), accuracy=float(correct[mask].mean()), count=int(mask.sum())))
    write_csv(out/'reliability.csv', bins, ['mean_score','accuracy','count'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    defined = [r for r in curve if r['precision'] is not None]
    ax[0].plot([r['threshold'] for r in defined], [r['precision'] for r in defined], 'o-')
    ax[0].set(xlabel='Score threshold', ylabel='Same-group IoU precision', ylim=(0,1.05))
    ax[1].plot([r['mean_score'] for r in bins], [r['accuracy'] for r in bins], 'o-')
    ax[1].plot([0,1],[0,1], '--', color='gray')
    ax[1].set(xlabel='Mean heuristic score', ylabel='Observed correctness', ylim=(0,1.05))
    fig.tight_layout(); fig.savefig(out/'confidence_curves.png', dpi=160); plt.close(fig)
    suggestions = {}
    for name, target in [('omit_below', .75), ('keep_at', .90)]:
        eligible = [r['threshold'] for r in curve if r['retained'] >= 20 and r['precision'] is not None and r['precision'] >= target]
        suggestions[name] = min(eligible) if eligible else None
    suggestions['note'] = 'Proposals only; inspect reliability and visual correctness. If null, gate remains incomplete. Scores are not probabilities.'
    (out/'threshold_proposals.json').write_text(json.dumps(suggestions, indent=2))
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
