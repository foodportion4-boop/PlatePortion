"""Read-only integration check; does not retrain Week 1."""
import argparse
import csv
import json
from pathlib import Path
from plateclock.io.class_names import CLASS_NAMES
from plateclock.io.loaders import load_record

p = argparse.ArgumentParser()
p.add_argument('--root', default='data/raw/foodseg103')
p.add_argument('--subset', default='data/processed/plate_subset.csv')
p.add_argument('--checkpoint', default='checkpoints/foodseg-b0')
a = p.parse_args()
root, checkpoint = Path(a.root), Path(a.checkpoint)
required = [root/'classes.json', root/'validation.csv', Path(a.subset),
            checkpoint/'config.json', checkpoint/'preprocessor_config.json',
            Path('results/tables/seg_baseline.csv')]
missing = [str(f) for f in required if not f.is_file()]
if not any((checkpoint/f).is_file() for f in ['model.safetensors','pytorch_model.bin','model.safetensors.index.json','pytorch_model.bin.index.json']):
    missing.append(str(checkpoint) + '/model weights')
if missing:
    raise SystemExit('Ask teammates for missing Week 1 artifacts:\n'+'\n'.join(missing))
names = {int(k):v for k,v in json.loads((root/'classes.json').read_text()).items()}
if names != CLASS_NAMES:
    raise SystemExit('Class-ID mapping differs from the attached Week 1 mapping; reconcile before inference')
if json.loads((checkpoint/'config.json').read_text()).get('num_labels', 104) != 104:
    raise SystemExit('Checkpoint is not the 104-class Week 1 model')
rows = list(csv.DictReader(Path(a.subset).open()))
validation = {(r['id'],r['image_path'],r['mask_path']) for r in csv.DictReader((root/'validation.csv').open())}
if len(rows) != 300 or len({r['id'] for r in rows}) != 300:
    raise SystemExit('Need 300 distinct subset IDs')
for row in rows:
    if (row['id'],row['image_path'],row['mask_path']) not in validation:
        raise SystemExit('Subset row is not from validation: '+row['id'])
    load_record(row, root)
print('PASS: checkpoint files, exact classes and 300 loadable validation records')
if not Path('results/figures/checkpoint1/review.csv').exists():
    print('REVIEW PENDING: checkpoint1 review.csv absent; this does not block code integration.')
print('MANUAL GATE: verify completed Week 1 review, actual mIoU, detector results and go/no-go with teammates.')
