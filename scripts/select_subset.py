"""Create contact sheets for manual selection, then validate a 300-ID CSV."""
import argparse
import csv
from pathlib import Path
from PIL import Image, ImageDraw
p = argparse.ArgumentParser()
p.add_argument('--root', default='data/raw/foodseg103')
p.add_argument('--selected', help='Text file with one validation ID per line')
a = p.parse_args()
root = Path(a.root)
rows = list(csv.DictReader((root / 'validation.csv').open()))
if not a.selected:
    out = Path('results/figures/contact_sheets'); out.mkdir(parents=True, exist_ok=True)
    for start in range(0,len(rows),100):
        sheet = Image.new('RGB',(1000,1100),'white'); d=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows[start:start+100]):
            im = Image.open(root / row['image_path']).convert('RGB'); im.thumbnail((94,85))
            x=(j%10)*100; y=(j//10)*110
            sheet.paste(im,(x+(100-im.width)//2,y)); d.text((x+2,y+86),row['id'],fill='black')
        sheet.save(out / f'{start//100:02d}.jpg', quality=90)
    print('Review contact sheets; select 300 full, centered plates. IDs must come from validation.csv.')
else:
    ids=[s.strip() for s in Path(a.selected).read_text().splitlines() if s.strip()]
    if len(ids)!=300 or len(set(ids))!=300: raise ValueError('Exactly 300 distinct IDs required')
    lookup={r['id']:r for r in rows}
    if set(ids)-set(lookup): raise ValueError('IDs absent from validation split')
    out=Path('data/processed/plate_subset.csv'); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','image_path','mask_path']);w.writeheader();w.writerows(lookup[i] for i in ids)
    print('Wrote',out)
