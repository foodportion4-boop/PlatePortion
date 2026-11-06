"""Train SegFormer-B0 from random initialization on FoodSeg103 train only.

Runs on Mac MPS or a free Colab GPU. This is a practical baseline, not a promised score.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

class FoodTrain(Dataset):
    def __init__(self,root,rows):self.root=Path(root);self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        im=Image.open(self.root/row['image_path']).convert('RGB').resize((256,256))
        mask=Image.open(self.root/row['mask_path']).resize((256,256),Image.Resampling.NEAREST)
        x=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255
        y=np.asarray(mask,dtype=np.int64)
        return torch.tensor(x),torch.tensor(y)

p=argparse.ArgumentParser()
p.add_argument('--root',default='data/raw/foodseg103')
p.add_argument('--output',default='checkpoints/foodseg-b0')
p.add_argument('--epochs',type=int,default=12)
p.add_argument('--batch-size',type=int,default=8)
a=p.parse_args()
rows=list(csv.DictReader(open(Path(a.root)/'train.csv')))
if len(rows)!=4983:raise ValueError(f'Expected 4983 train samples, got {len(rows)}')
device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model=SegformerForSemanticSegmentation(SegformerConfig(num_labels=104,hidden_sizes=[32,64,160,256],depths=[2,2,2,2],decoder_hidden_size=256)).to(device)
loader=DataLoader(FoodTrain(a.root,rows),batch_size=a.batch_size,shuffle=True,num_workers=0)
opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=.01)
for epoch in range(a.epochs):
    model.train();running=0
    for x,y in loader:
        x=x.to(device);y=y.to(device)
        opt.zero_grad();loss=model(pixel_values=x,labels=y).loss;loss.backward();opt.step()
        running+=float(loss.detach().cpu())
    print(f'epoch {epoch+1}/{a.epochs}: loss {running/len(loader):.4f}',flush=True)
    Path(a.output).mkdir(parents=True,exist_ok=True)
    model.save_pretrained(a.output,safe_serialization=True)
    # Images passed to inference should receive the same 0–1 pixel scaling as training.
    SegformerImageProcessor(do_resize=True,size={'height':256,'width':256},do_rescale=True,do_normalize=False).save_pretrained(a.output)
