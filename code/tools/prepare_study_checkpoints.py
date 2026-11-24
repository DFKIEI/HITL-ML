"""Continue existing epoch-20 models to epoch 25; preserve all source checkpoints.
Run from the repository root: python code/tools/prepare_study_checkpoints.py
"""
import argparse
import json
import hashlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from data.data_loader import load_dataset
from model import get_model
from training.training import train_model
from training.training_utils import load_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',default=['CIFAR10','PAMAP2'],choices=['CIFAR10','PAMAP2','CIFAR100'])
    parser.add_argument('--epochs',type=int,default=25)
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='auto',choices=['auto','cuda','mps','cpu'])
    args=parser.parse_args()
    device=torch.device(('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device)
    for dataset in args.datasets:
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        folder=ROOT/'saved_checkpoints'/dataset
        destination=folder/f'checkpoint_epoch_{args.epochs}.pt'
        if destination.exists():
            print(f'{destination} already exists; skipping.',flush=True);continue
        train,val,test,classes,shape=load_dataset(dataset,args.batch)
        model=get_model('CNN_'+dataset,shape,classes).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
        source=folder/'checkpoint_epoch_20.pt'
        start,_=load_checkpoint(model,optimizer,source) if source.exists() else (0,{})
        if start>=args.epochs:raise ValueError('Source epoch must precede requested epoch')
        print(f'{dataset}: epochs {start+1}–{args.epochs} on {device}',flush=True)
        report=ROOT/'reports'/'study_pretraining'/dataset;report.mkdir(parents=True,exist_ok=True)
        train_model(model,optimizer,train,val,test,device,args.epochs-start,100,SimpleNamespace(get=lambda:0.0),str(report),
                    checkpoint_dir=str(folder),logger=SimpleNamespace(log_model_results=print),start_epoch=start)
        if not destination.exists():raise RuntimeError(f'Missing expected checkpoint: {destination}')
        saved=torch.load(destination,map_location='cpu',weights_only=False)
        metrics={key:float(value) for key,value in saved['loss_info'].items() if key in ('val_accuracy','test_accuracy')}
        digest=hashlib.sha256(destination.read_bytes()).hexdigest()
        (folder/f'study_epoch_{args.epochs}_manifest.json').write_text(json.dumps(dict(dataset=dataset,model='CNN_'+dataset,
            sha256=digest,metrics=metrics,
            epoch=args.epochs,source=str(source) if source.exists() else None,seed=args.seed,device=str(device),
            batch=args.batch,objective='cross_entropy_and_existing_scale_regularizer',
            note='Continues historical checkpoint; original training seed and split are not known.'),indent=2))
        print(f'Ready: {destination}',flush=True)

if __name__=='__main__':main()
