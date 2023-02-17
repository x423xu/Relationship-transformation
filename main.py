'''
use synsin model
'''
import sys
sys.path.append('lib/synsin')

from configs import args
import pytorch_lightning as pl
from data import make_data
from model import make_model
from utils.pl_config import set_arguments_pl


'''

'''
'''when in ddp training, set all tasks have same limited train and val batch size'''
def set_limited_train_val(args):
    d = {}
    if args.mode == 'train':
        if args.accelerator == 'ddp':
            if args.train_size ==8 and args.val_size==8:
                d.update({'limit_val_batches':100})
            if args.train_size ==4 and args.val_size==4:
                d.update({'limit_val_batches':200})
    return d
def main():
    dataloader = make_data(args)
    model = make_model(args)
    val_size = set_limited_train_val(args)
    pl_args = set_arguments_pl(args)
    pl_args.update(**val_size) 
    trainer = pl.Trainer(**pl_args) 
    if args.mode == 'train':      
        trainer.fit(model, dataloader)
    if args.mode == 'test':
        trainer.test(model, dataloaders=dataloader, ckpt_path=args.test_model_dir)
        

if __name__ == '__main__':
    main()