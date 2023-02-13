from configs import args
import pytorch_lightning as pl
from data import make_data
from model import make_model
from utils.pl_config import set_arguments_pl

'''

'''

def main():
    dataloader = make_data(args)
    model = make_model(args)

    pl_args = set_arguments_pl(args)
    trainer = pl.Trainer(**pl_args)
    if args.mode == 'train':
        trainer.fit(model, dataloader)
    if args.mode == 'test':
        trainer.test(ckpt_path="best")

if __name__ == '__main__':
    main()