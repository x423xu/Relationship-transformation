from .model import PLPredictionModule

def make_model(args):
    if args.mode == 'train':
        return PLPredictionModule(args)
    if args.mode == 'test':
        return PLPredictionModule(args).load_from_checkpoint(
            checkpoint_path=args.test_model_dir,
        )