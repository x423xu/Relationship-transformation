import importlib
dataloader_zoo={
    'real_estate':'RealEstateRelationshipDataModule'
}

def make_data(args):
    loader = importlib.import_module("data.{}.loader".format(args.dataset))
    loader = getattr(loader, dataloader_zoo[args.dataset])
    return loader(args)