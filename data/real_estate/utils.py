
from .realestate10k import RealEstate10KRelationships
def get_dataset(mode, args, **kwargs):
    dataset = RealEstate10KRelationships(mode, args)
    print('Load {} data, {} pairs in total'.format(mode, len(dataset)))
    return dataset