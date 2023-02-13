'''
select train validation test image pairs
train:val:test = 16k:4k:2k
'''


import os
from glob import glob
import numpy as np
from tqdm import tqdm


class TrainValSplit():
    def __init__(self, image_pairs_dir):
        self.ratio = {
            'train':16000, 
            'val':4000, 
            'test':2000
            }
        self.image_pairs_dir = image_pairs_dir

    def get_split(self, mode='test'):
        path = os.path.join(self.image_pairs_dir, mode)
        sequences = glob(os.path.join(path, '*.txt'))
        pairs = []
        for s in tqdm(sequences):
            sequence_name = os.path.basename(s)
            with open(s, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    continue
                lines = [l.strip() for l in lines]
                sequence_list = np.repeat([sequence_name], len(lines))
                lines = np.vstack([sequence_list, lines]).transpose()
                pairs.append(lines)
        pairs = np.vstack(pairs)
        length = pairs.shape[0]
        index = np.arange(length)
        np.random.shuffle(index)
        if mode == 'train':
            return {
                'train': pairs[index[:self.ratio['train']]],
                'val': pairs[index[self.ratio['train']:self.ratio['train']+self.ratio['val']]]
                    }
        elif mode == 'test':
            split_pairs = pairs[index[:self.ratio[mode]]]
            return {'test': split_pairs}
        else:
            raise "UNSEEN MODE ERROR"

if __name__ == '__main__':
    T = TrainValSplit(image_pairs_dir = '/home/xxy/Documents/data/RealEstate10K/pairs')
    train_val_pairs = T.get_split('train')
    test_pairs = T.get_split('test')
    pairs = {}
    pairs.update(train_val_pairs)
    pairs.update(test_pairs)
    np.save('train_val_test.npy', pairs, allow_pickle=True)
    