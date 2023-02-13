import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import h5py

'''
Dataset for RealEstate10K relationships
return: image, relationships, bounding boxes, Intrinsic params, camera pose
'''
class RealEstate10KRelationships(data.Dataset):
    def __init__(self, mode, args) -> None:
        self.args = args
        self.mode = mode
        self.transform  = Compose(
            [
                Resize((args.W, args.W)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        pairs = np.load(args.train_val_test_dir, allow_pickle=True).item()
        self.train_val_test_pairs = self._remove_non_exist(pairs)
        self.offset = np.array(
            [[2, 0, -1], [0, -2, 1], [0, 0, -1]],  # Flip ys to match habitat
            dtype=np.float32,
        )
        self.K = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _remove_non_exist(self, pairs):
        new_train_val_test_pairs = {}
        for mode, value in pairs.items():
            new_train_val_test_pairs[mode] = []
            for v in value:
                camera_txt = v[0]
                seq_name = camera_txt.rstrip('.txt')
                [t1,t2] = v[1].split()
                path1 = os.path.join(self.args.frames_dir, mode, seq_name, t1+'.png')
                path2 = os.path.join(self.args.frames_dir, mode, seq_name, t2+'.png')
                if not os.path.exists(path1) or not os.path.exists(path2):
                    # print('{} {} {}, {}'.format(mode, seq_name, t1, t2))
                    pass
                else:
                    new_train_val_test_pairs[mode].append(v)
        return new_train_val_test_pairs

    
    def load_img(self, seq_name, timestamp):
        png_name = os.path.join(self.args.frames_dir, self.mode, seq_name, timestamp+'.png')
        img = Image.open(png_name)
        img = self.transform(img)
        return img
    
    def load_rel(self,seq_name, timestamp):
        rel_name = os.path.join(self.args.frames_dir, self.mode, seq_name, timestamp+'.h5')
        with h5py.File(rel_name, 'r') as f:
            rel_features = np.array(f["rel_features"])
            bbox = np.array(f["bbox"])
            idx_pairs = np.array(f["idx_pairs"])
            labels = np.array(f["labels"])
        return {
            'rel_features':rel_features,
            'bbox': bbox,
            'idx_pairs': idx_pairs,
            'labels': labels
        }
    
    '''
    given camera text file, return intrinsic, camera pose
    '''
    def get_camera_params(self, camera_txt, timestamp):
        if self.mode =='train' or self.mode == 'val':
            path = os.path.join(self.args.cameras_dir, 'train', camera_txt)
        if self.mode == 'test':
            path = os.path.join(self.args.cameras_dir, 'test', camera_txt)
        with open(path) as f:
            lines = f.readlines()
            for l in lines:
                anno = l.strip().split(' ')
                if anno[0] == timestamp:
                    camera_pose = [float(x) for x in anno[-12:]]
                    intrisinc_parameters = [float(x) for x in anno[1:7]]
                    K = np.array(
                                    [
                                        [intrisinc_parameters[0], 0, intrisinc_parameters[2]],
                                        [0, intrisinc_parameters[1], intrisinc_parameters[3]],
                                        [0, 0, 1],
                                    ],
                                    dtype=np.float32,
                                )
                    P = np.array(camera_pose).reshape(3, 4)
                    return K,P
            raise "{}: {} not found in {}".format(self.mode, timestamp, camera_txt)

    
    def __getitem__(self, index):
        pair = self.train_val_test_pairs[self.mode][index]
        camera_txt = pair[0]
        seq_name = camera_txt.rstrip('.txt')
        [t1, t2] = pair[1].split()
        
        img1 = self.load_img(seq_name, t1)
        img2 = self.load_img(seq_name, t2)
        # R1, bbox1 = self.get_rel(seq_name, t1)
        # R2, bbox2 = self.get_rel(seq_name, t2)
        K, P1 = self.get_camera_params(camera_txt, t1)
        _, P2 = self.get_camera_params(camera_txt, t2)
        # R1 = torch.zeros([1, 6032, 128])
        # B1 = torch.zeros([1, 6032, 8])
        # R2 = torch.zeros([1, 6032, 128])
        # B2 = torch.zeros([1, 6032, 8])
        R1 = self.load_rel(seq_name, t1)
        R2 = self.load_rel(seq_name, t2)

        #modify to be compatible with habitat
        K = np.matmul(self.offset, K)
        P1 = np.matmul(K, P1)
        P1 = np.vstack((P1, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
        P1[3, 3] = 1

        P2 = np.matmul(K, P2)
        P2 = np.vstack((P2, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
        P2[3, 3] = 1
        P1inv = np.linalg.inv(P1)
        P2inv = np.linalg.inv(P2)
        return {
            'images': [img1, img2],
            'K': self.K,
            'P': [P1, P2],
            'Pinv': [P1inv, P2inv],
            'R': [R1, R2],
        }
    
    def __len__(self):
        return len(self.train_val_test_pairs[self.mode])
    

if __name__=='__main__':
    import sys, os
    from tqdm import tqdm
    sys.path.append(os.getcwd())
    from configs import args
    for m in ['train', 'val', 'test']:
        dataset = RealEstate10KRelationships(m,args)
        for d in tqdm(dataset):
            pass