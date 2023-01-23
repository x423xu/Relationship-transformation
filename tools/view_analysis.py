
'''
Analyze the angle changes in each sequence. Keep frames with angle changes greater than $threshold$. Remove trivial angle changes.
The best case is the camera face to the same center, while the position and angle changes.
'''
import numpy as np
from math import sqrt
import cv2
import os
opj = os.path.join
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, Process, Queue
import gc




# def get_relative_change(mat1, mat2):
#     rotation_matrix1 = mat1[:3, :3]
#     translation_vector1 = mat1[:3, 3]
#     rotation_matrix2 = mat2[:3, :3]
#     translation_vector2 = mat2[:3, 3]
    

#     relative_rotation_matrix = np.matmul( np.linalg.inv(rotation_matrix1), rotation_matrix2)
    
#     relative_translation_vector = translation_vector2 - np.matmul(relative_rotation_matrix, translation_vector1)
#     angle, _ = cv2.Rodrigues(relative_rotation_matrix)
#     relative_translation = relative_translation_vector
#     relative_angle = np.degrees(angle)
#     # relative_angle = angle
#     return relative_angle, relative_translation

# sample_dir = '/home/xxy/Documents/data/RealEstate10K/test/9981436f1448c131.txt'
# sample_video_dir = '/home/xxy/Documents/data/RealEstate10K/videos/test/9981436f1448c131'

# frames = os.listdir(sample_video_dir)
# annotaions = []
# with open(sample_dir) as f:
#     lines = f.readlines()
#     for idx, line in enumerate(lines):
#         if idx == 0:
#             continue
#         else:
#             annotaions.append(line.strip().split(' '))

# end_frame_idx = 30
# annotations = sorted(annotaions, key=lambda x: x[0])
# start_frame = annotaions[0][0]
# start_camera_pose = annotaions[0][-12:]

# end_frame = annotaions[end_frame_idx][0]
# end_camera_pose = annotaions[end_frame_idx][-12:]

# print(start_frame, end_frame)


# start_P = np.array([float(i) for i in start_camera_pose]).reshape([3,4])
# end_P = np.array([float(i) for i in end_camera_pose]).reshape([3,4])

# print(get_relative_change(start_P, end_P))



class DataAnalysis():
    def __init__(self, mode='test', num_worksers = -1) -> None:
        self.mode = mode
        self.annotation_dir = '/home/xxy/Documents/data/RealEstate10K/'
        self.video_dir = '/home/xxy/Documents/data/RealEstate10K/videos'
        self.ANGLE_THRESH = 5
        self.TRANS_THRESH = 0.15
        self.FRAME_RANGE = 30
        self.num_workers = num_worksers
        self.len = 0

    def _get_deltas(self, mat1, mat2):
        mat1 = np.vstack((mat1, np.array([0, 0, 0, 1])))
        mat2 = np.vstack((mat2, np.array([0, 0, 0, 1])))

        dMat = np.matmul(np.linalg.inv(mat1), mat2)
        dtrans = dMat[0:3, 3] ** 2
        dtrans = sqrt(dtrans.sum())

        origVec = np.array([[0], [0], [1]])
        rotVec = np.matmul(dMat[0:3, 0:3], origVec)
        arccos = (rotVec * origVec).sum() / sqrt((rotVec ** 2).sum())
        dAngle = np.arccos(arccos) * 180.0 / np.pi

        return dAngle, dtrans

    def _parse_txt(self):
        if os.path.exists('parse_{}.npy'.format(self.mode)):
            return np.load('parse_{}.npy'.format(self.mode), allow_pickle=True)
        annotation_files = glob(opj(self.annotation_dir, self.mode, '*.txt'))
        annotations = {}
        for af in tqdm(annotation_files):
            with open(af,'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if idx == 0:
                        seq_name = os.path.basename(af).rstrip('.txt')
                        annotations[seq_name] = []
                        continue
                    else:
                        anno = line.strip().split(' ')
                        # anno[1:] = [float(l) for l in anno[1:]]
                        annotations[seq_name].append(anno)
        np.save('parse_{}.npy'.format(self.mode), annotations)
        return annotations
    
    def _create_image_pairs(self, seq_annotation, skip = 10):
        pass
        '''
        1. within 30 frames
        2. rotation greater than thresh
        3. translation greater than thresh
        input: seq_annotation: timestamp, intrinsic parameters, camera poses
        '''
        image_pairs_mask = np.zeros([len(seq_annotation), len(seq_annotation)], dtype=bool)
        for n in range(0, len(seq_annotation), skip):
            sa = seq_annotation[n]
            top = min(n+self.FRAME_RANGE, len(seq_annotation))
            bottom = max(0,n-self.FRAME_RANGE)
            adjacent_frames_30  = seq_annotation[bottom: top]
            start_P = np.array([float(i) for i in sa[-12:]]).reshape([3,4])
            mask = []
            for af in adjacent_frames_30:
                end_P = np.array([float(i) for i in af[-12:]]).reshape([3,4])
                dr, dt = self._get_deltas(start_P, end_P)
                mask.append((dr > self.ANGLE_THRESH) | (dt > self.TRANS_THRESH))
            image_pairs_mask[n][bottom:top] = mask
            # print('{} get {} pairs'.format(sa[0], np.array(mask).sum()))
        return image_pairs_mask
    
    def process(self, k, n_total, value):
        mask = self._create_image_pairs(value)
        frame_collected = []
        write_pair = []
        for n, frame in enumerate(value):
            if np.array(mask[n]).sum() == 0:
                continue
            valid_frames = np.array(value)[mask[n]]
            target_img = frame[0]
            refer_imgs = list(valid_frames[:, 0])
            write_pair.append([[target_img]*len(refer_imgs), refer_imgs]) 
            if target_img not in frame_collected:
                frame_collected.append(target_img)
            for ri in refer_imgs:
                if ri not in frame_collected:
                    frame_collected.append(ri)
        
        num_pairs = np.array(mask).sum()
        num_frames = len(frame_collected)
        print('{} {}/{}, {} pairs, {} frames'.format(k, n_total, self.len, np.array(mask).sum(), num_frames))
        if len(write_pair)>0:
            if not os.path.exists(opj(self.annotation_dir, 'pairs',self.mode)):
                os.makedirs(opj(self.annotation_dir, 'pairs',self.mode))
            with open(opj(self.annotation_dir, 'pairs',self.mode, k+'.txt'), 'w') as f:
                for wp in write_pair:
                    for (ti, ri) in zip(*wp):
                        f.write('{} {}\n'.format(ti, ri))
        return num_pairs, num_frames
    
    def wrap_process(self, list_args):
        return self.process(*list_args)
    
    def worker(self, list_data, queue):
        for ld in list_data:
            (k, n, v)  = ld
            result = self.process(k, n, v)
            queue.put(result)
        
            
    
    def get_valid_image_pairs(self):
        annotations = self._parse_txt()
        total_frames = 0
        keys = list(annotations.keys())
        values = list(annotations.values())
        self.len = len(keys)
        if self.num_workers == -1:
            for n, (k,v) in enumerate(zip(keys, values)):
                num_pairs, num_frames = self.process(k, n, v)
                # print('seq {}: {}/{}, {} pairs, {} frames'.format(k, n, len(keys), num_pairs, num_frames))
                total_frames += num_frames
        else:
            procs = []
            data_queue = Queue()
            NTASKS = self.num_workers
            list_data = [(k, n, v) for (k,n,v) in zip(keys, np.arange(len(keys)), values)]
            split_data_list = np.array_split(np.array(list_data), NTASKS)
            for n in range(NTASKS):
                proc = Process(target = self.worker, args = (split_data_list[n], data_queue), name = 'process_{}'.format(n))
                proc.start()
                procs.append(proc)
            try:
                for p in procs:
                    p.join()
            except KeyboardInterrupt:
                for p in procs:
                    p.terminate()
                    p.join()
            
            while not data_queue.empty():
                result = data_queue.get()
                total_frames += result[1]


        print('total frames to be downloaded {}'.format(total_frames))

D = DataAnalysis(mode = 'train', num_worksers = 6)
D.get_valid_image_pairs()
