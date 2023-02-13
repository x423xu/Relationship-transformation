'''
Download: sequence_name, timestamp
process: N workers
'''
import argparse
import os
import numpy as np
from tqdm import tqdm
from skimage import io
from pytube import YouTube
from cv2 import resize as imresize
from multiprocessing import Pool, Process
import multiprocessing
from itertools import islice


def check_existence(dir, mode,  seq, timestamp):
    if 'videos' in dir:
        if mode=='val':
            mode='train'
    seq_dir = os.path.join(dir, mode, seq)
    if not os.path.exists(seq_dir):
        return False
    png_name = os.path.join(dir, mode, seq, timestamp+'.png')   
    if os.path.exists(png_name):
        return True
    else:
        return False

def check_dest():
    pass
# def download(url, timestamp, dest_dir, mode, seq):
#     if not os.path.exists(os.path.join(dest_dir, mode, seq)):
#         os.makedirs(os.path.join(dest_dir, mode, seq))
#     timestamp = int(timestamp/1000) 
#     str_hour = str(int(timestamp/3600000)).zfill(2)
#     str_min = str(int(int(timestamp%3600000)/60000)).zfill(2)
#     str_sec = str(int(int(int(timestamp%3600000)%60000)/1000)).zfill(2)
#     str_mill = str(int(int(int(timestamp%3600000)%60000)%1000)).zfill(3)
#     _str_timestamp = str_hour+":"+str_min+":"+str_sec+"."+str_mill
#     command = 'ffmpeg -loglevel error -y -ss '+_str_timestamp+' -i '+videoname+' -vframes 1 -f image2 '+output_root+'/'+seqname+'/'+str(data.list_list_timestamps[seq_id][idx])+'.png'
#     os.system(command)

def copy(videos_dir, mode, dest_dir, seq, t):
    dest_dir = os.path.join(dest_dir, mode, seq)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if mode == 'val':
        source_png = os.path.join(videos_dir, 'train', seq, t+'.png')
    else:
        source_png = os.path.join(videos_dir, mode, seq, t+'.png')
    if os.path.exists(os.path.join(dest_dir, t+'.png')):
        return
    os.system('cp {} {}'.format(source_png, os.path.join(dest_dir, t+'.png')))

def dict_worker(d, mode, cameras_dir, dest_dir):
    curr_proc = multiprocessing.current_process()
    for seq_name, timestamps in tqdm(d.items()):
        if mode == 'val':
            seq_dir = os.path.join(cameras_dir, 'train', seq_name+'.txt')
        else:
            seq_dir = os.path.join(cameras_dir, mode, seq_name+'.txt')
        with open(seq_dir, "r") as fyt:
            youtube_url = fyt.readline().strip() 
        try:
            yt = YouTube(youtube_url)  
            stream = yt.streams.filter(progressive=True).order_by('resolution').desc().first()
            stream.download(dest_dir,'{}_video.tmp'.format(curr_proc.name))
        except:
            continue
        list_str_timestamps = []
        for timestamp in timestamps:
            timestamp = int(int(timestamp)/1000) 
            str_hour = str(int(timestamp/3600000)).zfill(2)
            str_min = str(int(int(timestamp%3600000)/60000)).zfill(2)
            str_sec = str(int(int(int(timestamp%3600000)%60000)/1000)).zfill(2)
            str_mill = str(int(int(int(timestamp%3600000)%60000)%1000)).zfill(3)
            _str_timestamp = str_hour+":"+str_min+":"+str_sec+"."+str_mill
            list_str_timestamps.append(_str_timestamp)
        for ts, str_timestamp in zip(timestamps, list_str_timestamps):
            
            videoname = os.path.join(dest_dir, '{}_video.tmp'.format(curr_proc.name))
            output_root = os.path.join(dest_dir, mode)
            pngname = os.path.join(output_root,seq_name, ts+'.png')
            if os.path.exists(pngname):
                continue
            if not os.path.exists(os.path.join(output_root, seq_name)):
                os.makedirs(os.path.join(output_root, seq_name))
            command = 'ffmpeg -loglevel error -y -ss '+str_timestamp+' -i '+ videoname +' -vframes 1 -f image2 '+output_root+'/'+seq_name+'/'+ts+'.png'
            # print("current command is {}".format(command))
            os.system(command)
            
            image = io.imread(pngname)
            if int(image.shape[1]/2) < 500:
                print('{} shape is legal'.format(pngname))
                continue
            image = imresize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
            io.imsave(pngname, image)


def split_dict(d, n):
    it = iter(d)
    sd = []
    size = len(d.keys())//n
    for i in range(0, len(d.keys()), size):
        sd.append({k:d[k] for k in islice(it, size)})
    return sd
def group_by_seq(missed_frames, mode, cameras_dir, dest_dir):
    d = {}
    for m in tqdm(missed_frames):
        [seq, timestamp] = m
        if seq not in d.keys():
            d[seq]=  []
        else:
            d[seq].append(timestamp)
    dict_worker(d, mode, cameras_dir, dest_dir)
    # NTASKS = 16
    # sd = split_dict(d, NTASKS)
    # procs = []
    # for n in range(NTASKS+1):       
    #     proc = Process(target = dict_worker, args = (sd[n],mode, cameras_dir, dest_dir), name = 'process_{}'.format(n))
    #     proc.start()
    #     procs.append(proc)
    # try:
    #     for p in procs:
    #         p.join()
    # except KeyboardInterrupt:
    #     for p in procs:
    #         p.terminate()
    #         p.join()    
    

parser = argparse.ArgumentParser()
parser.add_argument('--cameras_dir', default='/home/xxy/Documents/data/RealEstate10K', type=str)
parser.add_argument('--videos_dir', default='/home/xxy/Documents/data/RealEstate10K/videos', type=str)
parser.add_argument('--dest_dir', default='/home/xxy/Documents/data/RealEstate10K/benchmark_frames', type=str)
# parser.add_argument('--videos_dir', default='/home/xxy/HDD', type=str)
parser.add_argument('--ntasks', default=4, type=int)
parser.add_argument('--cpus_per_task', default=4, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    pairs = np.load('train_val_test.npy', allow_pickle=True).item()
    for mode, value in pairs.items():
        missed_frames = []
        if mode == 'train' or mode == 'val':
            videos_dir = '/home/xxy/HDD'
        if mode == 'test':
            videos_dir = '/home/xxy/Documents/data/RealEstate10K/videos'
        for v in tqdm(value):
            seq_name, pair = v
            seq = seq_name.rstrip('.txt')
            [t1, t2] = pair.split()
            if not check_existence(args.dest_dir, mode, seq, t1):
                if not check_existence(videos_dir, mode, seq, t1):
                    missed_frames.append([seq, t1])
                else:
                    copy(videos_dir, mode, args.dest_dir, seq, t1)
            if not check_existence(args.dest_dir, mode, seq, t2):
                if not check_existence(videos_dir, mode, seq, t2):
                    missed_frames.append([seq, t2])
                else:
                    copy(videos_dir, mode, args.dest_dir, seq, t2)
            
            
        group_by_seq(missed_frames, mode, args.cameras_dir, args.dest_dir)