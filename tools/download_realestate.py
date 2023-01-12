import youtube_dl
# from typing import Dict, List
from tqdm import tqdm
import os
from glob import glob

class RealEstate10K_Downloader:
    def __init__(self, data_dir: str, output_dir: str) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir

    def download(self):
        play_list_train, play_list_test = self._get_play_list()
        
        '''
        download train and test
        '''
        t1 = tqdm(play_list_train)
        for pl in t1:  
            name, url = pl.split('+')
            t1.set_description(url)
            ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(self.output_dir,'train', "{}.%(ext)s".format(name)),      
                   }      
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        
        t2 = tqdm(play_list_test)
        for pl in t2:  
            name, url = pl.split('+')
            t1.set_description(name)
            ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(self.output_dir,'test', "{}.%(ext)s".format(name)),      
                   }      
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        
    
    def _get_play_list(self):
        '''
        get train and test files
        '''
        def _get_list(files):
            out_list = []
            for f in tqdm(files):
                with open(f, 'r') as record:
                    video_url = str(record.readline().rstrip())
                    out_list.append(os.path.splitext(os.path.basename(f))[0]+'+'+video_url)
            return out_list
                    
        record_files_train = glob(os.path.join(self.data_dir, 'train', '*.txt'))
        play_list_train = _get_list(record_files_train)
        record_files_test = glob(os.path.join(self.data_dir, 'test', '*.txt'))
        play_list_test = _get_list(record_files_test)
        return play_list_train, play_list_test


if __name__ == '__main__':
    RD = RealEstate10K_Downloader(data_dir='/home/xxy/Documents/data/RealEstate10K', output_dir='/home/xxy/Documents/data/RealEstate10K/videos')
    RD.download()