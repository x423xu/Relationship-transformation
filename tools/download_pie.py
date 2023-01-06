from urllib.request import urlopen
import re, os
import wget
url = 'https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/'
save_path = "/home/xxy/Documents/data/pie"

urlpath = urlopen(url)
string = urlpath.read().decode('utf-8')
pattern = re.compile('set0[0-6]/')
filelist = set(pattern.findall(string))
print(filelist)
url_list = [os.path.join(url, f) for f in filelist]
print(url_list)

for ul in url_list:
    video_path = urlopen(ul)
    string = video_path.read().decode('utf-8')
    print(string)
    pattern = re.compile('video_[0-9]*.mp4')
    video_list = set(pattern.findall(string))
    for vl in video_list:
        out_path = os.path.join(save_path, ul.split('/')[-2])
        print(out_path, vl, '\n')
        wget.download(os.path.join(ul, vl), out = out_path)