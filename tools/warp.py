# import os
# import numpy as np
# import cv2

# sample_dir = '/home/xxy/Documents/data/RealEstate10K/test/0a5eeb4466dd19bb.txt'
# sample_video_dir = '/home/xxy/Documents/data/RealEstate10K/videos/test/0a5eeb4466dd19bb'

# frames = sorted(os.listdir(sample_video_dir))
# annotaions = []
# with open(sample_dir) as f:
#     lines = f.readlines()
#     for idx, line in enumerate(lines):
#         if idx == 0:
#             continue
#         else:
#             annotaions.append(line.strip().split(' '))


# def get_k_p(ind):
#     start_frame = annotaions[ind][0]
#     start_camera_pose = [float(x) for x in annotaions[ind][-12:]]
#     start_intrisinc_parameters = [float(x) for x in annotaions[ind][1:7]]
#     startK = np.array(
#                     [
#                         [start_intrisinc_parameters[0], 0, start_intrisinc_parameters[2]],
#                         [0, start_intrisinc_parameters[1], start_intrisinc_parameters[3]],
#                         [0, 0, 1],
#                     ],
#                     dtype=np.float32,
#                 )
#     startP = np.array(start_camera_pose).reshape(3, 4)
#     return start_frame, startK, startP

# start_frame, startK, startP = get_k_p(0)
# end_frame, endK, endP = get_k_p(100)
# proj1 = startK @ startP
# proj2 = endK @ endP

# img1 = cv2.imread(os.path.join(sample_video_dir, start_frame+'.png'))
# img2 = cv2.imread(os.path.join(sample_video_dir, end_frame+'.png'))

# warped_frame = cv2.warpPerspective(img1, proj2, (img2.shape[1], img2.shape[0]))
# cat = np.hstack([warped_frame, img2])
# cv2.imshow('concated',cat)


# import cv2
# import torch
# import urllib.request

# import matplotlib.pyplot as plt

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform

# print('ok')
# img = cv2.imread('/home/xxy/Documents/data/RealEstate10K/videos/test/0bcde26e5a802638/70870800.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# input_batch = transform(img).to(device)

# with torch.no_grad():
#     prediction = midas(input_batch)

#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()

# output = prediction.cpu().numpy()
# plt.imshow(output)
# plt.show()

import os
from pprint import pprint
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
'''
Warp an image to another viewpoint (homogenous) image based on estimated depth value
steps:
    1. read image
    2. estimate depth
    3. mapping
    4. plotting
'''

'''
given image, depth map, intrinsics, camera poses -> mapping
'''
class Warp():
    def __init__(self, image_name, depth, intrinsics, camera_pose) -> None:
        self.img = Image.open(image_name)
        self.depth = depth
        self.intrinsics = intrinsics
        self.camera_pose = camera_pose

    def mapping(self):
        K = self.intrinsics
        R = self.camera_pose[:,:3]
        T = self.camera_pose[:,3]
        height, width = self.depth.shape[:2]
        points = np.array([[[x, y, self.depth[y, x]] for x in range(width)] for y in range(height)])
        points = points.reshape(-1, 3)
        points = cv2.reprojectImageTo3D(points, Q=None)
        points = points.reshape(height, width, 3)
        points = np.dot(np.dot(np.linalg.inv(K), R), points)
        points = points / points[2]
        points = np.dot(K, points)
        image_new = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                x_new = int(points[y, x, 0])
                y_new = int(points[y, x, 1])
                if 0 <= x_new < width and 0 <= y_new < height:
                    image_new[y_new, x_new] = self.img[y, x]
        return image_new


'''
Given the image return a depth map
'''
class GetDepth():
    def __init__(self, model_mode = 'synsin') -> None:
        self.model_mode = model_mode
        
    def get_depth(self, filename):
        # filename = '/home/xxy/Documents/github/synsin/demos/im.jpg'
        if self.model_mode == 'synsin':
            depth = self.get_depth_from_synsin(filename) 
        if self.model_mode == 'midas':
            depth = self.get_depth_from_midas(filename)
        
        depth = depth.detach().cpu().squeeze().numpy()
        cv2.imwrite('./depth.png', depth)
        # plt.imshow(depth)
        # plt.show()
        return depth
    
    def get_depth_from_synsin(self, filename):
        import sys
        sys.path.insert(1,'/home/xxy/Documents/github/synsin')
        from options.options import get_model
        torch.backends.cudnn.enabled = True
        import torch.nn as nn
        import torchvision.transforms as transforms
        from models.base_model import BaseModel
        from models.networks.sync_batchnorm import convert_model
        MODEL_PATH = '/home/xxy/Documents/github/synsin/modelcheckpoints/realestate/zbufferpts.pth'
        opts = torch.load(MODEL_PATH)['opts']
        opts.render_ids = [1]
        print(opts)
        model = get_model(opts)
        torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
        if 'sync' in opts.norm_G:
            model = convert_model(model)
            model = nn.DataParallel(model, torch_devices[0:1]).cuda()
        else:
            model = nn.DataParallel(model, torch_devices[0:1]).cuda()

        model_to_test = BaseModel(model, opts)
        model_to_test.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
        model_to_test.eval()

        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        depth_model = model_to_test.model.module.pts_regressor
        img = Image.open(filename)
        input_batch = transform(img).unsqueeze(0).to(device)
        prediction = depth_model(input_batch)
        prediction = nn.Sigmoid()(prediction)
        return prediction


    def get_depth_from_midas(self, filename):
        model_type = "MiDaS_small"
        # model_type = 'DPT_Large'
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)
        prediction = midas(input_batch)
        # prediction =  1/prediction
        return prediction
        
        


'''
Given the image path of the RealEstate10K
return: image, intrinsics, camera pose
'''
class ImageReader_RealEstate10K():
    def __init__(self, annotation_dir, video_dir) -> None:
        self.video_dir = video_dir
        annotaions = []
        with open(annotation_dir) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                else:
                    annotaions.append(line.strip().split(' '))
        self.annotations = annotaions

    def get_one_frame(self, ind):
        camera_pose = [float(x) for x in self.annotations[ind][-12:]]
        intrisinc_parameters = [float(x) for x in self.annotations[ind][1:7]]
        K = np.array(
                        [
                            [intrisinc_parameters[0], 0, intrisinc_parameters[2]],
                            [0, intrisinc_parameters[1], intrisinc_parameters[3]],
                            [0, 0, 1],
                        ],
                        dtype=np.float32,
                    )
        P = np.array(camera_pose).reshape(3, 4)
        filename = os.path.join(self.video_dir, self.annotations[ind][0]+'.png')
        return filename, K, P
    
    def get_relative_parameters(self, mat1, mat2):
        mat1 = np.vstack((mat1, np.array([0, 0, 0, 1])))
        mat2 = np.vstack((mat2, np.array([0, 0, 0, 1])))

        dMat = np.matmul(np.linalg.inv(mat1), mat2)
        return dMat

def main():
    reader = ImageReader_RealEstate10K('/home/xxy/Documents/data/RealEstate10K/test/0c0f298ace7c875b.txt',
                                       '/home/xxy/Documents/data/RealEstate10K/videos/test/0c0f298ace7c875b')
    GD = GetDepth('synsin')
    
    img_name, K, P1 = reader.get_one_frame(0)
    print(img_name)
    _, _, P2 = reader.get_one_frame(100)
    RelP = reader.get_relative_parameters(P1,P2)
    depth = GD.get_depth(img_name)
    W = Warp(img_name, depth, K, RelP)
    image_new = W.mapping()
    plt.show(image_new)
    plt.show()

if __name__=='__main__':
    main()