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
os.environ['DEBUG'] = '0'
from pprint import pprint
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import h5py
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
EPS = 1e-2

class Hook():
    def __init__(self, d=None) -> None:
        self._data = d

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, d):
        self._data = d

class Warp():
    def __init__(self) -> None:
        pass
    
    def plot_relationship(self, filename, filename2 = None, pts3D = None, num_relation = 5):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        nw, nh = 1000, int(h*1000/w)
        img = cv2.resize(img, (nw, nh))
        img2 = np.array(Image.open(filename2))
        _, ext = os.path.splitext(filename)
        rel_path = filename.replace(ext, '.h5')
        if not os.path.exists(rel_path):
            raise 'No relationship file'
        f = h5py.File(rel_path, 'r')
        bbox_f = np.array(f['bbox'])
        # bbox[:,0], bbox[:,2] = (bbox[:,0]*w), (bbox[:,2]*w)
        # bbox[:,1], bbox[:,3] = (bbox[:,1]*h), (bbox[:,3]*h)
        bbox = bbox_f.copy().astype(np.int16)
        # rel_all_scores = np.array(f['rel_all_scores'])
        rel_pairs = np.array(f['rel_pairs'])
        top_pairs = rel_pairs[:num_relation]
        top_boxes_subj = bbox[top_pairs[:,0]]
        top_boxes_obj = bbox[top_pairs[:,1]]
        img = self._plot_single_img(img, top_boxes_subj, top_boxes_obj)
        '''
        transformed
        '''
        pts3D = pts3D.squeeze().detach().cpu().numpy()
        pts3D[:,1] = (- pts3D[:,1])*128+128
        pts3D[:,0] = (- pts3D[:,0])*128+128
        pts3D[pts3D<0] = 0
        pts3D[pts3D>255] = 255
        pts3D = pts3D.astype(np.int16)

        img2 = cv2.resize(img2, [256,256])
        bbox_t = bbox_f.copy()
        bbox_t[:, 0::2] = bbox_t[:, 0::2]*256/nw
        bbox_t[:, 1::2] = bbox_t[:, 1::2]*256/nh
        bbox_t = bbox_t.astype(np.int16)
        new_bbox = []
        for bt in bbox_t:
            x1y1 = pts3D[bt[1]*256+bt[0], :2]
            x2y2 = pts3D[bt[3]*256+bt[2], :2]
            new_bbox.append(np.hstack([x1y1,x2y2]))
        new_bbox = np.vstack(new_bbox)
        top_boxes_subj = new_bbox[top_pairs[:,0]]
        top_boxes_obj = new_bbox[top_pairs[:,1]]
        img2 = self._plot_single_img(img2, top_boxes_subj, top_boxes_obj)

        
        img3 = cv2.imread(filename2)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        h,w = img3.shape[:2]
        nw, nh = 1000, int(h*1000/w)
        img3 = cv2.resize(img3, (nw, nh))
        _, ext = os.path.splitext(filename2)
        rel_path = filename2.replace(ext, '.h5')
        if not os.path.exists(rel_path):
            raise 'No relationship file'
        f = h5py.File(rel_path, 'r')
        bbox_f = np.array(f['bbox'])
        # bbox[:,0], bbox[:,2] = (bbox[:,0]*w), (bbox[:,2]*w)
        # bbox[:,1], bbox[:,3] = (bbox[:,1]*h), (bbox[:,3]*h)
        bbox = bbox_f.copy().astype(np.int16)
        # rel_all_scores = np.array(f['rel_all_scores'])
        rel_pairs = np.array(f['rel_pairs'])
        top_pairs = rel_pairs[:num_relation]
        top_boxes_subj = bbox[top_pairs[:,0]]
        top_boxes_obj = bbox[top_pairs[:,1]]
        img3 = self._plot_single_img(img3, top_boxes_subj, top_boxes_obj)
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.subplot(1,3,2)
        plt.imshow(cv2.resize(img2, (w,h)))
        plt.subplot(1,3,3)
        plt.imshow(img3)
        # plt.show()

        return img, img2
        
        
        
    
    def _plot_single_img(self, img, top_boxes_subj,top_boxes_obj):
        for sb, ob in zip(top_boxes_subj, top_boxes_obj):
            img = cv2.rectangle(img, sb[:2], sb[2:], color = [255,0,0])
            img = cv2.rectangle(img, ob[:2], ob[2:], color = [255,0,0])
            img = cv2.line(img, (sb[2], sb[3]), (ob[2], ob[3]), color=[0,0,255])
        return img

        

    def mapping(self, filename, depth, K, P1, P2, filename2 = None):
        offset = np.array(
            [[2, 0, -1], [0, -2, 1], [0, 0, -1]],  # Flip ys to match habitat
            dtype=np.float32,
        ) 
        K_identity = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        invK_indentity = np.linalg.inv(K_identity)
        K = np.matmul(offset, K)
        RT_cam1 = np.matmul(K, P1)
        RT_cam1= np.vstack((RT_cam1, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
        RT_cam1[3, 3] = 1
        RTinv_cam1 = np.linalg.inv(RT_cam1)

        RT_cam2 = np.matmul(K, P2)
        RT_cam2= np.vstack((RT_cam2, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
        RT_cam2[3, 3] = 1
        RTinv_cam2 = np.linalg.inv(RT_cam2)
        def _to_tensor(kwargs):
            rs = []
            for k in kwargs:
                rs.append(torch.Tensor(k).unsqueeze(0).to(device))
            return rs
        K_identity, invK_indentity, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2 = _to_tensor([K_identity, invK_indentity, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2])
        gen_img, pointclouds = self.mapping_from_synsin(filename, depth, K_identity, invK_indentity, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2)
        ori_img_plot = self.plot_relationship(filename, pts3D=pointclouds, filename2=filename2)
        return gen_img


    def mapping_from_synsin(self, filename, depth, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2):
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
        # vs = vars(opts)
        # vs['pp_pixel'] = 1
        # vs['radius'] = 1
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
        transform_model = model_to_test.model.module.pts_transformer.forward_justpts
        img = Image.open(filename)
        input= transform(img).unsqueeze(0).to(device)
        hook = Hook()
        gen_img = transform_model(input, depth, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, hook)
        return gen_img, hook.data

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
        
        depth_save = depth.detach().cpu().squeeze().numpy()
        cv2.imwrite('./depth.png', depth_save)
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
        # prediction = (torch.rand(prediction.shape)).to(device)
        prediction = prediction * (opts.max_z - opts.min_z) + opts.min_z
        return prediction


    def get_depth_from_midas(self, filename):
        import torchvision.transforms as transforms
        model_type = "MiDaS_small"
        # model_type = 'DPT_Large'
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        resize_transform = transforms.Resize((256,256))

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = resize_transform(transform(img)).to(device)
        prediction = midas(input_batch)
        # prediction = (prediction/prediction.max())
        # prediction = 1/(nn.Sigmoid()(prediction)*100+0.01)
        prediction = 1/prediction  
        prediction = 100 *(prediction-prediction.min())/(prediction.max()-prediction.min())+0.01
        return prediction.unsqueeze(0)
        

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

        dMat = np.matmul(mat2, np.linalg.inv(mat1))
        return dMat


'''
map one single pixel to a transformed view
image coordinate: (u,v), K1, P1 -> camera coordinate -> world coordinare -> K2, P2, new viewpoint
'''
class MyMap():
    def __init__(self) -> None:
        pass
    
    def map(self, img1, img2, depth, K , T):
        img1 = np.array(Image.open(img1))
        img2 = np.array(Image.open(img2))
        # img1 = cv2.resize(img1, [256, 256])
        s, t = 1, 0
        depth = s*depth + t
        '''step1: pixel to relative'''
        w, h = img1.shape[:2]
        u, v ,w = 100, 100, 1
        ur = u/w
        vr = v/h
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
        x_prime = (ur-cx)/fx
        y_prime = (vr-cy)/fy
        z_prime = depth[u,v]
        x_tilde = x_prime * z_prime
        y_tilde = y_prime * z_prime
        z_tilde = z_prime
        camera_point = np.array([x_tilde, y_tilde, z_tilde, 1])
        mapped_point = K@T[:3,:]@camera_point
        print(mapped_point)


def main():
    reader = ImageReader_RealEstate10K('/home/xxy/Documents/data/RealEstate10K/test/0a9f2831a3e73de8.txt',
                                       '/home/xxy/Downloads/tmp_data/0a9f2831a3e73de8')
    GD = GetDepth('synsin')
    
    img_name, K, P1 = reader.get_one_frame(50)
    
    name2, _, P2 = reader.get_one_frame(20)
    # T = reader.get_relative_parameters(P1, P2)
    print(img_name, name2)
    depth = GD.get_depth(img_name)
    # M = MyMap()
    # M.map(img_name, name2, depth.squeeze().detach().cpu().numpy(), K, T)
    W = Warp()
    image_new = W.mapping(img_name, depth, K, P1, P2, filename2=name2)
    image_new = image_new.squeeze().permute(1,2,0).detach().cpu().numpy()
    
    img = np.array(Image.open(img_name))
    image_new = cv2.resize(image_new,img.shape[:2][::-1])
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img)
    img2 = np.array(Image.open(name2))
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.subplot(1,3,3)
    plt.imshow(image_new)
    plt.show()

if __name__=='__main__':
    main()