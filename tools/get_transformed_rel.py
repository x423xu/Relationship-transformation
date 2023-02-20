"""
get transformed relationships from given relationships and transformation
"""
import os, sys

sys.path.append(os.getcwd())
sys.path.append("/home/xxy/Documents/github/Relationship-transformation/lib/synsin")
from model import make_model
from configs import args
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import numpy as np
from glob import glob
import h5py
from PIL import Image
from tqdm import tqdm
import quaternion
import cv2


def load_rel(img_name):
    rel_name = img_name.replace(".png", ".h5")
    """
    laod relationship
    """
    if not os.path.exists(rel_name):
        raise "no such file"
    with h5py.File(rel_name, "r") as f:
        rel_features = np.array(f["rel_features"])
        bbox = np.array(f["bbox"])
        idx_pairs = np.array(f["idx_pairs"])
        labels = np.array(f["labels"])
    if labels.shape[0] > 100:
        raise "> 100"
    labels_100 = np.zeros(
        [
            100,
        ]
    )
    labels_100[: labels.shape[0]] = labels

    # bbox[:, ::2] = bbox[:, ::2] / img_shape[0]
    # bbox[:, 1::2] = bbox[:, 1::2] / img_shape[1]
    return {
        "rel_features": torch.tensor(rel_features).unsqueeze(0),
        "bbox": torch.tensor(bbox).unsqueeze(0),
        "idx_pairs": torch.tensor(idx_pairs).unsqueeze(0),
        "labels": torch.tensor(labels_100).unsqueeze(0),
    }


def load_img(png_name):
    img = Image.open(png_name)
    shape = [img.width, img.height]
    return img, shape


def random_kp_rollout():
    K = torch.eye(4).unsqueeze(0)
    invK = torch.eye(4).unsqueeze(0)

    range = np.random.choice([0.1, -0.1], p=[0.5, 0.5])
    theta = range * np.random.rand()
    phi = range * np.random.rand()
    gamma = range * np.random.rand()
    range2 = np.random.choice([0.15, -0.15], p=[0.5, 0.5])
    tx = range2 * np.random.rand()
    ty = range2 * np.random.rand()
    tz = range2 * np.random.rand()

    RT = torch.eye(4).unsqueeze(0)
    # Set up rotation
    RT[0, 0:3, 0:3] = torch.Tensor(
        quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector([phi, theta, gamma])
        )
    )
    # Set up translation
    RT[0, 0:3, 3] = torch.Tensor([tx, ty, tz])
    return K, invK, RT


"""
given an image list, return transformed relationship
"""


def get_batch_imgs(img, transform, device="cuda"):
    batch = {}
    rel_orig = load_rel(img)
    img, shape = load_img(img)
    img = transform(img).unsqueeze(0).cuda()
    rel_orig["bbox"][:, ::2] = rel_orig["bbox"][:, ::2] / shape[0]
    rel_orig["bbox"][:, 1::2] = rel_orig["bbox"][:, 1::2] / shape[1]
    """
    camera pose
    """
    K, invK, RT = random_kp_rollout()
    identity = torch.eye(4).unsqueeze(0)
    batch.update(
        {
            "images": [
                img,
            ],
            "K": K.cuda(),
            "Kinv": invK.cuda(),
            "P": [identity.cuda(), RT.cuda()],
            "Pinv": [identity.cuda(), identity.cuda()],
            "rel_features": [
                rel_orig["rel_features"].cuda(),
            ],
            "bbox": [
                rel_orig["bbox"].cuda(),
            ],
        }
    )
    return batch

'''
get new bbox from pts3d
'''
def generate_new_bbox(il, bbox, pts3d, argsW):
    import matplotlib.pyplot as plt
    img = Image.open(il)
    w,h = img.size
    bbox[0, :, ::2] *= w
    bbox[0, :, 1::2] *= h
    bbox = bbox[0, :5, :].int().cpu().numpy()
    img = np.array(img)
    for box in bbox:
        sub_box = box[:4]
        obj_box = box[4:]
        img = cv2.rectangle(img, sub_box[:2], sub_box[2:], color = [255,0,0])
        img = cv2.rectangle(img, obj_box[:2], obj_box[2:], color = [255,0,0])
        img = cv2.line(img, (sub_box[2], sub_box[3]), (obj_box[2], obj_box[3]), color=[0,0,255])
    plt.imshow(img)
    plt.show()



def get_rel_transformed_from_list(img_list, W=256):
    args.mode = "test"
    model = make_model(args)
    trans_model = model.model.cuda()
    transform = Compose(
        [
            Resize((W, W)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    with torch.no_grad():
        for il in tqdm(img_list):
            # if os.path.exists(il.replace(".png", "_trans.h5")):
            #     continue
            
            batch = get_batch_imgs(il, transform)
            R_tilde, pts3d = trans_model(batch)
            generate_new_bbox(il, batch['bbox'][0], pts3d, args.W)
            rel_features = R_tilde.cpu().numpy()
            bbox = batch["bbox"][0].cpu().numpy()
            rel_name = il.replace(".png", "_trans.h5")
            im_rel_h5 = h5py.File(
                rel_name,
                "w",
            )
            im_rel_h5.create_dataset("rel_features", data=rel_features)
            im_rel_h5.create_dataset("bbox", data=bbox)
            im_rel_h5.close()


def get_transformed_rel(dataset):
    if dataset == "pie":
        frame_path = "/home/xxy/Documents/github/PIE/images"
        sets = os.listdir(frame_path)
        img_list = []
        for s in sets:
            seqs = os.listdir(os.path.join(frame_path, s))
            for seq in seqs:
                png_names = glob(os.path.join(frame_path, s, seq, "*.png"))
                img_list.extend(png_names)
        get_rel_transformed_from_list(img_list)
    else:
        print("No implementation")
        return


if __name__ == "__main__":
    get_transformed_rel(dataset="pie")
