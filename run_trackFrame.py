import os

from options.test_options import TestOptions
from models import create_model
from util.util import tensor2labelim, tensor2confidencemap
from models.sne_model import SNE
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2


class dataset():
    def __init__(self):
        self.num_labels = 2

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.isTrain = False

    example_dataset = dataset()
    model = create_model(opt, example_dataset)
    model.setup(opt)
    model.eval()

    # if you want to use your own data, please modify rgb_image, depth_image, camParam and use_size correspondingly.
    rgb_image = cv2.cvtColor(cv2.imread(os.path.join('datasets', 'track','testing','image_2','frame_700.jpg')), cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join('output', 'oimage.png'), rgb_image)
    #depth_image = cv2.imread(os.path.join('datasets', 'track','testing', 'depth_u16','frame_1.png'), cv2.IMREAD_ANYDEPTH)
    
    # Try reading pfm depth directly
    fName = 'frame_700.pfm'
    depth_image = cv2.imread(os.path.join('b:\\','code','Midas','output',fName), cv2.IMREAD_UNCHANGED)
    
    # original depth written as a 16-bit .png looks good
    cv2.imwrite(os.path.join('output', 'odepth.png'), depth_image.astype(np.uint16))
    
    # even though we eventually want float, put it back in uint16
    depth_image = depth_image.astype(np.uint16)

    oriHeight, oriWidth, _ = rgb_image.shape
    oriSize = (oriWidth, oriHeight)

    # resize image to enable sizes divisible  by 32
    # Or assume we already have matching images
    use_size = (1248, 384) # use_size = (1280, 480)
    rgb_image = cv2.resize(rgb_image, use_size)
    rgb_image = rgb_image.astype(np.float32) / 255 # Normalize to 0-1

    depth_image = cv2.resize(depth_image, use_size)

    # depth_image is also inverted. Try to invert, but zeros?
    maxDepth = 2200 # seems arbitrary?
    #depth_image = (maxDepth - depth_image) * 30 #scale back to u16

    # Sometimes we need to flip it
    #depth_image = cv2.flip(depth_image, 0) # 0 means vertical, 2 means horizontal
    #depth_image = cv2.flip(depth_image, 2) # 0 means vertical, 2 means horizontal

    # compute normal using SNE
    sne_model = SNE()
    
    # Not sure what we want for camera parameters for our size image!
    camParam = torch.tensor([[7.215377e+02, 0.000000e+00, 6.095593e+02],
                             [0.000000e+00, 7.215377e+02, 1.728540e+02],
                             [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=torch.float32)  # camera parameters
    
    # SNE converts depth to single() in lower range, so let's export single()
    # instead of u16!

    # our depthmaps have a different scale, so use a smaller divisor
    normal = sne_model(torch.tensor(depth_image.astype(np.float32)), camParam)
    #depth_image = depth_image.astype(np.float32)/1000
    #depth_image = depth_image.astype(np.float32)

    #    normal = sne_model(torch.tensor(depth_image), camParam)

    normal_image = normal.cpu().numpy()
    normal_image = np.transpose(normal_image, [1, 2, 0])
    cv2.imwrite(os.path.join('output', 'normal.png'), cv2.cvtColor(255*(1+normal_image)/2, cv2.COLOR_RGB2BGR))
    normal_image = cv2.resize(normal_image, use_size)

    rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(dim=0)
    normal_image = transforms.ToTensor()(normal_image).unsqueeze(dim=0)

    with torch.no_grad():
        pred = model.netRoadSeg(rgb_image, normal_image)

        palet_file = 'datasets/palette.txt'
        impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3*256))
        
        pred_img = tensor2labelim(pred, impalette)
        pred_img = cv2.resize(pred_img, oriSize)
        prob_map = tensor2confidencemap(pred)
        prob_map = cv2.resize(prob_map, oriSize)
        cv2.imwrite(os.path.join('output', 'pred.png'), pred_img)
        cv2.imwrite(os.path.join('output', 'prob_map.png'), prob_map)
        cv2.imwrite(os.path.join('output', 'depth.png'), depth_image.astype(np.uint16))
