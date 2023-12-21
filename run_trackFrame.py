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
    opt.num_threads = 4
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.isTrain = False

    example_dataset = dataset()
    model = create_model(opt, example_dataset)
    model.setup(opt)
    model.eval()

    # NN doesn't like our track video frames. It was trained on KITTI, where all the "action" is
    # in the bottom half of the frame. SO, we can try expanding the rgb and depth images to include
    # a "black" top half. TBD.

    rgb_image = cv2.cvtColor(cv2.imread(os.path.join('datasets', 'track','testing','image_2','frame_500.jpg')), cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join('output', 'oimage.png'), rgb_image)

    # pad the image by its height    
    topPad, width, channels = rgb_image.shape
    rgb_image = cv2.copyMakeBorder(rgb_image, topPad, 0, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
    fName = 'frame_700.pfm'

    # cv2 brings images in with shape rows, columns 
    depth_image = cv2.imread(os.path.join('b:\\','code','Midas','output',fName), cv2.IMREAD_UNCHANGED)
    depth_image = cv2.copyMakeBorder(depth_image, topPad, 0, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
    
    # original depth written as a 16-bit .png looks good, but inverted and small scale
    cv2.imwrite(os.path.join('output', 'odepth.png'), depth_image.astype(np.uint16))
    
    # even though we eventually want float, put it back in uint16
    depth_image = depth_image.astype(np.uint16)

    oriHeight, oriWidth, _ = rgb_image.shape
    oriSize = (oriWidth, oriHeight)

    # resize image to enable sizes divisible  by 32
    # Or assume we already have matching images
    # note that .resize uses Width, Height (ugh)
    # resize image to enable sizes divide 32
    sizeModulo = 32
    padMultiplier = 2 # how much we add to the top of the image (as a multiplier)
    use_size = (sizeModulo * 40, sizeModulo * 10 * padMultiplier)
    rgbMax = rgb_image.max()

    rgb_image = cv2.resize(rgb_image, use_size)
    rgb_image = rgb_image.astype(np.float32) / (rgbMax + 1) # Normalize to 0-1

    depth_image = cv2.resize(depth_image, use_size)

    # depth_image is also inverted. 
    maxDepth = depth_image.max()
    depth_image = (maxDepth - depth_image) * round(pow(2,16) / maxDepth) #scale to u16 range

    # Sometimes we need to flip it
    #depth_image = cv2.flip(depth_image, 0) # 0 means vertical, 2 means horizontal
    #depth_image = cv2.flip(depth_image, 2) # 0 means vertical, 2 means horizontal

    # compute normal using SNE
    sne_model = SNE()
    
    # Not sure what we want for camera parameters for our size image!
    #camParam = torch.tensor([[7.215377e+02, 0.000000e+00, 6.095593e+02],
    #                        [0.000000e+00, 7.215377e+02, 1.728540e+02],
    #                         [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=torch.float32)  # camera parameters

    # guesstimate for our 1280 x 480 source material
    # Fx in pixels, Fy in pixels, so 1280 pixels / (say) 6mm wide * 4mm focal = 860
    camParam = torch.tensor([[860, 0, 640],
                             [0, 860, 40],
                             [0, 0, 1]], dtype=torch.float32)  # camera parameters

    # But now we've rescaled!    
    # our depthmaps have a different scale, so use a smaller divisor
    normal = sne_model(torch.tensor(depth_image.astype(np.float32)/1000), camParam)

    normal_image = normal.cpu().numpy()
    normal_image = np.transpose(normal_image, [1, 2, 0])
    cv2.imwrite(os.path.join('output', 'normal.png'), cv2.cvtColor(255*(1+normal_image)/2, cv2.COLOR_RGB2BGR))
    
    # Normal is 3 channel so maybe we need a 3-channel use_size?
    normal_image = cv2.resize(normal_image, use_size)

    cv2.imwrite(os.path.join('output', 'rgb_image.png'), rgb_image * 255)
    rgb_image_tensor = transforms.ToTensor()(rgb_image).unsqueeze(dim=0)
    normal_image = transforms.ToTensor()(normal_image).unsqueeze(dim=0)

    with torch.no_grad():
        pred = model.netRoadSeg(rgb_image_tensor, normal_image)

        palet_file = 'datasets/palette.txt'
        impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3*256))
        
        pred_img = tensor2labelim(pred, impalette)
        pred_img = cv2.resize(pred_img, oriSize)
        prob_map = tensor2confidencemap(pred)
        prob_map = cv2.resize(prob_map, oriSize)
        cv2.imwrite(os.path.join('output', 'pred.png'), pred_img)
        cv2.imwrite(os.path.join('output', 'prob_map.png'), prob_map)
        cv2.imwrite(os.path.join('output', 'depth.png'), depth_image.astype(np.uint16))
