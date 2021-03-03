import os
import sys
import time
import torch
import imageio
import skimage
import argparse
import numpy as np
import scipy as scp
import skimage.transform
from convcrf import convcrf
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from torch.autograd import Variable
from alisuretool.Tools import Tools
from utils import pascal_visualizer as vis
from pydensecrf.utils import unary_from_softmax


def plot_results(image, prediction, label, save_file):
    figure = plt.figure()
    figure.tight_layout()

    ax = figure.add_subplot(1, 3, 1)
    ax.set_title('Image ')
    ax.axis('off')
    ax.imshow(image)

    ax = figure.add_subplot(1, 3, 2)
    ax.set_title('Label')
    ax.axis('off')
    ax.imshow((label * 255).astype(np.uint8))

    ax = figure.add_subplot(1, 3, 3)
    ax.set_title('CRF Output')
    ax.axis('off')
    ax.imshow((prediction * 255).astype(np.uint8))

    plt.savefig(save_file)
    pass


def do_crf_inference(image, annotation, t=5, use_gpu=False):
    annotation = np.expand_dims(annotation, axis=-1)
    unary = np.concatenate([annotation, 1 - annotation], axis=-1)
    num_classes = unary.shape[2]
    shape = image.shape[0:2]

    # make input pytorch compatible
    img_var = torch.tensor(image.transpose(2, 0, 1).reshape([1, 3, shape[0], shape[1]]))
    unary_var = torch.tensor(unary.transpose(2, 0, 1).reshape([1, num_classes, shape[0], shape[1]]))
    img_var = img_var.cuda() if use_gpu else img_var
    unary_var = unary_var.cuda() if use_gpu else unary_var

    # Create CRF module
    gausscrf = convcrf.GaussCRF(conf=convcrf.default_conf, shape=shape, nclasses=num_classes, use_gpu=use_gpu)
    gausscrf = gausscrf.cuda() if use_gpu else gausscrf

    # Perform CRF inference
    prediction = gausscrf.forward(unary=unary_var, img=img_var, num_iter=t)
    prediction = prediction.data.cpu().numpy()[0]
    return np.argmax(prediction, axis=0)


def do_crf2_inference(image, annotation, t=5):
    image = np.ascontiguousarray(image)

    annotation = np.expand_dims(annotation, axis=0)
    annotation = np.concatenate([annotation, 1 - annotation], axis=0)
    h, w = image.shape[:2]

    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(annotation)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(image), compat=10)
    q = d.inference(t)

    result = np.array(q).reshape((2, h, w))
    return result[0]


if __name__ == '__main__':
    image_file = "./data_my/input/DUTS-TR_ILSVRC2012_test_00000004_image.bmp"
    label_file = "./data_my/input/DUTS-TR_ILSVRC2012_test_00000004_cam.bmp"
    save_file = Tools.new_dir("./data_my/output/DUTS-TR_ILSVRC2012_test_00000004.png")
    image = imageio.imread(image_file)
    label = imageio.imread(label_file)

    use_gpu = False

    # Compute CRF inference
    start_time = time.time()
    prediction = do_crf_inference(image, label / 255, use_gpu=use_gpu)
    # prediction = do_crf2_inference(image, label / 255)
    Tools.print("time={}".format(time.time() - start_time))

    # Plot output
    plot_results(image, prediction, label, save_file=save_file)
    pass
