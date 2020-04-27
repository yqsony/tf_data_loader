"""
load nyu v2 dataset
Quan Yuan
2020/04/31
"""

import argparse
import numpy
import cv2
import LoadData.load_nyu as load_nyu



def process(data_path):
    gen = load_nyu.generate_v2_data(data_path)
    # dataset = load_nyu.generator_v2(gen, output_shape=())
    for image, depth in gen:
        image = numpy.rollaxis(image, 0, 3)
        cv2.imshow('rgb', image.astype(numpy.uint8))
        cv2.waitKey()
        cv2.imshow('210', image.astype(numpy.uint8)[..., [2, 1, 0]])
        cv2.waitKey()
        cv2.imshow('depth', (numpy.squeeze(depth)*255.0/numpy.max(depth)).astype(numpy.uint8))
        cv2.waitKey()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="depth from rgb")
    parser.add_argument('mat_path', type=str, help="path to NYU data")
    parser.add_argument('--num_epoch', type=int, help="total number epoch", default=200)
    parser.add_argument('--batch_size', type=int, help="batch_size", default=24)
    parser.add_argument('--lr', type=float, help="learning rate", default=0.001)
    parser.add_argument('--optim', type=str, help="name of optimizer", default='adam')

    args = parser.parse_args()

    process(data_path=args.mat_path)




