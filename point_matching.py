# -*- coding: utf-8 -*-

import rasterio
from skimage import transform as tf
from skimage.feature import (match_descriptors, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


with rasterio.open('./data/IMG_0137.JPG') as src:
    print("The dataset name is: " + src.name)
    print("The number of bands is: " + str(src.count))
    print("The CRS is: " + str(src.crs))
    print("The bounds are: " + str(src.bounds))
    print("The number of columns is: " + str(src.width))
    print("The number of rows is: " + str(src.height))
    print("The datatype is: " + str(src.dtypes))
    print("The affine transform is: " + str(src.transform))
    
    img = src.read(1).astype('float64')
    img1 = rgb2gray(img)
    
    print("The object datatype is: " + str(type(img1)))
    print("The image dimensions are: " + str(img1.shape))
    print("The pixel datatype is: " + str(img1.dtype))
    t = src.transform    

img2 = tf.rotate(img1, 180)
tform = tf.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
                           translation=(0, -200))
img3 = tf.warp(img1, tform)

descriptor_extractor = ORB(n_keypoints=10)

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img3)
keypoints3 = descriptor_extractor.keypoints
descriptors3 = descriptor_extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)

out_coords = []
for kp in keypoints2:
    out_coords.append(t * kp)

fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12)
ax[0].axis('off')
ax[0].set_title("Original Image vs. Transformed Image")

plot_matches(ax[1], img1, img3, keypoints1, keypoints3, matches13)
ax[1].axis('off')
ax[1].set_title("Original Image vs. Transformed Image")


plt.show()


def georeference(georefed, to_georef):
    new_georef = []
    return new_georef
