"""
A script to perform image registration
"""
import numpy as np
from matplotlib import pyplot as plt
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def iterate(image1,image2):
    difference = []
    minimum = 0
    difference.append(np.sum(image1-image2))
    for i in range(1,360):
        difference.append(np.sum(image1-rotate_image(image2,i)))
    minimum = np.argmin(difference)
    print(minimum)

    return difference,minimum

def find_center(image):
    rows1, cols1 = image.shape
    print(image.shape)
    xs = []
    ys = []
    for x in range(0, rows1):
        for y in range(0, cols1):

            if image[x, y] != 0:

                xs.append(x)
                ys.append(y)

    print(np.average(xs))
    print(np.average(ys))
    avgx = int(np.average(xs))
    avgy = int(np.average(ys))

    print(avgx, "|", avgy)
    return avgx, avgy

def iteration_pass(im, im2):
    im2_x, im2_y = find_center(im2)
    im1_x, im1_y = find_center(im)

    needed_transition = [im1_x - im2_x, im1_y - im2_y]
    print("Needed transition:", needed_transition)

    rows, cols = im2.shape
    M = np.float32([[1, 0, needed_transition[1]], [0, 1, needed_transition[0]]])
    im2 = cv2.warpAffine(im2, M, (cols, rows))



    print(im)
    print(im.shape)
    print(im2.shape)

    difference = im - im2
    dif = np.sum(im-im2)
    print(np.sum(difference))
    differences, minimum = iterate(im, im2)

    im2 = im2 = rotate_image(im2, minimum)
    difference = im - im2
    print(np.sum(difference))
    """
    cv2.imshow("correct", im)
    cv2.imshow("rotated", im2)
    cv2.imshow("difference", difference)

    plt.figure()
    plt.plot(differences)
    plt.xlabel("Rotation (degrees)")
    plt.ylabel("Difference")
    plt.show()

    cv2.waitKey(0)
    """

    # closing all open windows

    return im, im2, dif

import cv2
epochs = 4
image = "C:/Users/Lenovo/Desktop/shepplogan.png"
im= cv2.imread(image,0)
im2 = im

im2 = rotate_image(im2, 30)
rows,cols = im2.shape
M = np.float32([[1,0,-20],[0,1,-20]])
im2 = cv2.warpAffine(im2,M,(cols,rows))

differences = []
differences.append(np.sum(im-im2))

for epoch in range(0, epochs):
    im, im2, dif = iteration_pass(im, im2)
    differences.append(dif)

print(differences)
plt.figure()
plt.title("Registration epoch effect on matching error")
plt.xlabel("Epoch")
plt.subplot(2,1,1)
plt.plot(differences)
plt.subplot(2,2,3)
plt.imshow(im-im2)

plt.show()

