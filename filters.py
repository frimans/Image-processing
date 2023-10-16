"""
Non-local wighted means filtering
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def sobel_edge(im):
    padding = 2
    placeholder = np.zeros([im.shape[0], im.shape[1]], dtype=np.uint8)
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                        [0,   0,  0],
                        [1,   2,  1]])
    for x in range(0 +padding,im.shape[0]-padding):
        for y in range(0+padding, im.shape[1]-padding):
            value1 = np.sum(kernel_x*im[x-1:x+2, y-1:y+2])
            value2 = np.sum(kernel_y*im[x-1:x+2, y-1:y+2])
            if value1 >= 255:
                value1 = 255
            if value2 >= 255:
                value2 = 255
            placeholder[x,y] = np.sqrt(np.square(value1) + np.square(value2))
    return placeholder



def NLWM(im,win,ext, h):
    """

    :param im:
    :return:
    """
    print("NLM running...")
    shape_x, shape_y = im.shape
    placeholder = np.zeros([shape_x, shape_y], dtype=np.uint8)

    shape_x, shape_y = im.shape
    for x in range(0 + ext +win, shape_x-ext -win):
        for y in range(0+ ext +win,shape_y - ext -win):

            window = im[x-win:x+win, y -win:y + win]

            color = 0
            totweight = 0

            for ext_x in range(-ext, ext):
                for ext_y in range(-ext,  ext):
                    com_window = im[x +ext_x - win:x + ext_x + win, y +ext_y -win:y + ext_y + win]


                    diff = np.sqrt(np.sum(np.square(window - com_window)))
                    weight = np.exp(-diff / h)

                    #print(weight)
                    #print(diff)
                    totweight += weight
                    color += weight * im[x + ext_x, y + ext_y]
            color /= totweight
            placeholder[x,y] = int(color)
    return placeholder


def median(im, win):
    """

    :param im:
    :return:
    """
    shape_x, shape_y = im.shape
    placeholder = np.zeros([shape_x, shape_y],dtype=np.uint8)

    for x in range(0, shape_x):
        for y in range(0, shape_y):
            placeholder[x,y] = np.median(im[x-win:x+win, y-win:y+win])

    return placeholder

def mean(im, win):
    """

    :param im:
    :return:
    """
    shape_x, shape_y = im.shape
    placeholder = np.zeros([shape_x, shape_y],dtype=np.uint8)

    for x in range(0, shape_x):
        for y in range(0, shape_y):
            placeholder[x,y] = np.mean(im[x-win:x+win, y-win:y+win])

    return placeholder





search_window = 20 #pixels
filter_window = 5
extent = 2
h = 20
#image = "C:/Users/Lenovo/Desktop/hat.png"
image = "C:/Users/Lenovo/Desktop/shepplogan.png"
im= cv2.imread(image,0)



shape_x, shape_y = im.shape
print(shape_x, "|", shape_y)
im_med = median(im, filter_window)


im_mean = mean(im, filter_window)

im_sobel = sobel_edge(im)
im_NLM= NLWM(im, filter_window, extent, h)
#im_NLM= nonLocalMeans(im, extent, filter_window, h)
plt.Figure()
plt.subplot(2,4,1)
plt.title("original")
plt.imshow(im, cmap="gray")

plt.subplot(2,4,2)
plt.title("NLM")
plt.imshow(im_NLM, cmap="gray")

plt.subplot(2,4,3)
plt.title("mean")
plt.imshow(im_mean, cmap="gray")

plt.subplot(2,4,4)
plt.title("median")
plt.imshow(im_med, cmap="gray")

plt.subplot(2,4,5)
plt.title("sobel")
plt.imshow(im_sobel, cmap="gray")


plt.show()







