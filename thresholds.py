import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient is 'x':
        sobel_derivative = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel_derivative = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.abs(sobel_derivative)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    mask = (scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[mask] = 1  # Remove this line
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    sobel_deriv = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * sobel_deriv / np.max(sobel_deriv))
    # 5) Create a binary mask where mag thresholds are met
    mask = (scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[mask] = 1
    return binary_output


def color_thresh(img, s_thresh=(0, 255), v_thresh=(0, 255)):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    # return v_binary

    # if you wanted to show a single color channel image called 'gray'
    # for example, call as plt.imshow(gray, cmap='gray')
    # f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(h_channel)
    # ax1.set_title('H', fontsize=50)
    # ax2.imshow(l_channel)
    # ax2.set_title('L', fontsize=50)
    # ax3.imshow(s_channel)
    # ax3.set_title('S', fontsize=50)
    # ax4.imshow(hsv_h_channel)
    # ax4.set_title('H', fontsize=50)
    # ax5.imshow(hsv_s_channel)
    # ax5.set_title('S', fontsize=50)
    # ax6.imshow(hsv_v_channel)
    # ax6.set_title('V', fontsize=50)
    # ax7.imshow(r_channel)
    # ax7.set_title('R', fontsize=50)
    # ax8.imshow(g_channel)
    # ax8.set_title('G', fontsize=50)
    # ax9.imshow(b_channel)
    # ax9.set_title('B', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()

    # print(np.mean(s_channel))
    # print(np.mean(hsv_s_channel))

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary) == 1] = 1

    return output


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    mask = (grad_direction > thresh[0]) & (grad_direction < thresh[1])
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(grad_direction)
    binary_output[mask] = 1
    # binary_output = mask
    return binary_output

# # Read in an image
# image = mpimg.imread('example_images/signs_vehicles_xygrad.png')
#
# # Choose a Sobel kernel size
# ksize = 3  # Choose a larger odd number to smooth gradient measurements
#
# # Apply each of the thresholding functions
# gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
# grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
# mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(20, 100))
# dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.3, 1.3))
#
# combined = np.zeros_like(dir_binary)
# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
#
# # Run the function
# plt.interactive(True)
# # mag_binary = mag_thresh(gradx, sobel_kernel=ksize, mag_thresh=(30, 100))
# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(combined, cmap='gray')
# ax2.set_title('Thresholded Magnitude', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()
