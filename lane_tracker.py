from line import Line
from thresholds import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def warp_image(img, transform_matrix):
    shape = img.shape[1::-1]
    return cv2.warpPerspective(img, transform_matrix, shape, flags=cv2.INTER_LINEAR)


def edge_detection(img, s_thresh=(180, 255), sx_thresh=(30, 100)):  # sx_thresh was 40-100
    new_img = np.copy(img)

    # Sobel x
    r_channel = new_img[:, :, 0]
    gradx = abs_sobel_thresh(r_channel, orient='x', thresh=(12, 100))
    grady = abs_sobel_thresh(r_channel, orient='y', thresh=(25, 255))

    grad = np.zeros_like(gradx)
    grad[(gradx == 1) & (grady == 1)] = 1

    # return grad.astype(np.uint8) * 255

    # Threshold color channel

    c_binary = color_thresh(img, s_thresh=(100, 255), v_thresh=(150, 255))
    # return c_binary.astype(np.uint8) * 255

    final = np.zeros_like(gradx)
    final[(gradx == 1) & (grady == 1) | c_binary == 1] = 1
    # plot_image(output)
    return final.astype(np.uint8) * 255

class LaneTracker:
    def __init__(self, cal):
        self.cal = cal

        # the lanes we're tracking
        self.left_line = Line()
        self.right_line = Line()

        # the source/destination values for the bird's eye-view transform
        self.transform_source = np.float32([
            [556, 474],  # top left
            [728, 474],  # top right
            [1070, 689],  # bottom right
            [238, 689]  # bottom left
        ])

        self.transform_dest = np.float32([
            [238, 0],  # top left
            [1070, 0],  # top right
            [1070, 720],  # bottom right
            [238, 720]  # bottom left
        ])

        self.M = cv2.getPerspectiveTransform(self.transform_source, self.transform_dest)
        self.Minv = cv2.getPerspectiveTransform(self.transform_dest, self.transform_source)

    def handle_frame(self, image):
        # undistort image
        dst = cv2.undistort(image, self.cal['mtx'], self.cal['dist'], None, self.cal['mtx'])

        # warp the image to a bird's eye view
        warped = warp_image(dst, self.M)

        # pts = np.array(source, np.int32)
        # pts = pts.reshape((-1,1,2))
        # cv2.polylines(dst,[pts],True,(0,255,255))
        # plot_2_images(dst, warped)

        edges = edge_detection(warped)

        # color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        # return color_edges

        color_lines = self.find_lines(edges)

        return warped

    def find_lines(self, warped):
        """
        Code to find lines using code from the SDCEN lessons.
        :param out_img:
        :param warped:
        :return:
        """
        histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
        # plt.plot(histogram)
        # plt.show()
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(warped, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(warped, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        self.left_line.allx = nonzerox[left_lane_inds]
        self.left_line.ally = nonzeroy[left_lane_inds]
        self.right_line.allx = nonzerox[right_lane_inds]
        self.right_line.ally = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_line.current_fit = np.polyfit(self.left_line.ally, self.left_line.allx, 2)
        self.right_line.current_fit = np.polyfit(self.right_line.ally, self.right_line.allx, 2)

        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = self.left_line.current_fit[0] * ploty ** 2 + self.left_line.current_fit[1] * ploty + self.left_line.current_fit[2]
        right_fitx = self.right_line.current_fit[0] * ploty ** 2 + self.right_line.current_fit[1] * ploty + self.right_line.current_fit[2]

        color_warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)

        color_warped[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        color_warped[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        out_img = np.copy(warped)
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

        return left_lane_inds, nonzerox, nonzeroy, right_lane_inds
