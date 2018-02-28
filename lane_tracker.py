from line import Line
from thresholds import *
from plot_helper import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def warp_image(img, transform_matrix):
    shape = img.shape[1::-1]
    return cv2.warpPerspective(img, transform_matrix, shape, flags=cv2.INTER_LINEAR)


def edge_detection(img):
    """ Detect edges of an image using various thresholding techniques. """
    new_img = np.copy(img)

    # Sobel x
    r_channel = new_img[:, :, 0]
    gradx = abs_sobel_thresh(r_channel, orient='x', thresh=(12, 100))
    grady = abs_sobel_thresh(r_channel, orient='y', thresh=(25, 255))

    grad = np.zeros_like(gradx)
    grad[(gradx == 1) & (grady == 1)] = 1

    # Threshold color channel
    c_binary = color_thresh(img, s_thresh=(100, 255), v_thresh=(150, 255))

    # Combine the effects of all thresholds
    final = np.zeros_like(gradx)
    final[(gradx == 1) & (grady == 1) | c_binary == 1] = 1
    # return final.astype(np.uint8) * 255
    return final


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

        self.ploty = None

    def handle_frame(self, image):
        """Perform all the processing of images/videos to find and print an overlay of the lane."""

        # undistort image
        dst = cv2.undistort(image, self.cal['mtx'], self.cal['dist'], None, self.cal['mtx'])

        # warp the image to a bird's eye view
        warped = warp_image(dst, self.M)

        # pts = np.array(self.transform_source, np.int32)
        # pts = pts.reshape((-1, 1, 2))
        # cv2.polylines(dst, [pts], True, (0, 255, 255))
        # pts = np.array(self.transform_dest, np.int32)
        # pts = pts.reshape((-1, 1, 2))
        # cv2.polylines(warped, [pts], True, (0, 255, 255))
        # plot_2_images(dst, warped)

        edges = edge_detection(warped)

        # mpimg.imsave('binary.jpg', edges)
        # plot_2_images(dst, image)

        # color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        # return color_edges

        color_lines = self.find_lines(edges)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        y_points = self.ploty.astype(np.int64)
        x_points = self.left_line.bestx.astype(np.int64)
        pts_left = [np.vstack([x_points, y_points]).T]
        x_points = self.right_line.bestx.astype(np.int64)
        pts_right = [np.flipud(np.vstack([x_points, y_points]).T)]

        pts = np.hstack((pts_right, pts_left))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(warp_zero, pts, (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        new_warp = warp_image(warp_zero, self.Minv)
        # Combine the result with the original image
        result = cv2.addWeighted(dst, 1, new_warp, 0.3, 0)

        self.print_line_info(result)

        return result

    def print_line_info(self, image):
        color = (255, 255, 255)  # white
        curvature_output = "Curvature: {:.2f} m" \
            .format(np.average([self.left_line.radius_of_curvature, self.right_line.radius_of_curvature]))
        cv2.putText(image, curvature_output, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        lane_offset_text = "Lane Center Offset: {:.4f} m".format(np.abs(self.left_line.line_base_pos - self.right_line.line_base_pos)/2)
        cv2.putText(image, lane_offset_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def find_lines(self, warped):
        """
        Code to find lines using code from the SDCEN lessons.
        :param out_img:
        :param warped:
        :return:
        """
        if self.ploty is None:
            self.ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

        if self.left_line.detected is False or self.right_line.detected is False:
            lane_out = self.init_lane_search(warped)
        else:
            lane_out = warped
            self.cont_lane_search(warped)

        # plot_2_images(warped, lane_out)

        color_warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)

        color_warped[self.left_line.ally, self.left_line.allx] = [255, 0, 0]
        color_warped[self.right_line.ally, self.right_line.allx] = [0, 0, 255]

        # y_points = self.ploty.astype(np.int64)
        # x_points = self.left_line.bestx.astype(np.int64)
        # cv2.polylines(color_warped, [np.vstack([x_points, y_points]).T], 0, (255, 255, 0), 2)
        # x_points = self.right_line.bestx.astype(np.int64)
        # cv2.polylines(color_warped, [np.vstack([x_points, y_points]).T], 0, (255, 255, 0), 2)
        # plot_image(color_warped)

        return lane_out

    def init_lane_search(self, image):
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
        out_img = np.dstack((image, image, image)) * 255
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
        window_height = np.int(image.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
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
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
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
        left_x_points = nonzerox[left_lane_inds]
        left_y_points = nonzeroy[left_lane_inds]
        right_x_points = nonzerox[right_lane_inds]
        right_y_points = nonzeroy[right_lane_inds]

        self.left_line.add_points(left_x_points, left_y_points, self.ploty)
        self.right_line.add_points(right_x_points, right_y_points, self.ploty)

        return out_img

    def cont_lane_search(self, image):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit

        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.left_line.add_points(leftx, lefty, self.ploty)
        self.right_line.add_points(rightx, righty, self.ploty)
