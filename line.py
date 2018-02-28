import numpy as np
from collections import deque


def calculate_x(fit, ploty):
    return fit[0] * ploty ** 2 \
           + fit[1] * ploty \
           + fit[2]


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # number of iterations to track over
        self.n = 10
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque()
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 1 / 53.81  # meters per pixel in y dimension
        self.xm_per_pix = 1 / 219.28  # meters per pixel in x dimension
        # Define midpoint
        self.screen_midpoint = 1280 / 2

    def add_points(self, x_pts, y_pts, ploty):
        self.allx = x_pts
        self.ally = y_pts

        # Fit a second order polynomial to each
        self.current_fit = np.polyfit(self.ally, self.allx, 2)

        # update fits
        fitx = self.current_fit[0] * ploty ** 2 \
               + self.current_fit[1] * ploty \
               + self.current_fit[2]

        # TODO I should likely see if the fitx was far from the best fitx to test quality
        self.recent_xfitted.append(fitx)

        if len(self.recent_xfitted) > self.n:
            self.recent_xfitted.popleft()

        x_mean = np.mean(self.recent_xfitted, axis=0)
        self.best_fit = np.polyfit(ploty, x_mean, 2)
        self.bestx = calculate_x(self.best_fit, ploty)

        self.detected = True

        self.calculate_lane_curvature(ploty)
        self.calculate_center_offset()

    def calculate_lane_curvature(self, ploty):
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty * self.ym_per_pix, self.bestx * self.xm_per_pix, 2)
        # Calculate the new radii of curvature in meters
        y_eval = np.max(ploty)
        self.radius_of_curvature = ((1 + (
                    2 * fit_cr[0] * y_eval * self.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    def calculate_center_offset(self):
        self.line_base_pos = np.abs(self.bestx[-1] - self.screen_midpoint) * self.xm_per_pix
