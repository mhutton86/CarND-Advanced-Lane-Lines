import glob
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from plot_helper import *
from moviepy.editor import VideoFileClip
from lane_tracker import LaneTracker

# camera calibration variables
calibration_image_dir_pattern = './camera_cal/calibration*.jpg'
chessboard_inside_corners = (9, 6)
cal = {}

# output_mode = 'image'

output_mode = 'video'


# Helper functions
def load_image(image_path):
    loaded_imaged = mpimg.imread(image_path)
    return loaded_imaged


def calibrate_camera():
    """
    Function to grab matric and variables necessary to calibrate and undistort images from a camera.
    based on Udacity SDCEN calibrating camera section.
    :return: TODO
    """
    global cal
    print('calibrate_camera called')

    images = glob.glob(calibration_image_dir_pattern)

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plan

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ... (8, 5, 0)
    objp = np.zeros((chessboard_inside_corners[0] * chessboard_inside_corners[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_inside_corners[0], 0:chessboard_inside_corners[1]].T.reshape(-1, 2)

    for image_path in images:
        # load image and convert to grayscale
        image = load_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_inside_corners, None)

        if ret is True:
            # store for calibration
            imgpoints.append(corners)
            objpoints.append(objp)

            # image_with_corners = cv2.drawChessboardCorners(image, chessboard_inside_corners, corners, ret)
            # plot_image(image_with_corners)

    # take the found chessboards and feed into calibrate camera function to get distortion matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    cal['ret'] = ret
    cal['mtx'] = mtx
    cal['dist'] = dist
    cal['rvecs'] = rvecs
    cal['tvecs'] = tvecs

    test_image = load_image('camera_cal/calibration3.jpg')
    plot_2_images(test_image, cv2.undistort(test_image, cal['mtx'], cal['dist'], None, cal['mtx']))


# calibrate camera, store into pickle for easy access
pickle_file_name = 'objs.pkl'
if os.path.exists(pickle_file_name):
    with open(pickle_file_name, 'rb') as f:  # Python 3: open(..., 'rb')
        cal = pickle.load(f)
else:
    calibrate_camera()

    with open(pickle_file_name, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(cal, f)

#  First load a single image and run through pipeline
if output_mode is 'image':

    # image_paths = glob.glob('test_images/test3.jpg')
    image_paths = glob.glob('test_images/*.jpg')

    for idx, image_path in enumerate(image_paths):
        tracker = LaneTracker(cal)

        image = load_image(image_path)
        output = tracker.handle_frame(image)

        path = 'output_images/' + str(idx) + '.jpg'
        mpimg.imsave(path, output)

        # plot_image(output)
        # plt.show()

elif output_mode is 'video':
    tracker = LaneTracker(cal)
    output_video = 'output_video/output.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    # clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(tracker.handle_frame)  # NOTE: this function expects color images!!
    white_clip.write_videofile(output_video, audio=False)
