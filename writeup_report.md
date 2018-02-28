## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_undistort.png "Undistorted"
[image2]: ./output_images/pre-processed.jpg "Road Transformed"
[image3]: ./output_images/binary.jpg "Binary Example"
[image4]: ./output_images/warped_lanes.png "Warp Example"
[image5]: ./output_images/lane_line_fits.png "Fit Visual"
[image6]: ./output_images/plotted_result.jpg "Output"
[video1]: ./output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients.

The code for this step is contained in the `calibrate_camera` function of the python script `find_lanes.py` at line `28`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result (left: distorted, right: undistorted): 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how you used color transforms, gradients or other methods to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are found in the function `edge_detection` in the pythong file `lane_tracker.py` at ~ line `15`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a source and target matrices in lines `46-58` in `lane_tracker.py`, pre-calculations of the transform matrics to and from bird's eye view at lines `60-61` in `lane_tracker.py`. 
I then use the `warp_image` at line `72` and line `105` in `lane_tracker.py` that takes an image and tranformation matrix to perform the necessary pespective transform to/from bird's eye view.
```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 556, 474      | 238, 0        | 
| 728, 474      | 1070, 0       |
| 1070, 689     | 1070, 720     |
| 238, 689      | 238, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how you identified lane-line pixels and fit their positions with a polynomial?

I wrote a function `find_lines` in `lane_tracker.py` at line `124` that would initially detect lines by finding left and right lane peaks in a histogram and using the window search technique. 
The initial search would identify the lane line pixels. I would then perform a polynomial fit against the points. As I tracked my lines between each video frame, I would average each lane's fit over the last 15 frames to smooth choppy behavior.


I would also use the average polynomial fit to then help mask images and find points for future lane lines. The averaging and masking would help when I found less than ideal edges of lane lines.


My line averaging logic is in the function `add_points` in `line.py` at line `42`

![alt text][image5]

#### 5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the road curvature radius, I used the provided equation for radius of curvature in Lesson 15, Part 35 "Measuring Curvature". In function `calculate_lane_curvature` in `line.py` at line `69` I took the average x positions of each lane to generate a new fit equation, which converted units from pixels to meters. 

To obtain the conversion from pixels to meters, I looked up the average interstate lane width in california (12ft), and the dashed line length (10ft) in real world measurements. I then calculated the lane width and dashed line length in pixels for straight lines in a transformed bird's eye view image.
I then calculated the meters to pixels in the y and x directions. These can be found at line `37-38` in `line.py`.

```python
self.ym_per_pix = 1 / 53.81  # meters per pixel in y dimension
self.xm_per_pix = 1 / 219.28  # meters per pixel in x dimension
``` 

Final equation at line `69` in `line.py`:
```python
def calculate_lane_curvature(self, ploty):
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty * self.ym_per_pix, self.bestx * self.xm_per_pix, 2)
    # Calculate the new radii of curvature in meters
    y_eval = np.max(ploty)
    self.radius_of_curvature = ((1 + (
                2 * fit_cr[0] * y_eval * self.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
```

I then averaged each lane line's curvature, to give the lane's curvature. at line `119` in `lane_tracker.py`:
```python
np.average([self.left_line.radius_of_curvature, self.right_line.radius_of_curvature])
```

To calculate the line center offset, I tracked each line's individual distance from the midpoint at line in meters `78` in `line.py`:
```python
def calculate_center_offset(self):
    self.line_base_pos = np.abs(self.bestx[-1] - self.screen_midpoint) * self.xm_per_pix
```

Then took the absolute value of the difference between both lane's distance from the center, divided by 2 to give me the car's offset from the center. At line `121` in `lane_tracker.py`:
```python
np.abs(self.left_line.line_base_pos - self.right_line.line_base_pos)/2
```

Reference: 
  * [Lane Width reference](http://www.dot.ca.gov/hq/paffairs/faq/faq92.htm)
  * [Dashed Line length reference](http://www.ctre.iastate.edu/pubs/itcd/pavement%20markings.pdf)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines `96-110` in `lane_tracker.py` in the function `handle_frame()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first problem I had was getting the edge detection thresholds to work. As I'd adjust the thresholds to support the road color change or shadows, I'd break the thresholds in other parts of the road. I had to use a larger combination of thresholds to solve that problem. 

One of the other issues I faced was weirdly trying to create inputs to library's in the way that the library's were expecting them. I had to use example input values and the debugger to manipulate my data to work.

My threshold detection could still be improved in areas where there's inconsistent shadows, or differentiating between breaks in the road from lanes themselves.

I feel my pipeline could have been improved by using deep learning to identify lines.

My pipeline would hypothetically fail if there were removed old painted lines that still showed up in edge detection. My lane detection would fail if lane lines were missing for more than a quarter-half second

