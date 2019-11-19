# %%
# Imports libraries
import cv2
import imutils
import time
import numpy as np
from glob import glob
import tensorflow as tf
import plot_ellipse as e
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression


# %%
# get all image into the path
paths0 = sorted([str(i) for i in glob("./data/*_c0.pgm")])
paths1 = sorted([str(i) for i in glob("./data/*_c1.pgm")])

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# %%
first_time = True

# Colors rgb
# Prediction
red = (0, 0, 255)
center_red = []
# Detection
green = (0, 255, 0)
center_green = []
# Update
blue = (255, 0, 0)
center_blue = []

# Fotograph by second
fps = 50
# Define Deltas
dt = 1/fps
# used in Matrix of estimation error Q
dq = 0.01
# used in Matrix of detector error R
dr = 0.01

# init U0
U0 = np.array([[0, 0, 0, 0]], np.float32).T

# init sigma
S0 = np.zeros(shape=(4, 4), dtype=np.float32)

# State to the measurement
C = np.diag([1.0, 1.0, 1.0, 1.0])

# Transition matrix
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], np.float32)

# Input effect matrix
B = np.array([[dt**2, 0],
              [0, dt**2],
              [0, 0],
              [0, 0]], np.float32)

# Constant matrix to multiply B
a = np.array([[4], [3]], np.float32)


# Noices of covariances matrix
Q = np.array([[dq, 0, 0, 0],
              [0, dq, 0, 0],
              [0, 0, dq, 0],
              [0, 0, 0, dq]], np.float32)

# Mesurement error matrix
R = np.array([[dr, 0, 0, 0],
              [0, dr, 0, 0],
              [0, 0, dr, 0],
              [0, 0, 0, dr]], np.float32)


def get_vec(box):
    px_ = box[0, 0] + box[0, 2]/2
    py_ = box[0, 1] + box[0, 3]/2
    y = np.array([[px_], [py_], [0], [0]])
    return y


def get_xy(vector):
    center = int(vector[0] + .5), int(vector[1] + .5)
    return center


# %%
# Iterative Kalman Filter
for imagePath in paths1:
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(1200, image.shape[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect people in the image
    rectangles, weights = hog.detectMultiScale(
        gray, winStride=(8, 8), padding=(32, 32), scale=1.10)

    if not first_time:
        center_red.append(get_xy(U0))
        # Prediction
        # U1 = A@U0 + B@a
        U1 = A@U0 + B@U0[:2]
        S1 = ((A@S0)@(A.T)) + Q
        e.plot_ellipse(image, U1[:2], S0[:2, :2], red)
        U0 = U1
        S0 = S1

        if len(rectangles) > 0:
            # Detection position
            y = get_vec(rectangles)
            e.plot_ellipse(image, y[:2], R[:2, :2], green)
            center_green.append(get_xy(y))
            rt = y - C@U0
            # Get dif error
            s = np.linalg.inv(((C@S1)@(C.T)) + R)
            k = S1@(C.T)@s
            # Update miu and sigma
            U0 = U1 + k@rt
            S0 = (C - k@C)@S1
            e.plot_ellipse(image, U0[:2],  S0[:2, :2], blue)
            center_blue.append(get_xy(U0))

    # First detection
    elif len(rectangles) > 0:
        U0 = get_vec(rectangles)
        first_time = False

    # draw the original bounding boxes
    for (x, y, w, h) in rectangles:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(image, (x+pad_w, y+pad_h),
                      (x+w-pad_w, y+h-pad_h), green, 3)

    # Sleep and show image
    cv2.imshow("After NMS", image)
    cv2.waitKey(24)

cv2.destroyAllWindows()


# %%
# Create plot
fig = plt.figure()
ax = fig.subplots(1, 1)

plt.ion()
data = (center_red, center_green, center_blue)
# colors = ("red", "green", "blue")
colors = ('r--', 'g-', 'bs')
groups = ("Prediction", "Detection", "Update")

for cent, color, group in zip(data, colors, groups):
    x_only, y_only = zip(*cent)
    ax.plot(x_only, y_only, color,  alpha=0.8, label=group)

plt.title('Kalman Kilter')
plt.legend(loc=2)
plt.savefig("./plots/kalman_filter(dr="+str(dr) +
            ", dq="+str(dq)+", dt="+str(dt)+".png")
plt.show()


# %%
