import cv2
import numpy as np
import math
from datetime import datetime
import sys


if len(sys.argv) < 5:
    raise IndexError("\nUse this command to run:\npy main.py 1.tif DD.MM.YYYY-HH:MM 2.tif DD.MM.YYYY-HH:MM")

img1_path = sys.argv[1]
img1_time = sys.argv[2]
img2_path = sys.argv[3]
img2_time = sys.argv[4]
# py main.py img/016a.tif 16.06.2024-8:51 img/017a.tif 17.06.2024-16:14

new_width = 800
# Read image 1 and resize
img1 = cv2.imread(img1_path)
original_height, original_width = img1.shape[:2]
aspect_ratio = new_width / original_width
new_height = int(original_height * aspect_ratio)
output = cv2.resize(img1, (new_width, new_height))

# Read image 2 and resize
img2 = cv2.imread(img2_path)
original_height, original_width = img2.shape[:2]
aspect_ratio = new_width / original_width
new_height = int(original_height * aspect_ratio)
output2 = cv2.resize(img2, (new_width, new_height))


img1_point = (0, 0)
img2_point = (0, 0)

center_1 = (0, 0)
center_2 = (0, 0)

sun_rad_1 = 0
sun_rad_2 = 0


# Detect Sun
def detect_sun(img):
    return cv2.HoughCircles(
        cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5),
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=100,
        minRadius=int(img.shape[0] * 0.25),
        maxRadius=int(img.shape[0] * 0.5)
    )


circle_1 = detect_sun(output)
circle_2 = detect_sun(output2)

# Draw first circle
if circle_1 is not None and circle_2 is not None:
    circle_1 = np.uint16(np.around(circle_1))
    circle_2 = np.uint16(np.around(circle_2))
    x, y, r = circle_1[0][0]
    sun_rad_1 = r
    center_1 = (x, y)
    print(f"First image with D: {2 * r}px")
    cv2.circle(output, center_1, r, (0, 255, 0), 2)  # Circle outline
    cv2.circle(output, center_1, 2, (0, 0, 255), 3)  # Center point
    x, y, r = circle_2[0][0]
    sun_rad_2 = r
    center_2 = (x, y)
    print(f"First image with D: {2 * r}px")
    cv2.circle(output2, center_2, r, (0, 255, 0), 2)  # Circle outline
    cv2.circle(output2, center_2, 2, (0, 0, 255), 3)  # Center point


# Set point
def set_point(event, x, y, flags, dialog):
    global img1_point, img2_point
    out1 = output.copy()
    out2 = output2.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"First point, img {dialog}: {x, y}")
        if dialog == 1:
            img1_point = (x, y)
        else:
            img2_point = (x, y)

    cv2.circle(out1, img1_point, 3, (0, 255, 0), -1)
    cv2.circle(out2, img2_point, 3, (0, 255, 0), -1)
    cv2.imshow('First image', out1)
    cv2.imshow('Second image', out2)


def overlay_images():
    # Create the translation matrix
    translation_matrix = np.float32([
        [1, 0, int(center_1[0]) - int(center_2[0])],
        [0, 1, int(center_1[1]) - int(center_2[1])]
    ])

    # Get the dimensions of the image
    height, width = output.shape[:2]

    # Apply translation
    img2_translated = cv2.warpAffine(output2, translation_matrix, (width, height))

    # Blend images
    blended = cv2.addWeighted(output, 0.5, img2_translated, 0.5, 0)

    cv2.circle(blended, img1_point, 3, (0, 255, 0), -1)
    cv2.circle(blended, img2_point, 3, (0, 255, 0), -1)
    cv2.circle(blended, center_1, 3, (0, 0, 255), -1)

    dst = math.sqrt((img2_point[0] - img1_point[0]) ** 2 + (img2_point[1] - img1_point[1]) ** 2)

    cv2.line(blended, img1_point, img2_point, (0, 250, 0), 2)

    cx, cy = center_1
    for i in range(1, 18):
        # latitude angle (avoid poles)
        phi = np.pi / 2 - (i * np.pi / 18)

        # vertical position
        y = int(cy + sun_rad_1 * np.sin(phi))

        # horizontal radius (ellipse width)
        rx = int(sun_rad_1 * np.cos(phi))

        # draw ellipse (lat line)
        cv2.ellipse(blended, (cx, y), (rx, 0), 0, 0, 360, (0, 0, 255), 2)

    L = circle_line_segment_length(center_1, sun_rad_1, img1_point, img2_point)

    print(f"Intersection length: {round(L, 2)} Distance: {dst}")

    T = (np.pi * L) / (dst / seconds_between(img1_time, img2_time))
    print(f"Period is {round(T, 0)} s or {round(T / 86400, 3)} days")

    blended = cv2.putText(blended, f"D:{round(L, 1)}, d:{round(dst, 2)}, T:{round(T / 86400, 3)}", (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0, 255, 0), 2, cv2.LINE_AA)
    return blended


def circle_line_segment_length(cen, r, start, end):
    xc, yc = float(cen[0]), float(cen[1])
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    r = float(r)
    # direction vector of line
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0

    # normalize direction
    dx /= length
    dy /= length

    # vector from line start to circle center
    fx, fy = xc - x1, yc - y1

    # projection of center onto line
    t0 = fx * dx + fy * dy
    px, py = x1 + t0 * dx, y1 + t0 * dy

    # distance from center to line
    d = math.hypot(xc - px, yc - py)

    if d > r:
        return 0.0

    # half chord length
    l = math.sqrt(r * r - d * d)

    return 2 * l


def calculate_parallel_line_endpoint(first, second, third, length, flip):
    x1, y1 = first
    x2, y2 = second
    x3, y3 = third

    dx = x2 - x1
    dy = y2 - y1

    ref_len = math.sqrt(dx ** 2 + dy ** 2)
    if ref_len == 0:
        raise ValueError("Reference line length cannot be zero.")

    ux = dx / ref_len
    uy = dy / ref_len

    x4 = x3 + length * ux
    y4 = y3 + length * uy
    if flip:
        x4 = 2 * x3 - x4
        y4 = 2 * y3 - y4

    return int(x4), int(y4)

def seconds_between(t1_str, t2_str):
    fmt = "%d.%m.%Y-%H:%M"
    t1 = datetime.strptime(t1_str, fmt)
    t2 = datetime.strptime(t2_str, fmt)
    diff = (t2 - t1).total_seconds()
    return diff


cv2.imshow('First image', output)
cv2.imshow('Second image', output2)
cv2.setMouseCallback('First image', set_point, 1)
cv2.setMouseCallback('Second image', set_point, 2)
cv2.waitKey(0)

# Calculate rotation angle
cv2.line(output, center_1, img1_point, (250, 0, 0), 3)
cv2.line(output2, center_2, img2_point, (250, 0, 0), 3)

cv2.imshow('First image', output)
cv2.imshow('Second image', output2)

# Overlay images with 50% opacity
result = overlay_images()

cv2.imshow("Result", result)
cv2.imwrite("result.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()