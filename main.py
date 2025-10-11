import cv2
import numpy as np
import math
from datetime import datetime

# Read image
t1 = "20.05.2024 9:26"
img1 = cv2.imread("data/img/20.png") # 06.04 5:44
output = img1.copy()

t2 = "21.05.2024 8:15"
img2 = cv2.imread("data/img/21.png") # 06.05 6:15
output2 = img2.copy()

img1_first_point = (0, 0)
img1_second_point = (0, 0)
img2_first_point = (0, 0)
img2_second_point = (0, 0)

center_1 = (0,0)
center_2 = (0,0)

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
        minRadius=int(img.shape[0]*0.25),
        maxRadius=int(img.shape[0]*0.5)
    )


circle_1 = detect_sun(img1)
circle_2 = detect_sun(img2)

# Draw first circle
if circle_1 is not None and circle_2 is not None:
    circle_1 = np.uint16(np.around(circle_1))
    circle_2 = np.uint16(np.around(circle_2))
    x, y, r = circle_1[0][0]
    sun_rad_1 = r
    center_1 = (x,y)
    print(f"First image with D: {2*r}px")
    cv2.circle(output, center_1, r, (0, 255, 0), 2)  # Circle outline
    cv2.circle(output, center_1, 2, (0, 0, 255), 3)  # Center point
    x, y, r = circle_2[0][0]
    sun_rad_2 = r
    center_2 = (x,y)
    print(f"First image with D: {2 * r}px")
    cv2.circle(output2, center_2, r, (0, 255, 0), 2)  # Circle outline
    cv2.circle(output2, center_2, 2, (0, 0, 255), 3)  # Center point


# Set point
def set_point(event, x, y, flags, dialog):
    global img1_first_point, img1_second_point, img2_first_point, img2_second_point
    out1 = output.copy()
    out2 = output2.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"First point, img {dialog}: {x, y}")
        if dialog == 1:
            img1_first_point = (x, y)
        else:
            img2_first_point = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Second point, img {dialog}: {x, y}")
        if dialog == 1:
            img1_second_point = (x, y)
        else:
            img2_second_point = (x, y)

    cv2.circle(out1, img1_first_point, 3, (0, 255, 0), -1)
    cv2.circle(out2, img2_first_point, 3, (0, 255, 0), -1)
    cv2.circle(out1, img1_second_point, 3, (255, 255, 0), -1)
    cv2.circle(out2, img2_second_point, 3, (255, 255, 0), -1)
    cv2.imshow('First image', out1)
    cv2.imshow('Second image', out2)


def overlay_images(img1, img2, cen1, cen2, point1, point2, angle, out=False):
    # Create the translation matrix
    translation_matrix = np.float32([
        [1, 0, int(cen1[0])-int(cen2[0])],
        [0, 1, int(cen1[1])-int(cen2[1])]
    ])

    # Get the dimensions of the image
    height, width = img1.shape[:2]

    # Apply translation
    img2_translated = cv2.warpAffine(img2, translation_matrix, (width, height))

    # Perform the rotation
    
    rotation_matrix = cv2.getRotationMatrix2D(cen1, angle, (sun_rad_1/sun_rad_2))
    img2_rotated = cv2.warpAffine(img2_translated, rotation_matrix, (width, height))

    # Blend images
    blended = cv2.addWeighted(img1, 0.5, img2_rotated, 0.5, 0)

    r1, theta1 = rect_to_pol(int(img2_second_point[0]) - int(cen2[0]), int(img2_second_point[1] - int(cen2[1])))
    r2, theta2 = rect_to_pol(int(point2[0])-int(cen2[0]), int(point2[1]-int(cen2[1])))

    p2 = pol_to_rect(r2, theta2-math.radians(angle))
    p2 = (int(p2[0])+int(cen1[0]),int(p2[1])+int(cen1[1]))

    p1 = pol_to_rect(r1, theta1-math.radians(angle))
    p1 = (int(p1[0]) + int(cen1[0]), int(p1[1]) + int(cen1[1]))

    # print(r1, theta1*180/3.1415)
    # print(r2, (theta2*180/3.1415)+angle)

    cv2.circle(blended, point1, 3, (0, 255, 0), -1)
    cv2.circle(blended, p2, 3, (0, 255, 0), -1)
    cv2.circle(blended, img1_second_point, 3, (255, 255, 0), -1)
    cv2.circle(blended, p1, 3, (255, 255, 0), -1)
    cv2.circle(blended, cen1, 3, (0, 0, 255), -1)

    distance1 = math.sqrt((p2[0]-point1[0])**2+(p2[1]-point1[1])**2)
    # distance2 = math.sqrt((p1[0] - img1_second_point[0]) ** 2 - (p1[1] - img1_second_point[1]) ** 2)

    a1 = rect_to_pol(int(point1[0])-int(p2[0]),int(point1[1])-int(p2[1]))[1]
    a2 = rect_to_pol(int(img1_second_point[0]) - int(p1[0]), int(img1_second_point[1]) - int(p1[1]))[1]

    # eq_x, eq_y = pol_to_rect(sun_rad_1, theta2- math.radians(angle))
    eq_x, eq_y = calculate_parallel_line_endpoint(p2, point1, cen1, sun_rad_1, False)

    cv2.line(blended, (eq_x, eq_y), calculate_parallel_line_endpoint(p2, point1, (eq_x, eq_y), 2*sun_rad_1, True), (0, 0, 255), 3)
    cv2.line(blended, point1, p2, (0, 250, 0), 2)
    cv2.line(blended, img1_second_point, p1, (255, 255, 0), 2)

    rotation_matrix = cv2.getRotationMatrix2D(cen1, math.degrees(a1), 1)
    blended = cv2.warpAffine(blended, rotation_matrix, (width, height))

    cx, cy = cen1
    for i in range(1, 18):
        # latitude angle (avoid poles)
        phi = np.pi / 2 - (i * np.pi / (18))

        # vertical position
        y = int(cy + sun_rad_1 * np.sin(phi))

        # horizontal radius (ellipse width)
        rx = int(sun_rad_1 * np.cos(phi))

        # draw ellipse (lat line)
        cv2.ellipse(blended, (cx, y), (rx, 0), 0, 0, 360, (0,0,255), 2)

    L = circle_line_segment_length(cen1, sun_rad_1, point1, p2)
    if out:
        print("Intersection length:", round(L,2))

        T = (np.pi*L)/(distance1/seconds_between(t1, t2))
        print(f"Period is {round(T,0)} s or {round(T/86400,3)} days")

        blended = cv2.putText(blended, f"D:{round(L,1)}, d:{round(distance1,2)}, T:{round(T/86400,3)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2, cv2.LINE_AA)
        # print(math.degrees(a1), math.degrees(a2), math.degrees(theta1), math.degrees(theta2), angle)

    return blended, a1, a2, distance1


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
    fx, fy = (xc) - (x1), (yc) - (y1)

    # projection of center onto line
    t0 = fx * dx + fy * dy
    px, py = x1 + t0 * dx, y1 + t0 * dy

    # distance from center to line
    d = math.hypot((xc) - px, (yc) - py)

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
        x4 = 2*x3-x4
        y4 = 2*y3-y4

    return int(x4), int(y4)

def pol_to_rect(r, th):
    return int(r * math.cos(th)), int(r * math.sin(th))

def rect_to_pol(x, y):
    return math.sqrt(x ** 2 + y ** 2), math.atan(y / x)

def seconds_between(t1_str, t2_str):
    fmt = "%d.%m.%Y %H:%M"
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
cv2.line(output, center_1, img1_first_point, (250, 0, 0), 3)
cv2.line(output2, center_2, img2_first_point, (250, 0, 0), 3)

cv2.imshow('First image', output)
cv2.imshow('Second image', output2)

# rotated_img2 = rotate_image(img2, angle)
# # Overlay images with 50% opacity

min_angle = math.inf
min_i = 0
step = 5
i = -10
n_iter = 0
while i < 20:
    result, a1, a2, dst = overlay_images(img1, img2, center_1, center_2, img1_first_point, img2_first_point, i)
    dangle =math.fabs(math.degrees(a1) - math.degrees(a2))
    # print(i, dangle, step)
    if dangle < min_angle:
        min_angle = dangle
        min_i = i

    if dangle > min_angle:
        step = -step/2
        min_angle = dangle
        n_iter = 0

    if dangle < 0.05 or n_iter > 25:
        min_i = i
        break

    i += step
    n_iter+=1

result, a1, a2, dst = overlay_images(img1, img2, center_1, center_2, img1_first_point, img2_first_point, min_i, True)
cv2.imshow("Result", result)
print(f"dAngle: {round(min_angle,5)}, Angle: {round(min_i, 3)} Distance: {round(dst, 3)}")
cv2.imwrite("result.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()