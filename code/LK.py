import math
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import os
import time


# HOG + SVM pedestrian detection
def pedestrian_detection(frame):
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    region = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    points = np.array([[x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2)] for (x1, y1, x2, y2) in region])
    return points


# generate pyramid
def pyramid_generator(img_old, img_new, levels):
    # generate Image Pyramids
    pyramid_old = []
    pyramid_new = []
    pyramid_old.append(img_old)
    pyramid_new.append(img_new)
    origin_old = img_old.copy()
    origin_new = img_new.copy()
    for level in range(levels - 1):
        # gaussian blur
        image_blur_old = cv2.GaussianBlur(origin_old, (5, 5), 1)
        image_blur_new = cv2.GaussianBlur(origin_new, (5, 5), 1)

        image_small_old = cv2.resize(image_blur_old, (image_blur_old.shape[1] // 2, image_blur_old.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        image_small_new = cv2.resize(image_blur_new, (image_blur_new.shape[1] // 2, image_blur_new.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        pyramid_old.append(image_small_old)
        pyramid_new.append(image_small_new)
        origin_old = image_small_old
        origin_new = image_small_new

    return pyramid_old, pyramid_new


# check whether the new detected pedestrian have already existed by
# calculating the distance between old points and new points
def checkpoints(points_old, points_new):
    points_temp = points_old
    for i, point_new in enumerate(points_new):
        check = 0
        point_num = points_old.shape[0]
        if point_num > 0:
            for j, point_old in enumerate(points_old):
                distance = math.sqrt((point_old[0] - point_new[0]) ** 2 + (point_old[1] - point_new[1]) ** 2)
                if distance > 40:
                    check += 1
            if check == point_num:
                points_temp = np.concatenate((points_temp, np.array([point_new])), axis=0)
        else:
            points_temp = points_new

    return points_temp


def lucasKanade(img_old, img_new, window_size, points, pyramid_level, iterative, residual):
    # generate Image Pyramids
    pyramid_old, pyramid_new = pyramid_generator(img_old, img_new, pyramid_level)

    # overall optical flow
    flow_overall = np.zeros(points.shape, dtype=np.float64)

    # iterate through pyramid
    for level in range(pyramid_level - 1, -1, -1):
        img_old = pyramid_old[level].astype(np.float64)
        img_new = pyramid_new[level].astype(np.float64)

        current_points = np.round((points / (2 ** level))).astype(np.int16)

        # calculate gradient x, gradient y
        Ix_img = cv2.Sobel(img_old, cv2.CV_64F, 1, 0, ksize=3)
        Iy_img = cv2.Sobel(img_old, cv2.CV_64F, 0, 1, ksize=3)

        # matrix for [IxIx, IxIy, IxIy, IyIy], A variable refer to slide on Lecture 6
        A = np.zeros((2, 2))
        # matrix for [IxIt, IyIt], b variable refer to slide on Lecture 6
        b = np.zeros((2,))

        for i, point in enumerate(current_points):
            x = point[0]
            y = point[1]

            # calculate the left matrix
            offset = int(window_size / 2)
            A[0, 0] = np.sum((Ix_img[y - offset:y + offset + 1, x - offset:x + offset + 1]) ** 2)
            A[0, 1] = np.sum((Ix_img[y - offset:y + offset + 1, x - offset:x + offset + 1]) * (Iy_img[y - offset:y + offset + 1, x - offset:x + offset + 1]))
            A[1, 0] = np.sum((Ix_img[y - offset:y + offset + 1, x - offset:x + offset + 1]) * (Iy_img[y - offset:y + offset + 1, x - offset:x + offset + 1]))
            A[1, 1] = np.sum((Iy_img[y - offset:y + offset + 1, x - offset:x + offset + 1]) ** 2)

            # iterative within one pyramid level
            flow_iterative = np.zeros(points.shape, dtype=np.float64)
            for k in range(iterative):
                # calculate It
                mat_translation = np.array([[1, 0, -flow_overall[i][0] - flow_iterative[i][0]], [0, 1, -flow_overall[i][1] - flow_iterative[i][1]]])
                img_translate = cv2.warpAffine(img_new, mat_translation, (img_new.shape[1], img_new.shape[0]), flags=cv2.INTER_LINEAR)
                It_img = img_old - img_translate

                b[0] = np.sum((Ix_img[y - offset:y + offset + 1, x - offset:x + offset + 1]) * (It_img[y - offset:y + offset + 1, x - offset:x + offset + 1]))
                b[1] = np.sum((Iy_img[y - offset:y + offset + 1, x - offset:x + offset + 1]) * (It_img[y - offset:y + offset + 1, x - offset:x + offset + 1]))

                A_inverse = np.linalg.pinv(A)
                d = np.matmul(b, A_inverse)

                flow_iterative[i] = d + flow_iterative[i]

                # if residual smaller than the threshold, break the iterative process to speed up
                if np.linalg.norm(d) < residual:
                    break

            # initialize optical flow in the next level
            d = flow_iterative[i]
            if level > 0:
                flow_overall[i] = 2 * (d + flow_overall[i])
            else:
                flow_overall[i] = d + flow_overall[i]

    points = np.round(points + flow_overall).astype(np.int16)
    return points


def main(manul_points, videoPath, pedestrian_flag):
    start = time.time()

    cap = cv2.VideoCapture(videoPath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter('./output/keyboard_1.mp4', fourcc, fps, (width, height))

    # read old frame
    ret, frame_old = cap.read()
    gray_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)

    # mode selection: pedestrian detection or manual points or both
    if pedestrian_flag:
        points_old = pedestrian_detection(frame_old).astype(np.int16)
        if len(points_old) > 0:
            if len(manul_points) > 0:
                points_old = np.concatenate((points_old, manul_points), axis=0).astype(np.int16)
        else:
            if len(manul_points) > 0:
                points_old = manul_points.astype(np.int16)
            else:
                points_old = np.array([])
    elif not pedestrian_flag:
        if len(manul_points) > 0:
            points_old = manul_points.astype(np.int16)

    # storing the feature points in every frame
    points_list = np.array([points_old])

    # optical flow tracking start
    frame_num = 0
    while cap.isOpened():
        ret, frame_new = cap.read()
        print("[INFO] frame: ", frame_num)
        if not ret:
            break

        # read new frame
        gray_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)

        # recapture the pedestrian feature points every second
        if frame_num % fps == (fps - 1) and pedestrian_flag == 1:
            # re-detect pedestrian
            points_new = pedestrian_detection(frame_new).astype(np.int16)
            point_num_old = len(points_old)
            points_old = checkpoints(points_old, points_new)
            point_num_new = len(points_old)

            # if there are additional pedestrian points
            # add the additional feature points to the points_list
            if point_num_new > point_num_old:
                if point_num_old == 0:
                    points_list = np.array([points_old])
                else:
                    points_list_new = np.zeros((points_list.shape[0], point_num_new, 2), dtype=np.int16) - 1
                    points_list_new[:, 0:points_list.shape[1], :] = points_list
                    points_list_new[-1, :, :] = points_old
                    points_list = points_list_new

        # calculate the movement of the feature points
        points_new = lucasKanade(gray_old, gray_new, window_size=15, points=points_old, pyramid_level=4, iterative=20, residual=0.0005)

        # update the movement of the feature points
        if len(points_new) > 0:
            points_list = np.concatenate((points_list, np.array([points_new])), axis=0)

        # remove points close to the boundary
        final_frame = points_list[-1]
        delete_point_index = []
        for i, xy in enumerate(final_frame):
            last_x = xy[0]
            last_y = xy[1]
            if last_x > width - 15 or last_x < 15 or last_y > height - 15 or last_y < 15:
                delete_point_index.append(i)
        if len(delete_point_index) > 0:
            for i in range(len(delete_point_index)):
                index = delete_point_index[i]
                points_list = np.delete(points_list, index - i, 1)
                points_new = np.delete(points_new, index - i, 0)

        # draw the movement
        lines = np.zeros_like(frame_old)
        frame_show = frame_new.copy()
        for i in range(points_list.shape[1]):
            points = points_list[:, i, :]
            frame_number = points.shape[0]
            for j in range(frame_number - 1):
                x_old, y_old = points[j].ravel()
                x_new, y_new = points[j + 1].ravel()
                if x_old > 0 and y_old > 0 and x_new > 0 and y_new > 0:
                    if j == frame_number - 2:
                        frame_show = cv2.circle(frame_show, (x_new, y_new), 5, (255, 0, 0), -1)
                    lines = cv2.line(lines, (x_new, y_new), (x_old, y_old), (0, 255, 0), 2)
                    frame_show = cv2.add(frame_show, lines)

        cv2.imshow('frame', frame_show)
        cv2.waitKey(fps)
        output.write(frame_show)

        # update the new frame as old frame
        gray_old = gray_new
        points_old = points_new
        frame_num = frame_num + 1

    cv2.destroyAllWindows()
    cap.release()
    output.release()

    end = time.time()
    print("[INFO] ----------finish----------")
    print("[INFO] find the output video in the output folder")
    print("Running time: %s Seconds" % (end - start))
    os._exit(1)

