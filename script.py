import math

# import Q1

import os
import random

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from cv2 import drawMatchesKnn as drawMatchesTwoImages


def algebraic_distance(pts1, pts2, F):
    n = len(pts1)
    sum = 0
    for i in range(len(pts1)):
        sum += np.matmul(np.matmul(np.transpose(pts2[i]), F), pts1[i])
    return abs(sum / n)


def epipolar_distance(pts1, pts2, F):
    sum1 = 0
    sum2 = 0
    for i in range(len(pts1)):
        Fx = np.matmul(F, pts1[i])
        Fxt = np.transpose(Fx)
        a = Fxt[0]
        b = Fxt[1]
        sum1 += (np.matmul(np.transpose(pts2[i]), Fx) / math.sqrt(a ** 2 + b ** 2)) ** 2

    Ft = np.transpose(F)
    for i in range(len(pts2)):
        Fx = np.matmul(Ft, pts2[i])
        Fxt = np.transpose(Fx)
        a = Fxt[0]
        b = Fxt[1]
        sum2 += (np.matmul(np.transpose(pts1[i]), Fx) / math.sqrt(a ** 2 + b ** 2)) ** 2

    return (sum1 + sum2) / len(pts1)


def getPoints(img1, img2):
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    # grey_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    # grey_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    BF = cv.BFMatcher()
    matches = BF.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.9 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return pts1, pts2


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):

        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def findFundamentalMatrix(pts1, pts2, method=cv.FM_8POINT):
    if method == cv.FM_7POINT:
        indexes = [i for i in range(len(pts1))]
        taken = random.sample(indexes, 7)
        pts1 = [pts1[i] for i in taken]
        pts2 = [pts2[i] for i in taken]

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, method)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return F[:3], pts1, pts2


def findAndDrawEpilines(img1, img2, F, pts1, pts2):
    new_pts2 = [[p[0], p[1], 1] for p in pts2]
    #lines1 = np.matmul(new_pts2, F)

    lines1 = cv.computeCorrespondEpilines(pts2,2,F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    new_pts1 = [[p[0], p[1], 1] for p in pts1]
    #lines2 = np.matmul(new_pts1, F)
    lines2 = cv.computeCorrespondEpilines(pts1,1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    cv.imshow("1",img5)
    cv.imshow("2",img3)
    cv.waitKey(0)
    # plt.subplot(121), plt.imshow(img5)
    # plt.subplot(122), plt.imshow(img3)
    # plt.show()
    return new_pts1, new_pts2, F


def run_script(img1, img2, pts1, pts2, method=cv.FM_8POINT):
    #pts1, pts2 = getPoints(img1, img2)
    F, pts1, pts2 = findFundamentalMatrix(pts1[:25], pts2[:25], method)
    return findAndDrawEpilines(img1, img2, F, pts1, pts2)


if __name__ == "__main__":
    print(os.listdir("./input/"))

    img1Path = "./input/im_family_00084_left.jpg"
    img2Path = "./input/im_family_00100_right.jpg"
    pts1 = [(134.33870967741933, 246.17290322580652), (213.82258064516128, 307.2051612903226),
            (460.7903225806452, 149.65677419354836), (567.241935483871, 300.1083870967743),
            (624.016129032258, 172.36645161290323), (855.3709677419354, 209.26967741935493),
            (870.983870967742, 314.30193548387103), (122.9838709677419, 50.30193548387092)]
    pts2 = [(58.209677419354875, 249.0116129032259), (109.30645161290317, 307.2051612903226),
            (349.1774193548388, 126.9470967741936), (554.9838709677417, 335.59225806451616),
            (701.1774193548388, 152.49548387096775), (803.3709677419354, 206.43096774193555),
            (864.4032258064515, 417.9148387096775), (579.1129032258066, 34.68903225806457)]
    img1Path = "./input/im_courtroom_00086_left.jpg"
    img2Path = "./input/im_courtroom_00089_right.jpg"
    pts1 = [(164.14516129032256, 253.26967741935493), (280.53225806451616, 128.36645161290323),
            (435.2419354838709, 112.75354838709677), (524.6612903225807, 143.9793548387097),
            (541.6935483870968, 67.33419354838702), (614.0806451612902, 233.39870967741945),
            (741.8225806451612, 197.9148387096775), (699.241935483871, 412.23741935483883)]
    pts2 = [(295.241935483871, 190.81806451612908), (313.6935483870968, 88.62451612903226),
            (414.4677419354839, 71.59225806451616), (486.8548387096773, 94.30193548387103),
            (496.7903225806451, 29.011612903225796), (567.758064516129, 165.26967741935493),
            (667.1129032258066, 132.62451612903226), (681.3064516129032, 261.785806451613)]
    img1 = cv.imread(img1Path,0)  # queryimage # left image
    img2 = cv.imread(img2Path,0)  # trainimage # right image


    pts1, pts2, F = run_script(img1, img2, pts1, pts2, method=cv.FM_7POINT)
    print(epipolar_distance(pts1, pts2, F))
    print(algebraic_distance(pts1, pts2, F))
