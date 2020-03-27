#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from util.constant import *


def get_plot_info(filename):
    dataset = open('{}'.format(filename), 'r').read().split('\n')
    dataset.pop(-1)

    information, instant, points, tuples = list(), list(), list(), list()

    for i in range(len(dataset)):
        string = dataset[i].split('[')
        # Info
        basicinfo = string[1].replace(" ", "").split(',')
        basicinfo.pop(-1)
        instant.append(basicinfo[0][1:-1])
        instant.append(basicinfo[1][1:-1])
        instant.append(int(basicinfo[2]))
        instant.append(int(basicinfo[3]))
        instant.append(int(basicinfo[4]))
        # Points
        string2 = string[2].replace(']', '')
        string3 = string2.replace('), ', ');')
        string4 = string3.split(';')

        for j in range(len(string4)):
            pts = string4[j].split(',')
            tupla = (float(pts[0][1:]), float(pts[1]), float(pts[2][:-1]))
            tuples.append(tupla)

        points.append(tuples)

        instant.append(tuples)
        tuples = []
        information.append(instant)
        instant = []

    return information


def plot3D(ptsoriginal, ptsbspline=None, ptsspatiotemporal=None, fig=None, ax=None, plot=True):
    point_size = 2

    xpts, ypts, zpts = list(), list(), list()
    for i in range(len(ptsoriginal)):
        xpts.append(ptsoriginal[i][0])
        ypts.append(ptsoriginal[i][1])
        zpts.append(ptsoriginal[i][2])
    X = np.asarray(xpts)
    Y = np.asarray(ypts)
    Z = np.asarray(zpts)
    ax.scatter(X, Y, Z, s=point_size, c='r', marker='o')

    if ptsbspline is not None and len(ptsbspline) > 0:
        xbspline, ybspline, zbspline = list(), list(), list()
        for i in range(len(ptsbspline)):
            xbspline.append(ptsbspline[i][0])
            ybspline.append(ptsbspline[i][1])
            zbspline.append(ptsbspline[i][2])
        Xbspline = np.asarray(xbspline)
        Ybspline = np.asarray(ybspline)
        Zbspline = np.asarray(zbspline)
        ax.scatter(Xbspline, Ybspline, Zbspline, s=point_size, c='b', marker='^')

    if ptsspatiotemporal is not None and len(ptsspatiotemporal) > 0:
        xst, yst, zst = list(), list(), list()
        for i in range(len(ptsspatiotemporal)):
            xst.append(ptsspatiotemporal[i][0])
            yst.append(ptsspatiotemporal[i][1])
            zst.append(ptsspatiotemporal[i][2])
        Xst = np.asarray(xst)
        Yst = np.asarray(yst)
        Zst = np.asarray(zst)
        ax.scatter(Xst, Yst, Zst, s=point_size, c='g', marker='X')

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    if plot:
        plt.show()


def euclidean_distance(p1, p2):
    # p1 = (5, 6, 7)
    # p2 = (8, 9, 9)
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
    # print("Euclidean distance from P1 to P2: ",distance)

    return distance


if __name__ == '__main__':
    side = 0
    rootsequence = 9
    imgnumber = 3

    filename_original = '{}/test_points/{}-{}.txt'.format(DIR_RESULT, rootsequence, imgnumber)
    filename_bspline = '{}/test_points/{}-{}-bspline.txt'.format(DIR_RESULT, rootsequence, imgnumber)
    filename_spatiotemporal = '{}/test_points/{}-{}-spatiotemporal.txt'.format(DIR_RESULT, rootsequence, imgnumber)

    dataset_original = get_plot_info(filename=filename_original)
    dataset_bspline = get_plot_info(filename=filename_bspline)
    dataset_spatiotemporal = get_plot_info(filename=filename_spatiotemporal)

    missingsagittalsequences, missingcoronalsequences = list(), list()
    pts_bspline_sagittal, pts_bspline_coronal = list(), list()
    pts_st_sagittal, pts_st_coronal = list(), list()

    for i in range(len(dataset_bspline)):
        if dataset_bspline[i][3] == 0 and dataset_bspline[i][4] == 2:
            missingsagittalsequences.append(dataset_bspline[i][2])
            pts_bspline_sagittal.append(dataset_bspline[i][5])

        elif dataset_bspline[i][3] == 0 and dataset_bspline[i][4] == 3:
            missingcoronalsequences.append(dataset_bspline[i][2])
            pts_bspline_coronal.append(dataset_bspline[i][5])

    print("Missing sagittal sequences: {} ({})".format(missingsagittalsequences, len(missingsagittalsequences)))
    print("Missing coronal sequences: {} ({})".format(missingcoronalsequences, len(missingcoronalsequences)))

    for i in range(len(dataset_spatiotemporal)):
        if dataset_spatiotemporal[i][3] == 0 and dataset_spatiotemporal[i][4] == 2:
            pts_st_sagittal.append(dataset_spatiotemporal[i][5])

        elif dataset_spatiotemporal[i][3] == 0 and dataset_spatiotemporal[i][4] == 3:
            pts_st_coronal.append(dataset_spatiotemporal[i][5])

    pts_original_sagittal, pts_original_coronal = list(), list()

    for i in range(len(dataset_original)):
        if dataset_original[i][2] == missingsagittalsequences[0] and dataset_original[i][4] == 2:
            pts_original_sagittal.append(dataset_original[i][5])
        elif dataset_original[i][2] == missingcoronalsequences[0] and dataset_original[i][4] == 3:
            pts_original_coronal.append(dataset_original[i][5])

    # Euclidean
    bsplinedistance, stdistance = list(), list()
    for i in range(60):
        bsplinedistance.append(euclidean_distance(pts_original_sagittal[0][i], pts_bspline_sagittal[0][i]))
        stdistance.append(euclidean_distance(pts_original_sagittal[0][i], pts_st_sagittal[0][i]))
    print(bsplinedistance)
    print("")
    print(stdistance)

    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.set_xlabel('X axis')
    axes.set_ylabel('Y axis')
    axes.set_zlabel('Z axis')

    plot3D(
        ptsoriginal=pts_original_sagittal[0],  # dataset_bspline[0][5]
        ptsbspline=pts_bspline_sagittal[0],
        ptsspatiotemporal=pts_st_sagittal[0],
        fig=figure,
        ax=axes,
        plot=True)

    # bsplinedistance, stdistance = list(), list()
    # for i in range(60):
    #     bsplinedistance.append(euclidean_distance(pts_original_coronal[0][i], pts_bspline_coronal[0][i]))
    #     stdistance.append(euclidean_distance(pts_original_coronal[0][i], pts_st_coronal[0][i]))
    # print(bsplinedistance)
    # print("")
    # print(stdistance)

    # figure = plt.figure()
    # axes = figure.add_subplot(111, projection='3d')
    # axes.set_xlabel('X axis')
    # axes.set_ylabel('Y axis')
    # axes.set_zlabel('Z axis')

    # plot3D(
    #     ptsoriginal=pts_original_coronal[0],
    #     ptsbspline=pts_bspline_coronal[0],
    #     ptsspatiotemporal=pts_st_coronal[0],
    #     fig=figure,
    #     ax=axes,
    #     plot=True)
