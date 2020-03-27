#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import operator
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce

# import reg
from results.read_results import *
from util.constant import *
# from plot import PointCloud

import matplotlib.tri as mtri
from scipy.spatial import Delaunay

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline


def read(side, rootsequence, imgnumber):
    """ Read text file that contain information about point cloud """
    if side == 0:
        dataset = open(
            '{}/{}-{}.txt'.format(
                DIR_RESULT_INTERPOLATED_ST_LEFT, rootsequence, imgnumber),
            'r').read().split('\n')
    else:
        dataset = open(
            '{}/{}-{}.txt'.format(
                DIR_RESULT_INTERPOLATED_ST_RIGHT, rootsequence, imgnumber),
            'r').read().split('\n')
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


def point_cloud(X, Y, Z, size=1, color='#FF3232', bordercolor='#FF3232', legend='', width=0.5, opacity=1.0):
    """
    X, Y, Z: Lists
        Represents X, Y and Z values respectively
    size: int
        Point's size
    color: string
        Point's color
    bordercolor: string
        Border color of points
    legend: string
    width: float
    opacity: float
        Opacity of the trace. The value range from 0 to 1 (default value is 1) """
    point_cloud = go.Scatter3d(
        x=X,
        y=Y,
        z=Z,
        # showlegend=False,
        name=legend,
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            line=dict(
                color=bordercolor,
                width=width
            ),
            # opacity=opacity
        )
    )

    return point_cloud


def plot3D(pts, fig=None, ax=None, color='r', size=2, opt=0):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    xpts, ypts, zpts = list(), list(), list()
    for i in range(len(pts)):
        xpts.append(pts[i][0])
        ypts.append(pts[i][1])
        zpts.append(pts[i][2])

    X = np.asarray(xpts)
    Y = np.asarray(ypts)
    Z = np.asarray(zpts)

    if opt == 0:
        ax.scatter(X, Y, Z, s=size, c=color, marker='o')
    elif opt == 2:
        # Triangulate parameter space to determine the triangles
        tri = Delaunay(np.array([X, Y]).T)
        ax.plot_trisurf(X, Y, Z, triangles=tri.simplices, cmap=plt.cm.bone, linewidth=0.2, color='b', edgecolor='Black')

        # Plot the triangulation.
        # triang = mtri.Triangulation(X, Y)
        # plt.figure()
        # plt.gca().set_aspect('equal')
        # plt.triplot(triang, 'bo-', lw=1)
        # plt.title('triplot of Delaunay triangulation')
    elif opt == 3:
        pass

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    # plt.show()


def plotColor3D(pts_step1=None, pts_step2=None, pts_step3=None, color1='r', color2='b', color3='g', size=2):
    # Colors: bgrcmyk
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # STEP 1
    if pts_step1 is not None:
        xpts1, ypts1, zpts1 = list(), list(), list()
        for i in range(len(pts_step1)):
            xpts1.append(pts_step1[i][0])
            ypts1.append(pts_step1[i][1])
            zpts1.append(pts_step1[i][2])

        X1 = np.asarray(xpts1)
        Y1 = np.asarray(ypts1)
        Z1 = np.asarray(zpts1)
        ax.scatter(X1, Y1, Z1, s=size, c=color1, marker='o')

    # STEP 2
    if pts_step2 is not None:
        xpts2, ypts2, zpts2 = list(), list(), list()
        for i in range(len(pts_step2)):
            xpts2.append(pts_step2[i][0])
            ypts2.append(pts_step2[i][1])
            zpts2.append(pts_step2[i][2])

        X2 = np.asarray(xpts2)
        Y2 = np.asarray(ypts2)
        Z2 = np.asarray(zpts2)
        ax.scatter(X2, Y2, Z2, s=size, c=color2, marker='o')

    # STEP 3
    if pts_step3 is not None:
        xpts3, ypts3, zpts3 = list(), list(), list()
        for i in range(len(pts_step3)):
            xpts3.append(pts_step3[i][0])
            ypts3.append(pts_step3[i][1])
            zpts3.append(pts_step3[i][2])

        X3 = np.asarray(xpts3)
        Y3 = np.asarray(ypts3)
        Z3 = np.asarray(zpts3)
        ax.scatter(X3, Y3, Z3, s=size, c=color3, marker='o')

    pts_steps = list()
    if pts_step1 is not None:
        pts_steps.append(pts_step1)
    if pts_step2 is not None:
        pts_steps.append(pts_step2)
    if pts_step3 is not None:
        pts_steps.append(pts_step3)
    pts = list(itertools.chain.from_iterable(pts_steps))

    xpts, ypts, zpts = list(), list(), list()
    for i in range(len(pts)):
        xpts.append(pts[i][0])
        ypts.append(pts[i][1])
        zpts.append(pts[i][2])

    X = np.asarray(xpts)
    Y = np.asarray(ypts)
    Z = np.asarray(zpts)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()


def plotColor3Dboth(pts_step1_left, pts_step2_left, pts_step3_left, pts_step1_right, pts_step2_right, pts_step3_right):
    color1, color2, color3 = 'r', 'b', 'c'
    size = 2
    # Colors: bgrcmyk
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # STEP 1 - LEFT
    lxpts1, lypts1, lzpts1 = list(), list(), list()
    for i in range(len(pts_step1_left)):
        lxpts1.append(pts_step1_left[i][0])
        lypts1.append(pts_step1_left[i][1])
        lzpts1.append(pts_step1_left[i][2])

    lX1 = np.asarray(lxpts1)
    lY1 = np.asarray(lypts1)
    lZ1 = np.asarray(lzpts1)
    ax.scatter(lX1, lY1, lZ1, s=size, c=color1, marker='o')

    # STEP 1 - RIGHT
    rxpts1, rypts1, rzpts1 = list(), list(), list()
    for i in range(len(pts_step1_right)):
        rxpts1.append(pts_step1_right[i][0])
        rypts1.append(pts_step1_right[i][1])
        rzpts1.append(pts_step1_right[i][2])

    rX1 = np.asarray(rxpts1)
    rY1 = np.asarray(rypts1)
    rZ1 = np.asarray(rzpts1)
    ax.scatter(rX1, rY1, rZ1, s=size, c=color1, marker='o')

    # STEP 2 - LEFT
    lxpts2, lypts2, lzpts2 = list(), list(), list()
    for i in range(len(pts_step2_left)):
        lxpts2.append(pts_step2_left[i][0])
        lypts2.append(pts_step2_left[i][1])
        lzpts2.append(pts_step2_left[i][2])

    lX2 = np.asarray(lxpts2)
    lY2 = np.asarray(lypts2)
    lZ2 = np.asarray(lzpts2)
    ax.scatter(lX2, lY2, lZ2, s=size, c=color2, marker='o')

    # STEP 2 - RIGHT
    rxpts2, rypts2, rzpts2 = list(), list(), list()
    for i in range(len(pts_step2_right)):
        rxpts2.append(pts_step2_right[i][0])
        rypts2.append(pts_step2_right[i][1])
        rzpts2.append(pts_step2_right[i][2])

    rX2 = np.asarray(rxpts2)
    rY2 = np.asarray(rypts2)
    rZ2 = np.asarray(rzpts2)
    ax.scatter(rX2, rY2, rZ2, s=size, c=color2, marker='o')

    # STEP 3 - LEFT
    lxpts3, lypts3, lzpts3 = list(), list(), list()
    for i in range(len(pts_step3_left)):
        lxpts3.append(pts_step3_left[i][0])
        lypts3.append(pts_step3_left[i][1])
        lzpts3.append(pts_step3_left[i][2])

    lX3 = np.asarray(lxpts3)
    lY3 = np.asarray(lypts3)
    lZ3 = np.asarray(lzpts3)
    ax.scatter(lX3, lY3, lZ3, s=size, c=color3, marker='o')

    # STEP 3 - RIGHT
    rxpts3, rypts3, rzpts3 = list(), list(), list()
    for i in range(len(pts_step3_right)):
        rxpts3.append(pts_step3_right[i][0])
        rypts3.append(pts_step3_right[i][1])
        rzpts3.append(pts_step3_right[i][2])

    rX3 = np.asarray(rxpts3)
    rY3 = np.asarray(rypts3)
    rZ3 = np.asarray(rzpts3)
    ax.scatter(rX3, rY3, rZ3, s=size, c=color3, marker='o')

    #
    pts_steps = list()
    pts_steps.append(pts_step1_left + pts_step1_right)
    pts_steps.append(pts_step2_left + pts_step2_right)
    pts_steps.append(pts_step3_left + pts_step3_right)
    pts = list(itertools.chain.from_iterable(pts_steps))

    xpts, ypts, zpts = list(), list(), list()
    for i in range(len(pts)):
        xpts.append(pts[i][0])
        ypts.append(pts[i][1])
        zpts.append(pts[i][2])

    X = np.asarray(xpts)
    Y = np.asarray(ypts)
    Z = np.asarray(zpts)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()


def euclidean_distance(p1, p2):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
    return distance


if __name__ == '__main__':
    try:
        mode = 0  # 0 - Triangulation
        patient = 'Matsushita'  # 'Iwasawa'
        rootsequence = 14  # 9
        side = 0  # 0 - left | 1 - right
        imgnumber = 7  # Instant
        # Iwasawa
        # coronalsequences = '9, 10, 11, 12'
        # leftsagittalsequences = '1, 2, 3, 4, 5, 6, 7, 8'
        # rightsagittalsequences = '13, 14, 15, 16, 17'
        # Matsushita
        coronalsequences = '4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21'
        leftsagittalsequences = '2, 3, 4, 5, 6, 7'
        rightsagittalsequences = '10, 11, 12, 13, 14, 15'

        txtargv2 = '-mode={}'.format(mode)
        txtargv3 = '-patient={}'.format(patient)
        txtargv4 = '-rootsequence={}'.format(rootsequence)
        txtargv5 = '-side={}'.format(side)
        txtargv6 = '-imgnumber={}'.format(imgnumber)
        txtargv7 = '-coronalsequences={}'.format(coronalsequences)
        txtargv8 = '-leftsagittalsequences={}'.format(leftsagittalsequences)
        txtargv9 = '-rightsagittalsequences={}'.format(rightsagittalsequences)

        if len(sys.argv) > 1:
            txtargv2 = sys.argv[1]
            if len(sys.argv) > 2:
                txtargv3 = sys.argv[2]
                if len(sys.argv) > 3:
                    txtargv4 = sys.argv[3]
                    if len(sys.argv) > 4:
                        txtargv5 = sys.argv[4]
                        if len(sys.argv) > 5:
                            txtargv6 = sys.argv[5]
                            if len(sys.argv) > 6:
                                txtargv7 = sys.argv[6]
                                if len(sys.argv) > 7:
                                    txtargv8 = sys.argv[7]
                                    if len(sys.argv) > 8:
                                        txtargv9 = sys.argv[8]

        txtargv =\
            '{}|{}|{}|{}|{}|{}|{}|{}'.format(
                txtargv2,
                txtargv3,
                txtargv4,
                txtargv5,
                txtargv6,
                txtargv7,
                txtargv8,
                txtargv9)

        if txtargv.find('-mode') != -1:
            txttmp = txtargv.split('-mode')[1]
            txttmp = txttmp.split('=')[1]
            mode = int(txttmp.split('|')[0])

        if txtargv.find('-patient') != -1:
            txttmp = txtargv.split('-patient')[1]
            txttmp = txttmp.split('=')[1]
            patient = txttmp.split('|')[0]

        if txtargv.find('-rootsequence') != -1:
            txttmp = txtargv.split('-rootsequence')[1]
            txttmp = txttmp.split('=')[1]
            rootsequence = int(txttmp.split('|')[0])

        if txtargv.find('-side') != -1:
            txttmp = txtargv.split('-side')[1]
            txttmp = txttmp.split('=')[1]
            side = int(txttmp.split('|')[0])

        if txtargv.find('-imgnumber') != -1:
            txttmp = txtargv.split('-imgnumber')[1]
            txttmp = txttmp.split('=')[1]
            imgnumber = int(txttmp.split('|')[0])

        if txtargv.find('-coronalsequences') != -1:
            txttmp = txtargv.split('-coronalsequences')[1]
            txttmp = txttmp.split('=')[1]
            txttmp = txttmp.split('|')[0]
            txttmp = txttmp.split(',')
            if len(coronalsequences) > 0:
                coronalsequences = [int(x) for x in txttmp]
            else:
                coronalsequences = []

        if txtargv.find('-leftsagittalsequences') != -1:
            txttmp = txtargv.split('-leftsagittalsequences')[1]
            txttmp = txttmp.split('=')[1]
            txttmp = txttmp.split('|')[0]
            txttmp = txttmp.split(',')
            if len(leftsagittalsequences) > 0:
                leftsagittalsequences = [int(x) for x in txttmp]
            else:
                leftsagittalsequences = []

        if txtargv.find('-rightsagittalsequences') != -1:
            txttmp = txtargv.split('-rightsagittalsequences')[1]
            txttmp = txttmp.split('=')[1]
            txttmp = txttmp.split('|')[0]
            txttmp = txttmp.split(',')
            if len(rightsagittalsequences) > 0:
                rightsagittalsequences = [int(x) for x in txttmp]
            else:
                rightsagittalsequences = []

    except ValueError:
        print(
            """
            Example of use:\n

            $ python {} -mode=0 -patient=Iwasawa -rootsequence=9 -side=0 -imgnumber=1
            -coronalsequences=9,10,11
            -leftsagittalsequences=1,2,3,4,5,6,7,8
            -rightsagittalsequences=13,14,15,16,17
            """.format(sys.argv[0]))
        exit()

    online = False

    # plot_info_e = get_plot_info('{}/test_points/left-9-1'.format(DIR_RESULT))
    # plot_info_d = get_plot_info('{}/test_points/right'.format(DIR_RESULT))
    # plot_info_e = get_plot_info('{}/{}-{}'.format(DIR_RESULT_INTERPOLATED_NURBS_LEFT, rootsequence, imgnumber))
    # plot_info_d = get_plot_info('{}/{}-{}'.format(DIR_RESULT_INTERPOLATED_NURBS_RIGHT, rootsequence, imgnumber))
    plot_info_e = get_plot_info('{}/{}-{}'.format(DIR_RESULT_INTERPOLATED_ST_LEFT, rootsequence, imgnumber))
    plot_info_d = get_plot_info('{}/{}-{}'.format(DIR_RESULT_INTERPOLATED_ST_RIGHT, rootsequence, imgnumber))

    lpoints1_e, lpoints2_e, lpoints3_e, lpoints_e = list(), list(), list(), list()
    lpoints1_d, lpoints2_d, lpoints3_d, lpoints_d = list(), list(), list(), list()

    # Left
    for i in range(len(plot_info_e)):
        if plot_info_e[i][4] == 1:
            lpoints1_e.append(plot_info_e[i][5])
        elif plot_info_e[i][4] == 2:
            lpoints2_e.append(plot_info_e[i][5])
        else:
            lpoints3_e.append(plot_info_e[i][5])
    # print("{} ({})\n".format(lpoints1_e, len(lpoints1_e)))
    # print("{} ({})\n".format(lpoints2_e, len(lpoints2_e)))
    # print("{} ({})\n".format(lpoints3_e, len(lpoints3_e)))
    lpoints1_e = list(itertools.chain.from_iterable(lpoints1_e))
    lpoints2_e = list(itertools.chain.from_iterable(lpoints2_e))
    lpoints3_e = list(itertools.chain.from_iterable(lpoints3_e))

    lpoints_e.append(lpoints1_e)
    # lpoints_e.append(lpoints2_e)
    lpoints_e.append(lpoints3_e)
    lpoints_e = list(itertools.chain.from_iterable(lpoints_e))

    # Right
    for i in range(len(plot_info_d)):
        if plot_info_d[i][4] == 1:
            lpoints1_d.append(plot_info_d[i][5])
        elif plot_info_d[i][4] == 2:
            lpoints2_d.append(plot_info_d[i][5])
        else:
            lpoints3_d.append(plot_info_d[i][5])
    # print("{} ({})\n".format(lpoints1_d, len(lpoints1_d)))
    # print("{} ({})\n".format(lpoints2_d, len(lpoints2_d)))
    # print("{} ({})\n".format(lpoints3_d, len(lpoints3_d)))
    lpoints1_d = list(itertools.chain.from_iterable(lpoints1_d))
    lpoints2_d = list(itertools.chain.from_iterable(lpoints2_d))
    lpoints3_d = list(itertools.chain.from_iterable(lpoints3_d))

    lpoints_d.append(lpoints1_d)
    # lpoints_d.append(lpoints2_d)
    lpoints_d.append(lpoints3_d)
    lpoints_d = list(itertools.chain.from_iterable(lpoints_d))

    '''
    0 - Colored points according to the stage in which the silhouette was registered
    1 - Attempt to create triangulation by dividing the lung into two parts
    2 - Just show the points cloud
    3 - Marching cubes (Still needs to do it)
    4 - Plot silhouettes in different colors using matplotlib
    5 - Plot point cloud of each lung
    6 - Plot with different colors silhouttes that were interpolated
    7 - Plot with different colors silhouettes of each step of registration
    8 - Both lungs using Plotly
    9 - Show points cloud using Plotly
    10 - Show points cloud and surface using Plotly
    11 - Plotly offline
    12 - Try plot online, if it is not possible, plot offline
    13 - Plot only left lung and test some layout options
    14 - Plot using plotly diaphragm points (extract from Abe's work) and wall lungs points (extract from Madalosso's work)
    15 - Plot using Plotly but remove some silhuouttes
    '''

    if mode == 0:
        if side == 0:
            plotColor3D(
                pts_step1=lpoints1_e,
                pts_step2=lpoints2_e,
                pts_step3=lpoints3_e)
        elif side == 1:
            plotColor3D(
                pts_step1=lpoints1_d,
                pts_step2=lpoints2_d,
                pts_step3=lpoints3_d)
        else:
            plotColor3Dboth(
                pts_step1_left=lpoints1_e,
                pts_step2_left=lpoints2_e,
                pts_step3_left=lpoints3_e,
                pts_step1_right=lpoints1_d,
                pts_step2_right=lpoints2_d,
                pts_step3_right=lpoints3_d)
    elif mode == 1:
        xminmax = list()
        for i in range(len(lpoints2_e)):
            xminmax.append(lpoints2_e[i][0])
        xmin, xmax = min(xminmax), max(xminmax)
        # print("All: {} | Min: {} | Max: {}\n".format(sorted(set(xminmax)), xmin, xmax))
        # print("Mean: {}\n".format(reduce(operator.add, xminmax) / len(xminmax)))
        mean = reduce(operator.add, xminmax) / len(xminmax)

        half1, half2 = list(), list()
        for i in range(len(lpoints_e)):
            if lpoints_e[i][0] <= mean:
                half1.append(lpoints_e[i])
            else:
                half2.append(lpoints_e[i])
        # print("Half 1: {} ({})\n".format(half1, len(half1)))
        # print("Half 2: {} ({})\n".format(half2, len(half2)))

        fig = plt.figure(figsize=plt.figaspect(0.4))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        plot3D(pts=half1, fig=fig, ax=ax, opt=2)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        plot3D(pts=half2, fig=fig, ax=ax, opt=2)
        plt.show()
    elif mode == 2:
        lpts_e = list()
        lpts_e.append(lpoints1_e)
        lpts_e.append(lpoints2_e)
        lpts_e.append(lpoints3_e)
        lpts_e = list(itertools.chain.from_iterable(lpts_e))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot3D(pts=lpts_e, fig=fig, ax=ax, opt=0)
        plt.show()
    elif mode == 3:
        print("Marching Cubes")
        lpts = list()
        lpts.append(lpoints1_e)
        lpts.append(lpoints2_e)
        lpts.append(lpoints3_e)
        lpts = list(itertools.chain.from_iterable(lpts))
        # print(len(lpts))

        xminmax, yminmax, zminmax = list(), list(), list()
        print(lpts[0])
        print(lpts[0][0])
        print(lpts[0][1])
        print(lpts[0][2])
        # for i in range(len(points2plot_rd)):
        #     yminmax.append(points2plot_rd[i][0][1])
        # ymin, ymax = min(yminmax), max(yminmax)
    elif mode == 4:
        if side == 0:
            plot_info_e = get_plot_info(
                '{}/{}-{}'.format(DIR_RESULT_INTERPOLATED_ST_LEFT, rootsequence, imgnumber))

            lpoints1, lpoints2, lpoints3 = list(), list(), list()

            # Left
            for i in range(len(plot_info_e)):
                if plot_info_e[i][4] == 1:
                    lpoints1.append(plot_info_e[i][5])
                elif plot_info_e[i][4] == 2:
                    lpoints2.append(plot_info_e[i][5])
                else:
                    lpoints3.append(plot_info_e[i][5])

            lpoints1 = list(itertools.chain.from_iterable(lpoints1))
            # lpoints2 = list(itertools.chain.from_iterable(lpoints2))
            lpoints3 = list(itertools.chain.from_iterable(lpoints3))

            # lpoints = list()
            # lpoints.append(lpoints1)
            # lpoints.append(lpoints2)
            # lpoints.append(lpoints3)
            # lpoints = list(itertools.chain.from_iterable(lpoints))
        elif side == 1:
            plot_info_d = get_plot_info(
                '{}/{}-{}'.format(DIR_RESULT_INTERPOLATED_ST_RIGHT, rootsequence, imgnumber))

            rpoints1, rpoints2, rpoints3 = list(), list(), list()

            # Right
            for i in range(len(plot_info_d)):
                if plot_info_d[i][4] == 1:
                    rpoints1.append(plot_info_d[i][5])
                elif plot_info_d[i][4] == 2:
                    rpoints2.append(plot_info_d[i][5])
                else:
                    rpoints3.append(plot_info_d[i][5])

            rpoints1 = list(itertools.chain.from_iterable(rpoints1))
            rpoints2 = list(itertools.chain.from_iterable(rpoints2))
            rpoints3 = list(itertools.chain.from_iterable(rpoints3))

            # rpoints = list()
            # rpoints.append(rpoints1)
            # rpoints.append(rpoints2)
            # rpoints.append(rpoints3)
            # rpoints = list(itertools.chain.from_iterable(rpoints))

        else:
            pass

        # L1 R16
        if side == 0:
            # plotColor3D(
            #     pts_step1=lpoints1,
            #     pts_step2=lpoints2,
            #     pts_step3=lpoints3)
            # plotColor3D(
            #     pts_step1=lpoints1,
            #     pts_step2=lpoints2[4],
            #     pts_step3=lpoints3)
            # plotColor3D(
            #     pts_step2=lpoints2[0] + lpoints2[1] + lpoints2[2] + lpoints2[4] + lpoints2[5] + lpoints2[6] + lpoints2[7])
            plotColor3D(
                pts_step1=lpoints1,
                pts_step2=lpoints2[0] + lpoints2[1] + lpoints2[2] + lpoints2[3] + lpoints2[4] + lpoints2[5] + lpoints2[6] + lpoints2[7])

        elif side == 1:
            plotColor3D(
                pts_step1=lpoints1_d,
                pts_step2=lpoints2_d,
                pts_step3=lpoints3_d)
            # plotColor3D(
            #     pts_step1=rpoints1,
            #     pts_step2=rpoints2,
            #     pts_step3=rpoints3)

        else:
            # plotColor3Dboth(
            #     pts_step1_left=lpoints1_e,
            #     pts_step2_left=lpoints2_e,
            #     pts_step3_left=lpoints3_e,
            #     pts_step1_right=lpoints1_d,
            #     pts_step2_right=lpoints2_d,
            #     pts_step3_right=lpoints3_d)
            plotColor3Dboth(
                pts_step1_left=lpoints1_e,
                pts_step2_left=lpoints2_e[0],
                pts_step3_left=lpoints3_e,
                pts_step1_right=lpoints1_d,
                pts_step2_right=lpoints2_d[0],
                pts_step3_right=lpoints3_d)
    elif mode == 5:
        def get_points(side, rootsequence, imgnumber):
            if side == 0:
                points = read(
                    side=0,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)
            else:
                points = read(
                    side=1,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            points1, points2, points3, all_points = list(), list(), list(), list()

            for i in range(len(points)):
                if points[i][4] == 1:
                    points1.append(points[i][5])
                elif points[i][4] == 2:
                    points2.append(points[i][5])
                else:
                    points3.append(points[i][5])

            points1 = list(itertools.chain.from_iterable(points1))
            points2 = list(itertools.chain.from_iterable(points2))
            points3 = list(itertools.chain.from_iterable(points3))

            all_points.append(points1)
            all_points.append(points2)
            all_points.append(points3)
            all_points = list(itertools.chain.from_iterable(all_points))

            return points1, points2, points3, all_points

        def separate_points_by_coordinate(pointsstep1, pointsstep2, pointsstep3):
            points = list()

            points.append(pointsstep1)
            points.append(pointsstep2)
            points.append(pointsstep3)
            points = list(itertools.chain.from_iterable(points))

            X, Y, Z = list(), list(), list()

            for i in range(len(points)):
                # print(points)
                X.append(points[i][0])
                Y.append(points[i][1])
                Z.append(points[i][2])

            X = tuple(X)
            Y = tuple(Y)
            Z = tuple(Z)

            return X, Y, Z

        if side == 0:
            left_points1, left_points2, left_points3, all_left_points =\
                get_points(
                    side=0,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            leftX, leftY, leftZ =\
                separate_points_by_coordinate(left_points1, left_points2, left_points3)

        elif side == 1:
            right_points1, right_points2, right_points3, all_right_points =\
                get_points(
                    side=1,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            rightX, rightY, rightZ =\
                separate_points_by_coordinate(right_points1, right_points2, right_points3)

        else:
            left_points1, left_points2, left_points3, all_left_points =\
                get_points(
                    side=0,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            leftX, leftY, leftZ =\
                separate_points_by_coordinate(left_points1, left_points2, left_points3)

            right_points1, right_points2, right_points3, all_right_points =\
                get_points(
                    side=1,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            rightX, rightY, rightZ =\
                separate_points_by_coordinate(right_points1, right_points2, right_points3)

        if side == 0:
            lpoint_cloud = point_cloud(
                X=leftX,
                Y=leftY,
                Z=leftZ,
                size=1,
                color='#999999',
                bordercolor='#999999',
                width=0.5,
                opacity=1.0)

            data = [lpoint_cloud]

        elif side == 1:
            rpoint_cloud = point_cloud(
                X=rightX,
                Y=rightY,
                Z=rightZ,
                size=1,
                color='#999999',
                bordercolor='#999999',
                width=0.5,
                opacity=1.0)

            data = [rpoint_cloud]

        else:
            lpoint_cloud = point_cloud(
                X=leftX,
                Y=leftY,
                Z=leftZ,
                size=1,
                color='#999999',
                bordercolor='#999999',
                width=0.5,
                opacity=1.0)

            rpoint_cloud = point_cloud(
                X=rightX,
                Y=rightY,
                Z=rightZ,
                size=1,
                color='#999999',
                bordercolor='#999999',
                width=0.5,
                opacity=1.0)

            data = [lpoint_cloud, rpoint_cloud]

        layout = go.Layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    title='X',  # X
                    # showgrid=False,
                    gridcolor="#adad85",
                    gridwidth=3,
                    color="#000000",
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                yaxis=dict(
                    title='Y',  # Y
                    # showgrid=False,
                    gridcolor="#adad85",
                    gridwidth=3,
                    color="#000000",
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                zaxis=dict(
                    title='Z',  # Z
                    # showgrid=False,
                    color="#000000",
                    gridcolor="#adad85",
                    gridwidth=3,
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),),
            margin=dict(l=0, r=0, b=0, t=0))

        figure = dict(data=data, layout=layout)

        offline.plot(figure)
    elif mode == 6:
        def separate(side, rootsequence, imgnumber):
            # Separate origial points from interpolated points
            if side == 0:
                points = read(
                    side=0,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)
            else:
                points = read(
                    side=1,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            origial_points, interpolated_points, all_points = list(), list(), list()

            for i in range(len(points)):
                if points[i][3] == 0:
                    interpolated_points.append(points[i][5])
                else:
                    origial_points.append(points[i][5])

            origial_points = list(itertools.chain.from_iterable(origial_points))
            interpolated_points = list(itertools.chain.from_iterable(interpolated_points))

            all_points.append(origial_points)
            all_points.append(interpolated_points)
            all_points = list(itertools.chain.from_iterable(all_points))

            return origial_points, interpolated_points, all_points

        def coordinate(original_points, interpolated_points):
            # Separate points by coordinate (x, y, z)
            originalpts, interpolatedpts = list(), list()

            originalpts.append(original_points)
            interpolatedpts.append(interpolated_points)

            originalpts = list(itertools.chain.from_iterable(originalpts))
            interpolatedpts = list(itertools.chain.from_iterable(interpolatedpts))

            originalX, originalY, originalZ = list(), list(), list()
            interpolatedX, interpolatedY, interpolatedZ = list(), list(), list()

            for i in range(len(originalpts)):
                originalX.append(originalpts[i][0])
                originalY.append(originalpts[i][1])
                originalZ.append(originalpts[i][2])

            originalX = tuple(originalX)
            originalY = tuple(originalY)
            originalZ = tuple(originalZ)

            for i in range(len(interpolatedpts)):
                interpolatedX.append(interpolatedpts[i][0])
                interpolatedY.append(interpolatedpts[i][1])
                interpolatedZ.append(interpolatedpts[i][2])

            interpolatedX = tuple(interpolatedX)
            interpolatedY = tuple(interpolatedY)
            interpolatedZ = tuple(interpolatedZ)

            return originalX, originalY, originalZ, interpolatedX, interpolatedY, interpolatedZ

        # patient | plan | sequence | image | step | points
        if side == 0:
            left_original_points, left_interpolated_points, all_left_points =\
                separate(
                    side=side,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            leftoriginalX, leftoriginalY, leftoriginalZ, leftinterpolatedX, leftinterpolatedY, leftinterpolatedZ =\
                coordinate(left_original_points, left_interpolated_points)

        elif side == 1:
            right_original_points, right_interpolated_points, all_right_points =\
                separate(
                    side=side,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            rightoriginalX, rightoriginalY, rightoriginalZ, rightinterpolatedX, rightinterpolatedY, rightinterpolatedZ =\
                coordinate(right_original_points, right_interpolated_points)

        elif side == 2:
            left_original_points, left_interpolated_points, all_left_points =\
                separate(
                    side=0,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            leftoriginalX, leftoriginalY, leftoriginalZ, leftinterpolatedX, leftinterpolatedY, leftinterpolatedZ =\
                coordinate(left_original_points, left_interpolated_points)

            right_original_points, right_interpolated_points, all_right_points =\
                separate(
                    side=1,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            rightoriginalX, rightoriginalY, rightoriginalZ, rightinterpolatedX, rightinterpolatedY, rightinterpolatedZ =\
                coordinate(right_original_points, right_interpolated_points)

        else:
            print("Incorrect side!")

        if side == 0:
            loriginal_point_cloud = go.Scatter3d(
                x=leftoriginalX,
                y=leftoriginalY,
                z=leftoriginalZ,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#5769c5',
                    line=dict(
                        color='#5769c5',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )
            linterpolated_point_cloud = go.Scatter3d(
                x=leftinterpolatedX,
                y=leftinterpolatedY,
                z=leftinterpolatedZ,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#ff4040',
                    line=dict(
                        color='#ff4040',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            data = [loriginal_point_cloud, linterpolated_point_cloud]

        elif side == 1:
            roriginal_point_cloud = go.Scatter3d(
                x=rightoriginalX,
                y=rightoriginalY,
                z=rightoriginalZ,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#5769c5',
                    line=dict(
                        color='#5769c5',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )
            rinterpolated_point_cloud = go.Scatter3d(
                x=rightinterpolatedX,
                y=rightinterpolatedY,
                z=rightinterpolatedZ,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#ff4040',
                    line=dict(
                        color='#ff4040',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            data = [roriginal_point_cloud, rinterpolated_point_cloud]

        elif side == 2:
            loriginal_point_cloud = go.Scatter3d(
                x=leftoriginalX,
                y=leftoriginalY,
                z=leftoriginalZ,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#5769c5',
                    line=dict(
                        color='#5769c5',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )
            linterpolated_point_cloud = go.Scatter3d(
                x=leftinterpolatedX,
                y=leftinterpolatedY,
                z=leftinterpolatedZ,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#ff4040',
                    line=dict(
                        color='#ff4040',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            roriginal_point_cloud = go.Scatter3d(
                x=rightoriginalX,
                y=rightoriginalY,
                z=rightoriginalZ,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#5769c5',
                    line=dict(
                        color='#5769c5',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )
            rinterpolated_point_cloud = go.Scatter3d(
                x=rightinterpolatedX,
                y=rightinterpolatedY,
                z=rightinterpolatedZ,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#ff4040',
                    line=dict(
                        color='#ff4040',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            data = [loriginal_point_cloud, linterpolated_point_cloud, roriginal_point_cloud, rinterpolated_point_cloud]

        else:
            print("Invalid side!")

        layout = go.Layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    title='X',  # X
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    gridcolor="#adad85",
                    gridwidth=3,
                    color="#000000",
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                yaxis=dict(
                    title='Y',  # Y
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    gridcolor="#adad85",
                    gridwidth=3,
                    color="#000000",
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                zaxis=dict(
                    title='Z',  # Z
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    color="#000000",
                    gridcolor="#adad85",
                    gridwidth=3,
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),),
            margin=dict(l=0, r=0, b=0, t=0))

        figure = dict(data=data, layout=layout)

        offline.plot(figure)
    elif mode == 7:
        def get_points(side, rootsequence, imgnumber):
            if side == 0:
                points = read(
                    side=0,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)
            else:
                points = read(
                    side=1,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            points1, points2, points3, all_points = list(), list(), list(), list()

            for i in range(len(points)):
                if points[i][4] == 1:
                    points1.append(points[i][5])
                elif points[i][4] == 2:
                    points2.append(points[i][5])
                else:
                    points3.append(points[i][5])

            points1 = list(itertools.chain.from_iterable(points1))
            points2 = list(itertools.chain.from_iterable(points2))
            points3 = list(itertools.chain.from_iterable(points3))

            all_points.append(points1)
            all_points.append(points2)
            all_points.append(points3)
            all_points = list(itertools.chain.from_iterable(all_points))

            return points1, points2, points3, all_points

        def separate_points_by_coordinate(pointsstep1, pointsstep2, pointsstep3):
            pts1, pts2, pts3 = list(), list(), list()

            pts1.append(pointsstep1)
            pts2.append(pointsstep2)
            pts3.append(pointsstep3)

            pts1 = list(itertools.chain.from_iterable(pts1))
            pts2 = list(itertools.chain.from_iterable(pts2))
            pts3 = list(itertools.chain.from_iterable(pts3))

            pts1X, pts1Y, pts1Z = list(), list(), list()
            pts2X, pts2Y, pts2Z = list(), list(), list()
            pts3X, pts3Y, pts3Z = list(), list(), list()

            for i in range(len(pts1)):
                pts1X.append(pts1[i][0])
                pts1Y.append(pts1[i][1])
                pts1Z.append(pts1[i][2])
            pts1X = tuple(pts1X)
            pts1Y = tuple(pts1Y)
            pts1Z = tuple(pts1Z)

            for i in range(len(pts2)):
                pts2X.append(pts2[i][0])
                pts2Y.append(pts2[i][1])
                pts2Z.append(pts2[i][2])
            pts2X = tuple(pts2X)
            pts2Y = tuple(pts2Y)
            pts2Z = tuple(pts2Z)

            for i in range(len(pts3)):
                pts3X.append(pts3[i][0])
                pts3Y.append(pts3[i][1])
                pts3Z.append(pts3[i][2])
            pts3X = tuple(pts3X)
            pts3Y = tuple(pts3Y)
            pts3Z = tuple(pts3Z)

            return pts1X, pts1Y, pts1Z, pts2X, pts2Y, pts2Z, pts3X, pts3Y, pts3Z



            points = list()

            points.append(pointsstep1)
            points.append(pointsstep2)
            points.append(pointsstep3)
            points = list(itertools.chain.from_iterable(points))

            X, Y, Z = list(), list(), list()

            for i in range(len(points)):
                print(points)
                X.append(points[i][0])
                Y.append(points[i][1])
                Z.append(points[i][2])

            X = tuple(X)
            Y = tuple(Y)
            Z = tuple(Z)

            return X, Y, Z

        if side == 0:
            left_points1, left_points2, left_points3, all_left_points =\
                get_points(
                    side=0,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            leftpts1X, leftpts1Y, leftpts1Z, leftpts2X, leftpts2Y, leftpts2Z, leftpts3X, leftpts3Y, leftpts3Z =\
                separate_points_by_coordinate(left_points1, left_points2, left_points3)

        elif side == 1:
            right_points1, right_points2, right_points3, all_right_points =\
                get_points(
                    side=1,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            rightpts1X, rightpts1Y, rightpts1Z, rightpts2X, rightpts2Y, rightpts2Z, rightpts3X, rightpts3Y, rightpts3Z =\
                separate_points_by_coordinate(right_points1, right_points2, right_points3)

        else:
            left_points1, left_points2, left_points3, all_left_points =\
                get_points(
                    side=0,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            leftpts1X, leftpts1Y, leftpts1Z, leftpts2X, leftpts2Y, leftpts2Z, leftpts3X, leftpts3Y, leftpts3Z =\
                separate_points_by_coordinate(left_points1, left_points2, left_points3)

            right_points1, right_points2, right_points3, all_right_points =\
                get_points(
                    side=1,
                    rootsequence=rootsequence,
                    imgnumber=imgnumber)

            rightpts1X, rightpts1Y, rightpts1Z, rightpts2X, rightpts2Y, rightpts2Z, rightpts3X, rightpts3Y, rightpts3Z =\
                separate_points_by_coordinate(right_points1, right_points2, right_points3)

        if side == 0:
            left_step1_point_cloud = go.Scatter3d(
                x=leftpts1X,
                y=leftpts1Y,
                z=leftpts1Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#ff4040",
                    line=dict(
                        color="#ff4040",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            left_step2_point_cloud = go.Scatter3d(
                x=leftpts2X,
                y=leftpts2Y,
                z=leftpts2Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#2ac940",
                    line=dict(
                        color="#2ac940",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            left_step3_point_cloud = go.Scatter3d(
                x=leftpts3X,
                y=leftpts3Y,
                z=leftpts3Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#4c6ca0",
                    line=dict(
                        color="#4c6ca0",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            data = [left_step1_point_cloud, left_step2_point_cloud, left_step3_point_cloud]

        elif side == 1:
            right_step1_point_cloud = go.Scatter3d(
                x=rightpts1X,
                y=rightpts1Y,
                z=rightpts1Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#ff4040",
                    line=dict(
                        color="#ff4040",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            right_step2_point_cloud = go.Scatter3d(
                x=rightpts2X,
                y=rightpts2Y,
                z=rightpts2Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#2ac940",
                    line=dict(
                        color="#2ac940",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            right_step3_point_cloud = go.Scatter3d(
                x=rightpts3X,
                y=rightpts3Y,
                z=rightpts3Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#4c6ca0",
                    line=dict(
                        color="#4c6ca0",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            data = [right_step1_point_cloud, right_step2_point_cloud, right_step3_point_cloud]

        else:
            left_step1_point_cloud = go.Scatter3d(
                x=leftpts1X,
                y=leftpts1Y,
                z=leftpts1Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#ff4040",
                    line=dict(
                        color="#ff4040",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            left_step2_point_cloud = go.Scatter3d(
                x=leftpts2X,
                y=leftpts2Y,
                z=leftpts2Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#2ac940",
                    line=dict(
                        color="#2ac940",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            left_step3_point_cloud = go.Scatter3d(
                x=leftpts3X,
                y=leftpts3Y,
                z=leftpts3Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#4c6ca0",
                    line=dict(
                        color="#4c6ca0",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            right_step1_point_cloud = go.Scatter3d(
                x=rightpts1X,
                y=rightpts1Y,
                z=rightpts1Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#ff4040",
                    line=dict(
                        color="#ff4040",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            right_step2_point_cloud = go.Scatter3d(
                x=rightpts2X,
                y=rightpts2Y,
                z=rightpts2Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#2ac940",
                    line=dict(
                        color="#2ac940",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            right_step3_point_cloud = go.Scatter3d(
                x=rightpts3X,
                y=rightpts3Y,
                z=rightpts3Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color="#4c6ca0",
                    line=dict(
                        color="#4c6ca0",
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            data = [left_step1_point_cloud, left_step2_point_cloud, left_step3_point_cloud, right_step1_point_cloud, right_step2_point_cloud, right_step3_point_cloud]

        layout = go.Layout(
            showlegend=False,
            # width=800,
            # height=600,
            # autosize=False,
            # title='Point Cloud',
            scene=dict(
                xaxis=dict(
                    title='X',  # X
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    gridcolor="#adad85",
                    gridwidth=3,
                    color="#000000",
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                yaxis=dict(
                    title='Y',  # Y
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    gridcolor="#adad85",
                    gridwidth=3,
                    color="#000000",
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                zaxis=dict(
                    title='Z',  # Z
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    color="#000000",
                    gridcolor="#adad85",
                    gridwidth=3,
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                # aspectratio=dict(x=1, y=1, z=0.7),
                # aspectmode='manual'
            ),
            margin=dict(l=0, r=0, b=0, t=0))

        figure = dict(data=data, layout=layout)

        offline.plot(figure)
    elif mode == 8:
        lpts_e, lpts_d = list(), list()
        lpts_e.append(lpoints1_e)
        lpts_e.append(lpoints2_e)
        lpts_e.append(lpoints3_e)
        lpts_e = list(itertools.chain.from_iterable(lpts_e))

        lpts_d.append(lpoints1_d)
        lpts_d.append(lpoints2_d)
        lpts_d.append(lpoints3_d)
        lpts_d = list(itertools.chain.from_iterable(lpts_d))

        x_e, y_e, z_e = list(), list(), list()
        for i in range(len(lpts_e)):
            x_e.append(lpts_e[i][0])
            y_e.append(lpts_e[i][1])
            z_e.append(lpts_e[i][2])
        x_e = tuple(x_e)
        y_e = tuple(y_e)
        z_e = tuple(z_e)

        x_d, y_d, z_d = list(), list(), list()
        for i in range(len(lpts_d)):
            x_d.append(lpts_d[i][0])
            y_d.append(lpts_d[i][1])
            z_d.append(lpts_d[i][2])
        x_d = tuple(x_d)
        y_d = tuple(y_d)
        z_d = tuple(z_d)

        py.sign_in('PythonAPI', 'ubpiol2cve')

        # pts = np.loadtxt('mesh_dataset.txt')
        # x, y, z = zip(*pts)

        trace_e = go.Mesh3d(x=x_e, y=y_e, z=z_e,
                            alphahull=6,
                            opacity=1.0,
                            # delaunayaxis='x',
                            color='#00FFFF'
                            )
        trace_d = go.Mesh3d(x=x_d, y=y_d, z=z_d,
                            alphahull=7,
                            opacity=1.0,
                            color='#FF00FF')
        py.plot([trace_e, trace_d])
    elif mode == 9:
        lpts_e, lpts_d = list(), list()
        lpts_e.append(lpoints1_e)
        lpts_e.append(lpoints2_e)
        lpts_e.append(lpoints3_e)
        lpts_e = list(itertools.chain.from_iterable(lpts_e))

        lpts_d.append(lpoints1_d)
        lpts_d.append(lpoints2_d)
        lpts_d.append(lpoints3_d)
        lpts_d = list(itertools.chain.from_iterable(lpts_d))

        x_e, y_e, z_e = list(), list(), list()
        for i in range(len(lpts_e)):
            x_e.append(lpts_e[i][0])
            y_e.append(lpts_e[i][1])
            z_e.append(lpts_e[i][2])
        x_e = tuple(x_e)
        y_e = tuple(y_e)
        z_e = tuple(z_e)

        x_d, y_d, z_d = list(), list(), list()
        for i in range(len(lpts_d)):
            x_d.append(lpts_d[i][0])
            y_d.append(lpts_d[i][1])
            z_d.append(lpts_d[i][2])
        x_d = tuple(x_d)
        y_d = tuple(y_d)
        z_d = tuple(z_d)

        if online:
            py.sign_in('PythonAPI', 'ubpiol2cve')

        trace_e = go.Scatter3d(
            x=x_e,
            y=y_e,
            z=z_e,
            mode='markers',
            marker=dict(
                size=0.5,
                color='#FF3232',
                line=dict(
                    # color='rgba(217, 217, 217, 0.14)',
                    color='#FF3232',
                    width=0.5
                ),
                # opacity=0.8
            )
        )

        trace_d = go.Scatter3d(
            x=x_d,
            y=y_d,
            z=z_d,
            mode='markers',
            marker=dict(
                size=1,
                color='#FF3232',
                line=dict(
                    # color='rgba(117, 117, 117, 0.14)',
                    color='#FF3232',
                    width=0.5
                ),
                # opacity=0.8
            )
        )

        data = [trace_e, trace_d]

        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))

        fig = go.Figure(data=data, layout=layout)

        if online:
            py.plot(fig, filename='simple-3d-scatter')
        else:
            offline.plot(fig)
    elif mode == 10:
        lpts_e, lpts_d = list(), list()
        lpts_e.append(lpoints1_e)
        lpts_e.append(lpoints2_e)
        lpts_e.append(lpoints3_e)
        lpts_e = list(itertools.chain.from_iterable(lpts_e))

        lpts_d.append(lpoints1_d)
        lpts_d.append(lpoints2_d)
        lpts_d.append(lpoints3_d)
        lpts_d = list(itertools.chain.from_iterable(lpts_d))

        x_e, y_e, z_e = list(), list(), list()
        for i in range(len(lpts_e)):
            x_e.append(lpts_e[i][0])
            y_e.append(lpts_e[i][1])
            z_e.append(lpts_e[i][2])
        x_e = tuple(x_e)
        y_e = tuple(y_e)
        z_e = tuple(z_e)

        x_d, y_d, z_d = list(), list(), list()
        for i in range(len(lpts_d)):
            x_d.append(lpts_d[i][0])
            y_d.append(lpts_d[i][1])
            z_d.append(lpts_d[i][2])
        x_d = tuple(x_d)
        y_d = tuple(y_d)
        z_d = tuple(z_d)

        if online:
            py.sign_in('PythonAPI', 'ubpiol2cve')
            # py.sign_in('hgalon', 'gK9wn7glYHGnkS6s4J8T')

        trace_e = go.Mesh3d(x=x_e, y=y_e, z=z_e,
                            alphahull=5,  # 5
                            opacity=1.0,
                            color='#808080')  # A91818

        trace_d = go.Mesh3d(x=x_d, y=y_d, z=z_d,
                            alphahull=6,  # 6
                            opacity=1.0,
                            color='#808080')  # FF3232

        trace_scatter_e = go.Scatter3d(
            x=x_e,
            y=y_e,
            z=z_e,
            mode='markers',
            marker=dict(
                size=1,
                color='#999999',  # FF3232
                line=dict(
                    # color='rgba(217, 217, 217, 0.14)',
                    color='#999999',  # FF3232
                    width=0.5
                ),
                # opacity=0.8
            )
        )

        trace_scatter_d = go.Scatter3d(
            x=x_d,
            y=y_d,
            z=z_d,
            mode='markers',
            marker=dict(
                color='#999999',  # 701700
                size=1,
                line=dict(
                    # color='rgba(117, 117, 117, 0.14)',
                    color='#999999',  # 701700
                    width=0.5
                ),
                # opacity=0.8
            )
        )

        data = [trace_e, trace_d, trace_scatter_e, trace_scatter_d]

        layout = go.Layout(
            scene=dict(
                xaxis=dict(
                    title='X-AXIS'),
                yaxis=dict(
                    title='Y-AXIS'),
                zaxis=dict(
                    title='Z-AXIS'),),
            # width=700,
            margin=dict(l=0, r=0, b=0, t=0))

        # fig = go.Figure(data=data)
        fig = dict(data=data, layout=layout)

        if online:
            py.plot(fig)
        else:
            offline.plot(fig)
    elif mode == 11:
        lpts_e, lpts_d = list(), list()

        lpts_e.append(lpoints1_e)
        lpts_e.append(lpoints2_e)
        lpts_e.append(lpoints3_e)
        lpts_e = list(itertools.chain.from_iterable(lpts_e))

        lpts_d.append(lpoints1_d)
        lpts_d.append(lpoints2_d)
        lpts_d.append(lpoints3_d)
        lpts_d = list(itertools.chain.from_iterable(lpts_d))

        x_e, y_e, z_e = list(), list(), list()
        for i in range(len(lpts_e)):
            x_e.append(lpts_e[i][0])
            y_e.append(lpts_e[i][1])
            z_e.append(lpts_e[i][2])
        x_e = tuple(x_e)
        y_e = tuple(y_e)
        z_e = tuple(z_e)

        x_d, y_d, z_d = list(), list(), list()
        for i in range(len(lpts_d)):
            x_d.append(lpts_d[i][0])
            y_d.append(lpts_d[i][1])
            z_d.append(lpts_d[i][2])
        x_d = tuple(x_d)
        y_d = tuple(y_d)
        z_d = tuple(z_d)

        # py.sign_in('PythonAPI', 'ubpiol2cve')
        # offline.init_notebook_mode()

        trace_e = go.Mesh3d(x=x_e, y=y_e, z=z_e,
                            alphahull=6,
                            opacity=1.0,
                            color='#00FFFF')

        trace_d = go.Mesh3d(x=x_d, y=y_d, z=z_d,
                            alphahull=7,
                            opacity=1.0,
                            color='#FF00FF')

        trace_scatter_e = go.Scatter3d(
            x=x_e,
            y=y_e,
            z=z_e,
            mode='markers',
            marker=dict(
                size=3,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5
                ),
                # opacity=0.8
            )
        )

        trace_scatter_d = go.Scatter3d(
            x=x_d,
            y=y_d,
            z=z_d,
            mode='markers',
            marker=dict(
                size=4,
                line=dict(
                    color='rgba(117, 117, 117, 0.14)',
                    width=0.5
                ),
                # opacity=0.8
            )
        )

        data = [trace_e, trace_d, trace_scatter_e, trace_scatter_d]

        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))

        # fig = go.Figure(data=data)
        fig = dict(data=data, layout=layout)

        offline.plot(fig)
    elif mode == 12:
        lpts_e = list()
        lpts_e.append(lpoints1_e)
        lpts_e.append(lpoints2_e)
        lpts_e.append(lpoints3_e)
        lpts_e = list(itertools.chain.from_iterable(lpts_e))

        x_e, y_e, z_e = list(), list(), list()
        for i in range(len(lpts_e)):
            x_e.append(lpts_e[i][0])
            y_e.append(lpts_e[i][1])
            z_e.append(lpts_e[i][2])
        x_e = tuple(x_e)
        y_e = tuple(y_e)
        z_e = tuple(z_e)

        failure = False
        try:
            py.sign_in('PythonAPI', 'ubpiol2cve')
        except:
            failure = True
            print("""\n
                  Attempt to plot in online mode failed ...\n
                  Starting process to plot in offline mode ...\n
                  """)

        trace_e = go.Scatter3d(
            x=x_e,
            y=y_e,
            z=z_e,
            mode='markers',
            marker=dict(
                size=2,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5
                ),
                # opacity=0.8
            )
        )

        data = [trace_e]
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)

        if failure:
            print("Plot offline")
            offline.plot(fig)
        else:
            py.plot(fig, filename='simple-3d-scatter')
    elif mode == 13:
        lpts_e, lpts_d = list(), list()
        lpts_e.append(lpoints1_e)
        lpts_e.append(lpoints2_e)
        lpts_e.append(lpoints3_e)
        lpts_e = list(itertools.chain.from_iterable(lpts_e))

        x_e, y_e, z_e = list(), list(), list()
        for i in range(len(lpts_e)):
            x_e.append(lpts_e[i][0])
            y_e.append(lpts_e[i][1])
            z_e.append(lpts_e[i][2])
        x_e = tuple(x_e)
        y_e = tuple(y_e)
        z_e = tuple(z_e)

        # trace_e = go.Mesh3d(x=x_e, y=y_e, z=z_e,
        #                     # vertexcolor='#000000',
        #                     # facecolor='#000000',
        #                     alphahull=5,
        #                     opacity=1.0,
        #                     color='#A91818')  # #00FFFF

        trace_e = go.Scatter3d(
            x=x_e,
            y=y_e,
            z=z_e,
            mode='markers',
            marker=dict(
                size=2,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5
                ),
                # opacity=0.8
            )
        )

        # print("Type: {}".format(type(trace_e)))
        # print("First vertex: {}".format(trace_e.i))
        # print("Second vertex: {}".format(trace_e.j))
        # print("Third vertex: {}".format(trace_e.k))

        data = [trace_e]

        layout = go.Layout(
            # title='Lung',
            # showlegend=False,
            # paper_bgcolor='#000',
            # plot_bgcolor='#000',
            scene=dict(
                xaxis=dict(
                    autorange=True,
                    # range=[0, -100],
                    # showgrid=False,
                    zeroline=False,
                    # showline=False,
                    # ticks='',
                    # dtick=10,
                    # nticks=5,
                    # showticklabels=False,
                    # gridwidth=8,
                    # gridcolor='#BDBDBD',
                    title='X'),
                yaxis=dict(
                    autorange=True,
                    # range=[-80, 80],
                    # showgrid=False,
                    zeroline=False,
                    # showline=False,
                    # ticks='',
                    # dtick=10,
                    # nticks=5,
                    # showticklabels=False,
                    # gridwidth=8,
                    # gridcolor='#BDBDBD',
                    title='Y'),
                zaxis=dict(
                    autorange=True,
                    # range=[-100, 100],
                    # showgrid=False,
                    zeroline=False,
                    # showline=False,
                    # ticks='',
                    # dtick=10,
                    # nticks=5,
                    # showticklabels=False,
                    # gridwidth=8,
                    # gridcolor='#BDBDBD',
                    title='Z'),),
            # width=800,
            # height=500,
            margin=dict(l=0, r=0, b=0, t=0)
        )

        fig = dict(data=data, layout=layout)

        offline.plot(fig)
    elif mode == 14:
        def readpoints(side=side):
            """ Read text file that contain information about point cloud """
            if side == 0:
                dataset = open(
                    '{}/abe/Left/{}.txt'.format(DIR_RESULT, imgnumber),
                    'r').read().split('\n')
            else:
                dataset = open(
                    '{}/abe/Right/{}.txt'.format(DIR_RESULT, imgnumber),
                    'r').read().split('\n')
            dataset.pop(-1)

            points = list()

            # Points
            string = dataset[0].split('[')
            string2 = string[1].replace(']', '')
            string3 = string2.replace('), ', ');')
            string4 = string3.split(';')

            for j in range(len(string4)):
                pts = string4[j].split(',')
                tupla = (float(pts[0][1:]), float(pts[1]), float(pts[2][:-1]))
                points.append(tupla)

            return points

        def points_by_coordinate(allpoints):
            points = list()

            points.append(allpoints)
            points = list(itertools.chain.from_iterable(points))

            X, Y, Z = list(), list(), list()

            for i in range(len(points)):
                X.append(points[i][0])
                Y.append(points[i][1])
                Z.append(points[i][2])

            X = tuple(X)
            Y = tuple(Y)
            Z = tuple(Z)

            return X, Y, Z

        if side == 0 or side == 1:
            points = readpoints(side=side)
            X, Y, Z = points_by_coordinate(points)

            point_cloud = go.Scatter3d(
                x=X,
                y=Y,
                z=Z,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#ff4040',
                    line=dict(
                        color='#ff4040',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            lighting_effects = dict(ambient=0.5, roughness=0.2, diffuse=0.5, fresnel=0.2, specular=0.05)  # Combined effects
            shape = go.Mesh3d(
                x=X,
                y=Y,
                z=Z,
                alphahull=4,
                opacity=1.0,
                lighting=lighting_effects,
                color='#808080')

            data = [point_cloud, shape]
            # data = [point_cloud]

        else:
            lpoints = readpoints(side=0)
            lX, lY, lZ = points_by_coordinate(lpoints)
            rpoints = readpoints(side=1)
            rX, rY, rZ = points_by_coordinate(rpoints)

            lpoint_cloud = go.Scatter3d(
                x=lX,
                y=lY,
                z=lZ,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#ff4040',
                    line=dict(
                        color='#ff4040',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            rpoint_cloud = go.Scatter3d(
                x=rX,
                y=rY,
                z=rZ,
                # showlegend=False,
                name='',
                mode='markers',
                marker=dict(
                    size=1,
                    color='#ff4040',
                    line=dict(
                        color='#ff4040',
                        width=0.5
                    ),
                    # opacity=1.0
                )
            )

            lighting_effects = dict(ambient=0.5, roughness=0.2, diffuse=0.5, fresnel=0.2, specular=0.05)  # Combined effects

            lshape = go.Mesh3d(
                x=lX,
                y=lY,
                z=lZ,
                alphahull=4,
                opacity=1.0,
                lighting=lighting_effects,
                color='#808080')

            rshape = go.Mesh3d(
                x=rX,
                y=rY,
                z=rZ,
                alphahull=4,
                opacity=1.0,
                lighting=lighting_effects,
                color='#808080')

            data = [lpoint_cloud, rpoint_cloud, lshape, rshape]

        layout = go.Layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    title='X',  # X
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    gridcolor="#adad85",
                    gridwidth=3,
                    color="#000000",
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                yaxis=dict(
                    title='Y',  # Y
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    gridcolor="#adad85",
                    gridwidth=3,
                    color="#000000",
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                zaxis=dict(
                    title='Z',  # Z
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    color="#000000",
                    gridcolor="#adad85",
                    gridwidth=3,
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),),
            margin=dict(l=0, r=0, b=0, t=0))

        figure = dict(data=data, layout=layout)

        offline.plot(figure)
    elif mode == 15:
        linfo = get_plot_info('{}/{}-{}'.format(DIR_RESULT_INTERPOLATED_ST_LEFT, rootsequence, imgnumber))
        rinfo = get_plot_info('{}/{}-{}'.format(DIR_RESULT_INTERPOLATED_ST_RIGHT, rootsequence, imgnumber))

        lpoints1, lpoints2, lpoints3, lpoints = list(), list(), list(), list()
        rpoints1, rpoints2, rpoints3, rpoints = list(), list(), list(), list()

        # Left
        for i in range(len(linfo)):
            if linfo[i][4] == 1:
                lpoints1.append(linfo[i][5])
            elif linfo[i][4] == 2:
                lpoints2.append(linfo[i][5])
            else:
                lpoints3.append(linfo[i][5])
        lpoints2 = [[(-90.0, 12.96875, 55.06496899999999), (-90.0, 7.03125, 52.09621899999999), (-90.0, -0.390625, 49.12746899999999), (-90.0, -7.8125, 41.70559399999999), (-90.0, -13.75, 34.28371899999999), (-90.0, -16.71875, 26.86184399999999), (-90.0, -18.203125, 20.92434399999999), (-90.0, -19.6875, 14.98684399999999), (-90.0, -21.171875, 9.04934399999999), (-90.0, -21.171875, 3.1118439999999907), (-90.0, -22.65625, -2.8256560000000093), (-90.0, -24.140625, -8.76315600000001), (-90.0, -24.140625, -14.70065600000001), (-90.0, -25.625, -20.63815600000001), (-90.0, -25.625, -26.57565600000001), (-90.0, -25.625, -32.51315600000001), (-90.0, -25.625, -38.45065600000001), (-90.0, -25.625, -44.38815600000001), (-90.0, -27.109375, -50.32565600000001), (-90.0, -27.109375, -56.26315600000001), (-90.0, -27.109375, -62.20065600000001), (-90.0, -27.109375, -68.13815600000001), (-90.0, -28.59375, -75.56003100000001), (-90.0, -28.59375, -82.98190600000001), (-90.0, -30.078125, -90.40378100000001), (-90.0, -30.078125, -99.31003100000001), (-90.0, -21.171875, -87.43503100000001), (-90.0, -13.75, -80.01315600000001), (-90.0, -3.359375, -74.07565600000001), (-90.0, 8.515625, -69.62253100000001), (-90.0, 21.875, -69.62253100000001), (-90.0, 36.71875, -71.10690600000001), (-90.0, 47.109375, -77.04440600000001), (-90.0, 53.046875, -81.49753100000001), (-90.0, 60.46875, -85.95065600000001), (-90.0, 58.984375, -78.52878100000001), (-90.0, 57.5, -74.07565600000001), (-90.0, 56.015625, -68.13815600000001), (-90.0, 54.53125, -62.20065600000001), (-90.0, 51.5625, -56.26315600000001), (-90.0, 50.078125, -50.32565600000001), (-90.0, 48.59375, -44.38815600000001), (-90.0, 47.109375, -38.45065600000001), (-90.0, 47.109375, -32.51315600000001), (-90.0, 45.625, -26.57565600000001), (-90.0, 45.625, -20.63815600000001), (-90.0, 44.140625, -14.70065600000001), (-90.0, 42.65625, -8.76315600000001), (-90.0, 41.171875, -2.8256560000000093), (-90.0, 41.171875, 3.1118439999999907), (-90.0, 41.171875, 9.04934399999999), (-90.0, 39.6875, 14.98684399999999), (-90.0, 39.6875, 20.92434399999999), (-90.0, 38.203125, 26.86184399999999), (-90.0, 36.71875, 32.79934399999999), (-90.0, 33.75, 38.73684399999999), (-90.0, 32.265625, 44.67434399999999), (-90.0, 29.296875, 50.61184399999999), (-90.0, 24.84375, 55.06496899999999), (-90.0, 20.390625, 55.06496899999999)],
                    [(-80.0, 15.9375, 68.42434399999999), (-80.0, 10.0, 65.45559399999999), (-80.0, 2.578125, 61.00246899999999), (-80.0, -4.84375, 55.06496899999999), (-80.0, -10.78125, 47.64309399999999), (-80.0, -15.234375, 40.22121899999999), (-80.0, -19.6875, 32.79934399999999), (-80.0, -22.65625, 25.37746899999999), (-80.0, -25.625, 17.95559399999999), (-80.0, -28.59375, 10.53371899999999), (-80.0, -30.078125, 3.1118439999999907), (-80.0, -31.5625, -4.310031000000009), (-80.0, -33.046875, -11.73190600000001), (-80.0, -33.046875, -19.15378100000001), (-80.0, -34.53125, -26.57565600000001), (-80.0, -34.53125, -32.51315600000001), (-80.0, -36.015625, -38.45065600000001), (-80.0, -36.015625, -44.38815600000001), (-80.0, -36.015625, -50.32565600000001), (-80.0, -37.5, -56.26315600000001), (-80.0, -37.5, -62.20065600000001), (-80.0, -37.5, -68.13815600000001), (-80.0, -37.5, -74.07565600000001), (-80.0, -38.984375, -80.01315600000001), (-80.0, -38.984375, -85.95065600000001), (-80.0, -36.015625, -97.82565600000001), (-80.0, -27.109375, -87.43503100000001), (-80.0, -18.203125, -78.52878100000001), (-80.0, -4.84375, -71.10690600000001), (-80.0, 8.515625, -66.65378100000001), (-80.0, 20.390625, -66.65378100000001), (-80.0, 36.71875, -68.13815600000001), (-80.0, 48.59375, -74.07565600000001), (-80.0, 58.984375, -81.49753100000001), (-80.0, 69.375, -93.37253100000001), (-80.0, 69.375, -85.95065600000001), (-80.0, 67.890625, -80.01315600000001), (-80.0, 66.40625, -72.59128100000001), (-80.0, 66.40625, -65.16940600000001), (-80.0, 63.4375, -57.74753100000001), (-80.0, 61.953125, -50.32565600000001), (-80.0, 60.46875, -42.90378100000001), (-80.0, 58.984375, -35.48190600000001), (-80.0, 58.984375, -28.06003100000001), (-80.0, 57.5, -20.63815600000001), (-80.0, 57.5, -13.21628100000001), (-80.0, 54.53125, -5.794406000000009), (-80.0, 53.046875, 1.6274689999999907), (-80.0, 51.5625, 9.04934399999999), (-80.0, 51.5625, 16.47121899999999), (-80.0, 50.078125, 22.40871899999999), (-80.0, 50.078125, 28.34621899999999), (-80.0, 48.59375, 34.28371899999999), (-80.0, 48.59375, 40.22121899999999), (-80.0, 45.625, 46.15871899999999), (-80.0, 42.65625, 52.09621899999999), (-80.0, 39.6875, 58.03371899999999), (-80.0, 36.71875, 63.97121899999999), (-80.0, 30.78125, 66.93996899999999), (-80.0, 24.84375, 68.42434399999999)],
                    [(-70.0, 17.421875, 81.78371899999999), (-70.0, 10.0, 78.81496899999999), (-70.0, 1.09375, 74.36184399999999), (-70.0, -7.8125, 69.90871899999999), (-70.0, -15.234375, 61.00246899999999), (-70.0, -21.171875, 53.58059399999999), (-70.0, -25.625, 46.15871899999999), (-70.0, -28.59375, 38.73684399999999), (-70.0, -31.5625, 31.31496899999999), (-70.0, -34.53125, 23.89309399999999), (-70.0, -36.015625, 16.47121899999999), (-70.0, -38.984375, 9.04934399999999), (-70.0, -40.46875, 1.6274689999999907), (-70.0, -43.4375, -5.794406000000009), (-70.0, -43.4375, -13.21628100000001), (-70.0, -44.921875, -20.63815600000001), (-70.0, -46.40625, -28.06003100000001), (-70.0, -46.40625, -35.48190600000001), (-70.0, -44.921875, -42.90378100000001), (-70.0, -44.921875, -50.32565600000001), (-70.0, -44.921875, -57.74753100000001), (-70.0, -44.921875, -65.16940600000001), (-70.0, -44.921875, -75.56003100000001), (-70.0, -44.921875, -84.46628100000001), (-70.0, -43.4375, -91.88815600000001), (-70.0, -41.953125, -102.27878100000001), (-70.0, -31.5625, -90.40378100000001), (-70.0, -21.171875, -78.52878100000001), (-70.0, -10.78125, -71.10690600000001), (-70.0, 5.546875, -65.16940600000001), (-70.0, 21.875, -63.68503100000001), (-70.0, 35.234375, -66.65378100000001), (-70.0, 47.109375, -72.59128100000001), (-70.0, 58.984375, -80.01315600000001), (-70.0, 70.859375, -91.88815600000001), (-70.0, 70.859375, -82.98190600000001), (-70.0, 69.375, -77.04440600000001), (-70.0, 69.375, -69.62253100000001), (-70.0, 69.375, -62.20065600000001), (-70.0, 69.375, -54.77878100000001), (-70.0, 67.890625, -47.35690600000001), (-70.0, 67.890625, -39.93503100000001), (-70.0, 67.890625, -32.51315600000001), (-70.0, 67.890625, -25.09128100000001), (-70.0, 67.890625, -17.66940600000001), (-70.0, 67.890625, -10.24753100000001), (-70.0, 67.890625, -2.8256560000000093), (-70.0, 66.40625, 4.596218999999991), (-70.0, 64.921875, 12.01809399999999), (-70.0, 63.4375, 19.43996899999999), (-70.0, 61.953125, 26.86184399999999), (-70.0, 61.953125, 34.28371899999999), (-70.0, 60.46875, 41.70559399999999), (-70.0, 56.015625, 49.12746899999999), (-70.0, 53.046875, 56.54934399999999), (-70.0, 48.59375, 63.97121899999999), (-70.0, 45.625, 71.39309399999999), (-70.0, 39.6875, 77.33059399999999), (-70.0, 32.265625, 81.78371899999999), (-70.0, 26.328125, 81.78371899999999)],
                    [(-60.0, 20.390625, 93.65871899999999), (-60.0, 12.96875, 87.72121899999999), (-60.0, 4.0625, 81.78371899999999), (-60.0, -4.84375, 75.84621899999999), (-60.0, -13.75, 66.93996899999999), (-60.0, -22.65625, 59.51809399999999), (-60.0, -28.59375, 52.09621899999999), (-60.0, -31.5625, 44.67434399999999), (-60.0, -36.015625, 37.25246899999999), (-60.0, -38.984375, 29.83059399999999), (-60.0, -40.46875, 22.40871899999999), (-60.0, -43.4375, 14.98684399999999), (-60.0, -46.40625, 7.564968999999991), (-60.0, -47.890625, 0.14309399999999073), (-60.0, -50.859375, -7.278781000000009), (-60.0, -52.34375, -14.70065600000001), (-60.0, -52.34375, -22.12253100000001), (-60.0, -53.828125, -29.54440600000001), (-60.0, -53.828125, -36.96628100000001), (-60.0, -53.828125, -44.38815600000001), (-60.0, -52.34375, -51.81003100000001), (-60.0, -52.34375, -62.20065600000001), (-60.0, -50.859375, -72.59128100000001), (-60.0, -49.375, -82.98190600000001), (-60.0, -47.890625, -91.88815600000001), (-60.0, -44.921875, -105.24753100000001), (-60.0, -41.953125, -94.85690600000001), (-60.0, -34.53125, -82.98190600000001), (-60.0, -21.171875, -72.59128100000001), (-60.0, -0.390625, -63.68503100000001), (-60.0, 26.328125, -63.68503100000001), (-60.0, 44.140625, -69.62253100000001), (-60.0, 57.5, -77.04440600000001), (-60.0, 63.4375, -81.49753100000001), (-60.0, 69.375, -87.43503100000001), (-60.0, 72.34375, -84.46628100000001), (-60.0, 72.34375, -77.04440600000001), (-60.0, 70.859375, -68.13815600000001), (-60.0, 70.859375, -60.71628100000001), (-60.0, 70.859375, -53.29440600000001), (-60.0, 69.375, -45.87253100000001), (-60.0, 69.375, -38.45065600000001), (-60.0, 69.375, -31.02878100000001), (-60.0, 69.375, -23.60690600000001), (-60.0, 67.890625, -16.18503100000001), (-60.0, 66.40625, -8.76315600000001), (-60.0, 66.40625, -1.3412810000000093), (-60.0, 64.921875, 6.080593999999991), (-60.0, 64.921875, 13.50246899999999), (-60.0, 63.4375, 20.92434399999999), (-60.0, 61.953125, 28.34621899999999), (-60.0, 60.46875, 35.76809399999999), (-60.0, 58.984375, 43.18996899999999), (-60.0, 57.5, 50.61184399999999), (-60.0, 54.53125, 58.03371899999999), (-60.0, 51.5625, 65.45559399999999), (-60.0, 48.59375, 72.87746899999999), (-60.0, 44.140625, 80.29934399999999), (-60.0, 38.203125, 86.23684399999999), (-60.0, 30.78125, 90.68996899999999)],
                    [(-40.0, 21.875, 102.56496899999999), (-40.0, 14.453125, 98.11184399999999), (-40.0, 5.546875, 93.65871899999999), (-40.0, -3.359375, 86.23684399999999), (-40.0, -12.265625, 77.33059399999999), (-40.0, -19.6875, 68.42434399999999), (-40.0, -27.109375, 59.51809399999999), (-40.0, -33.046875, 50.61184399999999), (-40.0, -37.5, 41.70559399999999), (-40.0, -41.953125, 32.79934399999999), (-40.0, -46.40625, 23.89309399999999), (-40.0, -50.859375, 14.98684399999999), (-40.0, -53.828125, 6.080593999999991), (-40.0, -55.3125, -2.8256560000000093), (-40.0, -56.796875, -11.73190600000001), (-40.0, -58.28125, -20.63815600000001), (-40.0, -59.765625, -29.54440600000001), (-40.0, -59.765625, -38.45065600000001), (-40.0, -59.765625, -47.35690600000001), (-40.0, -59.765625, -56.26315600000001), (-40.0, -59.765625, -63.68503100000001), (-40.0, -58.28125, -71.10690600000001), (-40.0, -58.28125, -80.01315600000001), (-40.0, -56.796875, -90.40378100000001), (-40.0, -55.3125, -99.31003100000001), (-40.0, -55.3125, -103.76315600000001), (-40.0, -43.4375, -90.40378100000001), (-40.0, -33.046875, -78.52878100000001), (-40.0, -21.171875, -72.59128100000001), (-40.0, -4.84375, -66.65378100000001), (-40.0, 12.96875, -65.16940600000001), (-40.0, 26.328125, -66.65378100000001), (-40.0, 42.65625, -72.59128100000001), (-40.0, 54.53125, -81.49753100000001), (-40.0, 66.40625, -94.85690600000001), (-40.0, 67.890625, -82.98190600000001), (-40.0, 67.890625, -75.56003100000001), (-40.0, 67.890625, -66.65378100000001), (-40.0, 67.890625, -57.74753100000001), (-40.0, 67.890625, -48.84128100000001), (-40.0, 67.890625, -39.93503100000001), (-40.0, 67.890625, -31.02878100000001), (-40.0, 67.890625, -22.12253100000001), (-40.0, 67.890625, -13.21628100000001), (-40.0, 67.890625, -5.794406000000009), (-40.0, 67.890625, 1.6274689999999907), (-40.0, 67.890625, 9.04934399999999), (-40.0, 67.890625, 16.47121899999999), (-40.0, 67.890625, 23.89309399999999), (-40.0, 67.890625, 31.31496899999999), (-40.0, 64.921875, 38.73684399999999), (-40.0, 61.953125, 46.15871899999999), (-40.0, 60.46875, 53.58059399999999), (-40.0, 57.5, 61.00246899999999), (-40.0, 56.015625, 68.42434399999999), (-40.0, 53.046875, 75.84621899999999), (-40.0, 51.5625, 83.26809399999999), (-40.0, 45.625, 90.68996899999999), (-40.0, 39.6875, 96.62746899999999), (-40.0, 32.265625, 101.08059399999999)],
                    [(-30.0, 18.90625, 108.50246899999999), (-30.0, 11.484375, 107.01809399999999), (-30.0, 2.578125, 98.11184399999999), (-30.0, -6.328125, 89.20559399999999), (-30.0, -12.265625, 80.29934399999999), (-30.0, -19.6875, 71.39309399999999), (-30.0, -27.109375, 62.48684399999999), (-30.0, -33.046875, 53.58059399999999), (-30.0, -38.984375, 44.67434399999999), (-30.0, -43.4375, 35.76809399999999), (-30.0, -46.40625, 26.86184399999999), (-30.0, -47.890625, 17.95559399999999), (-30.0, -50.859375, 9.04934399999999), (-30.0, -52.34375, 0.14309399999999073), (-30.0, -55.3125, -8.76315600000001), (-30.0, -56.796875, -17.66940600000001), (-30.0, -58.28125, -25.09128100000001), (-30.0, -58.28125, -32.51315600000001), (-30.0, -59.765625, -39.93503100000001), (-30.0, -59.765625, -47.35690600000001), (-30.0, -59.765625, -56.26315600000001), (-30.0, -59.765625, -66.65378100000001), (-30.0, -59.765625, -74.07565600000001), (-30.0, -58.28125, -84.46628100000001), (-30.0, -58.28125, -91.88815600000001), (-30.0, -55.3125, -102.27878100000001), (-30.0, -43.4375, -91.88815600000001), (-30.0, -33.046875, -82.98190600000001), (-30.0, -21.171875, -74.07565600000001), (-30.0, -10.78125, -69.62253100000001), (-30.0, 4.0625, -66.65378100000001), (-30.0, 21.875, -68.13815600000001), (-30.0, 39.6875, -74.07565600000001), (-30.0, 51.5625, -82.98190600000001), (-30.0, 60.46875, -91.88815600000001), (-30.0, 63.4375, -78.52878100000001), (-30.0, 63.4375, -72.59128100000001), (-30.0, 64.921875, -66.65378100000001), (-30.0, 64.921875, -59.23190600000001), (-30.0, 64.921875, -50.32565600000001), (-30.0, 64.921875, -41.41940600000001), (-30.0, 64.921875, -32.51315600000001), (-30.0, 64.921875, -23.60690600000001), (-30.0, 64.921875, -14.70065600000001), (-30.0, 64.921875, -5.794406000000009), (-30.0, 64.921875, 3.1118439999999907), (-30.0, 63.4375, 12.01809399999999), (-30.0, 63.4375, 20.92434399999999), (-30.0, 61.953125, 29.83059399999999), (-30.0, 61.953125, 37.25246899999999), (-30.0, 60.46875, 44.67434399999999), (-30.0, 58.984375, 52.09621899999999), (-30.0, 57.5, 59.51809399999999), (-30.0, 56.015625, 66.93996899999999), (-30.0, 53.046875, 74.36184399999999), (-30.0, 50.078125, 81.78371899999999), (-30.0, 47.109375, 89.20559399999999), (-30.0, 42.65625, 96.62746899999999), (-30.0, 36.71875, 104.04934399999999), (-30.0, 29.296875, 107.01809399999999)],
                    [(-20.0, 14.453125, 108.50246899999999), (-20.0, 7.03125, 105.53371899999999), (-20.0, -1.875, 96.62746899999999), (-20.0, -10.78125, 87.72121899999999), (-20.0, -19.6875, 78.81496899999999), (-20.0, -27.109375, 69.90871899999999), (-20.0, -34.53125, 61.00246899999999), (-20.0, -40.46875, 52.09621899999999), (-20.0, -43.4375, 43.18996899999999), (-20.0, -46.40625, 34.28371899999999), (-20.0, -49.375, 25.37746899999999), (-20.0, -53.828125, 16.47121899999999), (-20.0, -56.796875, 7.564968999999991), (-20.0, -58.28125, 0.14309399999999073), (-20.0, -59.765625, -7.278781000000009), (-20.0, -61.25, -14.70065600000001), (-20.0, -61.25, -22.12253100000001), (-20.0, -62.734375, -29.54440600000001), (-20.0, -62.734375, -36.96628100000001), (-20.0, -62.734375, -44.38815600000001), (-20.0, -62.734375, -51.81003100000001), (-20.0, -62.734375, -59.23190600000001), (-20.0, -61.25, -69.62253100000001), (-20.0, -59.765625, -80.01315600000001), (-20.0, -56.796875, -90.40378100000001), (-20.0, -56.796875, -96.34128100000001), (-20.0, -43.4375, -85.95065600000001), (-20.0, -28.59375, -78.52878100000001), (-20.0, -13.75, -72.59128100000001), (-20.0, -0.390625, -71.10690600000001), (-20.0, 12.96875, -69.62253100000001), (-20.0, 23.359375, -72.59128100000001), (-20.0, 35.234375, -78.52878100000001), (-20.0, 44.140625, -84.46628100000001), (-20.0, 53.046875, -94.85690600000001), (-20.0, 56.015625, -82.98190600000001), (-20.0, 57.5, -75.56003100000001), (-20.0, 58.984375, -66.65378100000001), (-20.0, 58.984375, -57.74753100000001), (-20.0, 58.984375, -48.84128100000001), (-20.0, 58.984375, -39.93503100000001), (-20.0, 58.984375, -31.02878100000001), (-20.0, 58.984375, -22.12253100000001), (-20.0, 60.46875, -13.21628100000001), (-20.0, 60.46875, -4.310031000000009), (-20.0, 58.984375, 4.596218999999991), (-20.0, 58.984375, 12.01809399999999), (-20.0, 57.5, 19.43996899999999), (-20.0, 56.015625, 26.86184399999999), (-20.0, 54.53125, 34.28371899999999), (-20.0, 53.046875, 41.70559399999999), (-20.0, 51.5625, 49.12746899999999), (-20.0, 50.078125, 56.54934399999999), (-20.0, 50.078125, 63.97121899999999), (-20.0, 47.109375, 71.39309399999999), (-20.0, 45.625, 78.81496899999999), (-20.0, 42.65625, 86.23684399999999), (-20.0, 38.203125, 93.65871899999999), (-20.0, 30.78125, 101.08059399999999), (-20.0, 24.84375, 107.01809399999999)],
                    [(-50.0, 21.380208333333332, 99.10142733333332), (-50.0, 13.958333333333334, 93.65871899999998), (-50.0, 5.052083333333333, 87.72121899999998), (-50.0, -3.8541666666666665, 80.79413566666666), (-50.0, -12.760416666666666, 72.38267733333332), (-50.0, -21.171875, 63.97121899999999), (-50.0, -28.098958333333332, 55.559760666666655), (-50.0, -33.046875, 47.148302333333326), (-50.0, -37.5, 38.73684399999999), (-50.0, -40.963541666666664, 30.32538566666666), (-50.0, -44.427083333333336, 21.913927333333323), (-50.0, -47.395833333333336, 13.50246899999999), (-50.0, -50.364583333333336, 5.091010666666658), (-50.0, -51.848958333333336, -3.320447666666676), (-50.0, -53.333333333333336, -11.73190600000001), (-50.0, -54.817708333333336, -20.14336433333334), (-50.0, -55.3125, -28.554822666666677), (-50.0, -55.807291666666664, -36.471489333333345), (-50.0, -55.807291666666664, -44.38815600000001), (-50.0, -55.807291666666664, -52.304822666666674), (-50.0, -55.3125, -59.726697666666674), (-50.0, -54.322916666666664, -68.13815600000001), (-50.0, -53.828125, -77.04440600000001), (-50.0, -52.838541666666664, -86.44544766666668), (-50.0, -51.354166666666664, -95.35169766666668), (-50.0, -49.375, -104.75273933333335), (-50.0, -41.458333333333336, -91.88815600000002), (-50.0, -32.552083333333336, -80.50794766666668), (-50.0, -20.677083333333332, -72.59128100000001), (-50.0, -2.3697916666666665, -66.15898933333334), (-50.0, 17.916666666666668, -65.16940600000001), (-50.0, 33.75, -68.13815600000001), (-50.0, 48.098958333333336, -73.58086433333334), (-50.0, 57.5, -80.50794766666668), (-50.0, 68.38541666666667, -91.88815600000002), (-50.0, 69.86979166666667, -83.97148933333334), (-50.0, 69.86979166666667, -76.54961433333334), (-50.0, 69.86979166666667, -67.64336433333334), (-50.0, 69.86979166666667, -59.23190600000001), (-50.0, 69.86979166666667, -50.820447666666674), (-50.0, 68.88020833333333, -42.408989333333345), (-50.0, 68.88020833333333, -33.99753100000001), (-50.0, 68.88020833333333, -25.586072666666677), (-50.0, 68.88020833333333, -17.17461433333334), (-50.0, 68.38541666666667, -9.752739333333343), (-50.0, 67.890625, -2.3308643333333428), (-50.0, 67.39583333333333, 5.091010666666658), (-50.0, 66.90104166666667, 12.512885666666657), (-50.0, 66.40625, 19.93476066666666), (-50.0, 65.41666666666667, 27.35663566666666), (-50.0, 63.4375, 34.778510666666655), (-50.0, 61.458333333333336, 42.200385666666655), (-50.0, 59.973958333333336, 49.622260666666655), (-50.0, 57.994791666666664, 57.044135666666655), (-50.0, 56.015625, 64.46601066666666), (-50.0, 53.046875, 71.88788566666666), (-50.0, 50.078125, 79.30976066666666), (-50.0, 45.130208333333336, 86.73163566666665), (-50.0, 39.192708333333336, 92.66913566666665), (-50.0, 31.770833333333332, 97.12226066666665)]
                    ]
        # ind2remove = [3, 4]
        # lpoints2 = [x for i, x in enumerate(lpoints2) if i not in ind2remove]

        lpoints1 = list(itertools.chain.from_iterable(lpoints1))
        lpoints2 = list(itertools.chain.from_iterable(lpoints2))
        lpoints3 = list(itertools.chain.from_iterable(lpoints3))

        lpoints.append(lpoints1)
        lpoints.append(lpoints2)
        lpoints.append(lpoints3)
        lpoints = list(itertools.chain.from_iterable(lpoints))

        # Right
        for i in range(len(rinfo)):
            if rinfo[i][4] == 1:
                rpoints1.append(rinfo[i][5])
            elif rinfo[i][4] == 2:
                rpoints2.append(rinfo[i][5])
            else:
                rpoints3.append(rinfo[i][5])
        rpoints1 = list(itertools.chain.from_iterable(rpoints1))
        rpoints2 = list(itertools.chain.from_iterable(rpoints2))
        rpoints3 = list(itertools.chain.from_iterable(rpoints3))

        rpoints.append(rpoints1)
        rpoints.append(rpoints2)
        rpoints.append(rpoints3)
        rpoints = list(itertools.chain.from_iterable(rpoints))

        lX, lY, lZ = list(), list(), list()
        for i in range(len(lpoints)):
            lX.append(lpoints[i][0])
            lY.append(lpoints[i][1])
            lZ.append(lpoints[i][2])
        lX = tuple(lX)
        lY = tuple(lY)
        lZ = tuple(lZ)

        rX, rY, rZ = list(), list(), list()
        for i in range(len(rpoints)):
            rX.append(rpoints[i][0])
            rY.append(rpoints[i][1])
            rZ.append(rpoints[i][2])
        rX = tuple(rX)
        rY = tuple(rY)
        rZ = tuple(rZ)

        lighting_effects = dict(ambient=0.5, roughness=0.2, diffuse=0.5, fresnel=0.2, specular=0.05)  # Combined effects
        lshape = go.Mesh3d(
            x=lX,
            y=lY,
            z=lZ,
            alphahull=6,
            opacity=1.0,
            lighting=lighting_effects,
            color='#808080')

        rshape = go.Mesh3d(
            x=rX,
            y=rY,
            z=rZ,
            alphahull=6,
            opacity=1.0,
            lighting=lighting_effects,
            color='#808080')

        lpoint_cloud = go.Scatter3d(
            x=lX,
            y=lY,
            z=lZ,
            # showlegend=False,
            name='',
            mode='markers',
            marker=dict(
                size=1,
                color='#ff4040',
                line=dict(
                    color='#ff4040',
                    width=0.5
                ),
                # opacity=1.0
            )
        )

        rpoint_cloud = go.Scatter3d(
            x=rX,
            y=rY,
            z=rZ,
            # showlegend=False,
            name='',
            mode='markers',
            marker=dict(
                size=1,
                color='#ff4040',
                line=dict(
                    color='#ff4040',
                    width=0.5
                ),
                # opacity=1.0
            )
        )

        data = [lshape, rshape]
        # data = [lpoint_cloud, rpoint_cloud]
        # data = [lshape, rshape, lpoint_cloud, rpoint_cloud]

        layout = go.Layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    title='X',  # X
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    gridcolor="#adad85",
                    gridwidth=3,
                    color="#000000",
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                yaxis=dict(
                    title='Y',  # Y
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    gridcolor="#adad85",
                    gridwidth=3,
                    color="#000000",
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),
                zaxis=dict(
                    title='Z',  # Z
                    # showgrid=False,
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    color="#000000",
                    gridcolor="#adad85",
                    gridwidth=3,
                    linecolor='#000000',
                    linewidth=3,
                    zeroline=False),),
            margin=dict(l=0, r=0, b=0, t=0))

        figure = dict(data=data, layout=layout)

        offline.plot(figure)
    else:
        print("Do nothing!")
