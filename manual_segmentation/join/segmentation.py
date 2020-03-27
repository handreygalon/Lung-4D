#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
# import sys


def get_points(image):
    # Set up data to send to mouse handler
    data = {}
    data['image'] = image.copy()
    data['lines'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    points = data['lines']

    return points, data['image']


def create_contour(points):
    mask = np.zeros((256, 256, 3), np.uint8)
    contour = cv2.imread('{}/IM ({}).png'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)

    for i in range(len(points)):
        if i == len(points) - 1:
            i = 0
            cv2.line(
                mask, points[i], points[len(points) - 1], white, 1)
            cv2.line(
                contour, points[i], points[len(points) - 1], white, 1)
        cv2.line(
            mask, points[i], points[i + 1], white, 1)
        cv2.line(
            contour, points[i], points[i + 1], white, 1)

    print('{}/segIM ({}).png'.format(DIR_dest, imgnumber))
    cv2.imwrite('{}/segIM ({}).png'.format(DIR_dest, imgnumber), mask)
    # cv2.imwrite('{}/maskIM ({}).png'.format(DIR_dest, imgnumber), mask)
    # cv2.imwrite('{}/segIM ({}).png'.format(DIR_dest, imgnumber), contour)


def read_points():
    """ Converter list of points (from a txt file) to int """

    dataset = open('{}/points.txt'.format(DIR_dest), 'r').read().split('\n')
    dataset.pop(-1)  # last element is empty

    all_points = []
    ltuplas = []

    for i in range(len(dataset)):
        lpts = dataset[i].replace('), ', ');')[1:-1].split(';')

        for j in range(len(lpts)):
            pts = lpts[j].split(',')
            tupla = (int(pts[0][1:]), int(pts[1][1:-1]))
            ltuplas.append(tupla)

        all_points.append(ltuplas)
        ltuplas = []

    return all_points


def rebuild_contour():
    if respiratory_stage == 0:
        pts = read_points()[0]
        # print("{} ({})".format(pts, len(pts)))
    elif respiratory_stage == 1:
        pts = read_points()[1]
    else:
        pts = read_points()[2]

    # newy = map(lambda x: x[1] + 1, pts)
    # for i in range(len(pts)):
    #     pts[i] = (pts[i][0], newy[i])
    # print(pts)

    mask = np.zeros((256, 256, 3), np.uint8)
    contour = cv2.imread('{}/IM ({}).png'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)

    i = 0
    for point in pts:
        if i == len(pts) - 1:
            i = 0
            cv2.line(
                mask, pts[i], pts[len(pts) - 1], white, 1)
            cv2.line(
                contour, pts[i], pts[len(pts) - 1], white, 1)
        cv2.line(
            mask, pts[i], pts[i + 1], white, 1)
        cv2.line(
            contour, pts[i], pts[i + 1], white, 1)
        i += 1

    if show == 1:
        cv2.imshow('Segmentation', contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save == 1:
        cv2.imwrite('{}/maskIM ({}).png'.format(DIR_dest, imgnumber), mask)
        cv2.imwrite('{}/segIM ({}).png'.format(DIR_dest, imgnumber), contour)

        file = open('{}/points.txt'.format(DIR_dest), 'a')
        file.write("{}\n".format(pts))
        file.close()


def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        data['lines'].append((x, y))  # prepend the point

        # print('Add point: ', data['lines'])
        print('Point {}: {}'.format(len(data['lines']), data['lines'][-1]))

        if len(data['lines']) >= 2:
            cv2.line(
                data['image'],
                data['lines'][len(data['lines']) - 2],
                data['lines'][len(data['lines']) - 1],
                (0, 0, 255),
                1)
            cv2.imshow("Image", data['image'])

        if len(data['lines']) == npoints:
            print("Finish segmentation! Please press ESC")
            cv2.line(
                data['image'],
                data['lines'][0],
                data['lines'][len(data['lines']) - 1],
                (0, 0, 255),
                1)
            cv2.imshow("Image", data['image'])

            create_contour(data['lines'])

            file = open('{}/points.txt'.format(DIR_dest), 'a')
            file.write("{}\n".format(data['lines']))
            file.close()

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif event == cv2.EVENT_MOUSEMOVE and len(data['lines']) >= 1:
        # print('Position: ({}, {})'.format(x, y))
        image = data['image'].copy()

        if len(data['lines']) < npoints:
            # this is just for a line visualization
            cv2.line(image, data['lines'][len(data['lines']) - 1], (x, y), (0, 255, 0), 1)

        cv2.imshow("Image", image)

    elif event == cv2.EVENT_RBUTTONDOWN:
        rebuild_contour()


if __name__ == '__main__':
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    patient = 'Matsushita'
    plan = 'Coronal'
    sequence = 17
    side = 1  # 0 - left | 1 - right
    imgnumber = 16
    save = 0  # 0 - Do not save | 1 - Save
    show = 1  # 0 - Do not show | 1 - Show
    respiratory_stage = 0  # 0 - Final inspiration | 1 - middle | 2 - final expiration

    npoints = 30  # 40

    DIR_ROOT = os.path.expanduser("~/Documents/UDESC")
    DIR_JPG = os.path.expanduser("~/Documents/UDESC/JPG")
    # DIR_orig = '{}/tests/segmentation/{}/pre_processed/{}/{}'.format(DIR_ROOT, patient, plan, sequence)
    DIR_orig = '{}/{}/{}/{}'.format(DIR_JPG, patient, plan, sequence)
    DIR_MANUAL = os.path.expanduser("~/Documents/UDESC/lung/manual_segmentation/join")

    if plan == 'Sagittal':
        DIR_dest = '{}/{}/{}/{}'.format(DIR_MANUAL, patient, plan, sequence)
    else:
        if side == 0:
            DIR_dest = '{}/{}/{}/{}_L'.format(DIR_MANUAL, patient, plan, sequence)
        else:
            DIR_dest = '{}/{}/{}/{}_R'.format(DIR_MANUAL, patient, plan, sequence)

    # img = cv2.imread('{}/IM ({}).png'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)
    img = cv2.imread('{}/IM ({}).jpg'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)
    pts, final_image = get_points(img)
    # cv2.imshow('Image', final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
