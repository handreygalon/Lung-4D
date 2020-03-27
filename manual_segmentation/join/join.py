#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import numpy as np


""" Get only the diaphragmataic points of manual segmentation and then join
    with the points of the rest of the lung """


def read_points(path):
    dataset = open('{}/points.txt'.format(path), 'r').read().split('\n')
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


def join_points(wallpoints, diaphragmpoints, imgnumber, stage):
    if stage == 0:
        leftwall = wallpoints[0][:25]
        rightwall = wallpoints[0][35:]
    elif stage == 1:
        leftwall = wallpoints[1][:25]
        rightwall = wallpoints[1][35:]
    elif stage == 2:
        leftwall = wallpoints[2][:25]
        rightwall = wallpoints[2][35:]
    else:
        leftwall = wallpoints[3][:25]
        rightwall = wallpoints[3][35:]

    diaphragm = diaphragmpoints[imgnumber - 1]

    lung_contour = leftwall + diaphragm + rightwall
    # print("{} ({})".format(lung_contour, len(lung_contour)))
    return lung_contour


def viewPoints(points, imgnumber):
    img = cv2.imread('{}/{}/{}/{}/IM ({}).jpg'.format(DIR_JPG, patient, plan, sequence, imgnumber), cv2.IMREAD_COLOR)
    for i in range(len(points)):
        if i <= 24:
            cv2.circle(img, points[i], 1, (0, 255, 255), -1)
        elif i > 24 and i < 35:  # Diaphragm region
            cv2.circle(img, points[i], 1, (0, 255, 0), -1)
        else:
            cv2.circle(img, points[i], 1, (0, 0, 255), -1)

    if show == 1:
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save == 1:
        if plan == 'Sagittal':
            cv2.imwrite('{}/pointsIM ({}).png'.format(DIR_SEGMENTED, imgnumber), img)
        else:
            cv2.imwrite('{}/{}/pointsIM ({}).png'.format(DIR_SEGMENTED, side, imgnumber), img)


def save_contour(points):
    mask = np.zeros((256, 256, 3), np.uint8)
    contour = cv2.imread('{}/{}/{}/{}/IM ({}).jpg'.format(DIR_JPG, patient, plan, sequence, imgnumber), cv2.IMREAD_COLOR)

    i = 0
    for point in points:
        if i == len(points) - 1:
            i = 0
            cv2.line(
                mask, points[i], points[len(points) - 1], red, 1)
            cv2.line(
                contour, points[i], points[len(points) - 1], red, 1)
        cv2.line(
            mask, points[i], points[i + 1], red, 1)
        cv2.line(
            contour, points[i], points[i + 1], red, 1)
        i += 1

    if show == 1:
        cv2.imshow('Segmentation', contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save == 1:
        cv2.imwrite('{}/maskIM ({}).png'.format(DIR_SEGMENTED, imgnumber), mask)
        cv2.imwrite('{}/segIM ({}).png'.format(DIR_SEGMENTED, imgnumber), contour)

        file = open('{}/points.txt'.format(DIR_SEGMENTED), 'a')
        file.write("{}\n".format(points))
        file.close()


if __name__ == '__main__':
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)

    try:
        patient = 'Matsushita'
        plan = 'Coronal'
        sequence = 21
        side = 1  # 0 - left | 1 - right
        imgnumber = 1
        show = 1
        save = 0
        respiratory_stage = 0  # 0 - Final inspiration | 1 - middle | 2 - final expiration

        txtargv2 = '-imgnumber={}'.format(imgnumber)
        txtargv3 = '-show={}'.format(show)
        txtargv4 = '-save={}'.format(save)
        txtargv5 = '-stage={}'.format(respiratory_stage)

        if len(sys.argv) > 1:
            txtargv2 = sys.argv[1]
            if len(sys.argv) > 2:
                txtargv3 = sys.argv[2]
                if len(sys.argv) > 3:
                    txtargv4 = sys.argv[3]
                    if len(sys.argv) > 4:
                        txtargv5 = sys.argv[4]

        txtargv = '{}|{}|{}|{}'.format(txtargv2, txtargv3, txtargv4, txtargv5)

        if txtargv.find('-imgnumber') != -1:
            txttmp = txtargv.split('-imgnumber')[1]
            txttmp = txttmp.split('=')[1]
            imgnumber = int(txttmp.split('|')[0])

        if txtargv.find('-show') != -1:
            txttmp = txtargv.split('-show')[1]
            txttmp = txttmp.split('=')[1]
            show = int(txttmp.split('|')[0])

        if txtargv.find('-save') != -1:
            txttmp = txtargv.split('-save')[1]
            txttmp = txttmp.split('=')[1]
            save = int(txttmp.split('|')[0])

        if txtargv.find('-stage') != -1:
            txttmp = txtargv.split('-stage')[1]
            txttmp = txttmp.split('=')[1]
            respiratory_stage = int(txttmp.split('|')[0])

    except ValueError:
        print("ERROR!")

    DIR_JPG = os.path.expanduser("~/Documents/UDESC/JPG")
    DIR_ROOT = os.path.expanduser("~/Documents/UDESC/lung/manual_segmentation/join")
    DIR_MAN_LUNG_DIAHPRAGM_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/lung/Manual")
    if plan == 'Sagittal':
        DIR_WALLS = '{}/{}/{}/{}'.format(DIR_ROOT, patient, plan, sequence)
        DIR_DIAPH = os.path.expanduser("~/Documents/UDESC/segmented/diaphragm/{}/{}/{}".format(patient, plan, sequence))
        # DIR_SEGMENTED = os.path.expanduser("{}/{}/{}/segmented/{}".format(DIR_ROOT, patient, plan, sequence))
        DIR_SEGMENTED = os.path.expanduser("{}/{}/{}/{}".format(DIR_MAN_LUNG_DIAHPRAGM_MASKS, patient, plan, sequence))
    else:
        if side == 0:
            DIR_WALLS = '{}/{}/{}/{}_L'.format(DIR_ROOT, patient, plan, sequence)
            DIR_DIAPH = os.path.expanduser("~/Documents/UDESC/segmented/diaphragm/{}/{}/{}_L".format(patient, plan, sequence))
            # DIR_SEGMENTED = os.path.expanduser("{}/{}/{}/segmented/{}_L".format(DIR_ROOT, patient, plan, sequence))
            DIR_SEGMENTED = os.path.expanduser("{}/{}/{}/{}_L".format(DIR_MAN_LUNG_DIAHPRAGM_MASKS, patient, plan, sequence))
        else:
            DIR_WALLS = '{}/{}/{}/{}_R'.format(DIR_ROOT, patient, plan, sequence)
            DIR_DIAPH = os.path.expanduser("~/Documents/UDESC/segmented/diaphragm/{}/{}/{}_R".format(patient, plan, sequence))
            # DIR_SEGMENTED = os.path.expanduser("{}/{}/{}/segmented/{}_R".format(DIR_ROOT, patient, plan, sequence))
            DIR_SEGMENTED = os.path.expanduser("{}/{}/{}/{}_R".format(DIR_MAN_LUNG_DIAHPRAGM_MASKS, patient, plan, sequence))

    ds_walls = read_points(path=DIR_WALLS)
    ds_diaph = read_points(path=DIR_DIAPH)

    lung_contour = join_points(ds_walls, ds_diaph, imgnumber, respiratory_stage)

    viewPoints(lung_contour, imgnumber)
    if save == 1:
        save_contour(lung_contour)
