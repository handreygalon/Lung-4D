#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
# import cv

from constant import *


"""

Joins in an image the left and right segmented lungs

"""


def read_points(pts):
    """ Converter list of points (from a txt file) to int """

    # dataset = open('{}/points.txt'.format(DIR_dest), 'r').read().split('\n')
    # dataset.pop(-1)  # last element is empty
    dataset = pts

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


def openFile():
    path_left = '{}_L'.format(DIR_orig)
    path_right = '{}_R'.format(DIR_orig)

    dataset_left =\
        open('{}/points.txt'.format(path_left), 'r').read().split('\n')
    del dataset_left[50:]
    dataset_left = read_points(dataset_left)

    dataset_right =\
        open('{}/points.txt'.format(path_right), 'r').read().split('\n')
    del dataset_right[50:]
    dataset_right = read_points(dataset_right)

    return dataset_left, dataset_right


def createNewFileCoronal(dsl, dsr):
    all_points = []

    for i in range(len(dsl)):
        all_points.append(dsl[i] + dsr[i])

    file = open('{}/points.txt'.format(DIR_dest), 'w')
    for i in range(len(all_points)):
        file.write("{}\n".format(all_points[i]))
    file.close()

    return all_points


def createMaskCoronal(points_lung_left, points_lung_right, number):
    mask = np.zeros((256, 256, 3), np.uint8)

    if mode == 0:
        for i in range(len(points_lung_left) - 1):
            cv2.line(
                mask,
                points_lung_left[i],
                points_lung_left[i + 1],
                (0, 0, 255), 1)

        for j in range(len(points_lung_right) - 1):
            cv2.line(
                mask,
                points_lung_right[j], points_lung_right[j + 1],
                (0, 0, 255), 1)
    else:
        i = 0
        for point in points_lung_left:
            if i == len(points_lung_left) - 1:
                i = 0
                cv2.line(
                    mask,
                    points_lung_left[i],
                    points_lung_left[len(points_lung_left) - 1],
                    (0, 0, 255), 1)

            cv2.line(
                mask,
                points_lung_left[i],
                points_lung_left[i + 1],
                (0, 0, 255), 1)

            i = i + 1

        j = 0
        for point in points_lung_right:
            if j == len(points_lung_right) - 1:
                j = 0
                cv2.line(
                    mask,
                    points_lung_right[j],
                    points_lung_right[len(points_lung_right) - 1],
                    (0, 0, 255), 1)

            cv2.line(
                mask,
                points_lung_right[j], points_lung_right[j + 1],
                (0, 0, 255), 1)

            j = j + 1

    # cv2.imshow('{}/maskIM ({}).png'.format(DIR_dest, number), mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('{}/maskIM ({}).png'.format(DIR_dest, number), mask)


if __name__ == '__main__':
    mode = 0  # 0 - diaphragm | 1 - lung
    option = 2  # 0 - ASM | 1 - Manual | 2 - ASM+Manual
    patient = 'Iwasawa'
    plan = 'Coronal'
    sequence = 4

    if mode == 0:
        DIR_orig = '{}/{}/{}/{}'.format(DIR_MAN_DIAHPRAGM_MASKS, patient, plan, sequence)
        DIR_dest = '{}/{}/{}/{}'.format(DIR_MAN_DIAHPRAGM_MASKS, patient, plan, sequence)
    else:
        if option == 0:
            DIR_orig = '{}/{}/{}/{}'.format(DIR_ASM_LUNG_MASK, patient, plan, sequence)
            DIR_dest = '{}/{}/{}/{}'.format(DIR_ASM_LUNG_MASK, patient, plan, sequence)
        elif option == 1:
            DIR_orig = '{}/{}/{}/{}'.format(DIR_MAN_LUNG_MASKS, patient, plan, sequence)
            DIR_dest = '{}/{}/{}/{}'.format(DIR_MAN_LUNG_MASKS, patient, plan, sequence)
        else:
            DIR_orig = '{}/{}/{}/{}'.format(DIR_ASM_MAN_LUNG_MASK, patient, plan, sequence)
            DIR_dest = '{}/{}/{}/{}'.format(DIR_ASM_MAN_LUNG_MASK, patient, plan, sequence)

    ds_left, ds_right = openFile()

    all_points = createNewFileCoronal(ds_left, ds_right)

    for i in range(50):
        createMaskCoronal(ds_left[i], ds_right[i], i + 1)
