#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
# import cv


"""

Joins in an image the left and right segmented lungs (using ASM)

"""


def openFile(plan):
    if plan == 'Coronal':
        path_lado_0 = '{}-Lado_0/'.format(DIR_origin)
        path_lado_1 = '{}-Lado_1/'.format(DIR_origin)

        dataset_lado_0 =\
            open('{}/Pontos.txt'.format(path_lado_0), 'r').read().split('\n')
        del dataset_lado_0[50:]

        dataset_lado_1 =\
            open('{}/Pontos.txt'.format(path_lado_1), 'r').read().split('\n')
        del dataset_lado_1[50:]

        return dataset_lado_0, dataset_lado_1
    else:
        dataset =\
            open('{}/Pontos.txt'.format(DIR_origin), 'r').read().split('\n')
        del dataset[50:]

        return dataset


def createNewFileCoronal(plan='Coronal'):
    all_points_0 = []
    all_points_1 = []
    all_points = []

    ds0, ds1 = openFile(plan)

    list_pts = []

    for i in range(len(ds0)):
        lpts = ds0[i].split(';')
        lpts.pop(-1)

        for j in range(len(lpts)):
            spts = (lpts[j].split(','))

            tupla = (int(spts[0][1:]), int(spts[1][1:-1]))
            list_pts.append(tupla)

        file = open('{}/points_left.txt'.format(DIR_dest), 'a')
        file.write("{}\n".format(list_pts))
        file.close()

        all_points_0.append(list_pts)

        list_pts = []

    list_pts = []

    for i in range(len(ds1)):
        lpts = ds1[i].split(';')
        lpts.pop(-1)

        for j in range(len(lpts)):
            spts = (lpts[j].split(','))

            tupla = (int(spts[0][1:]), int(spts[1][1:-1]))
            list_pts.append(tupla)

        file = open('{}/points_right.txt'.format(DIR_dest), 'a')
        file.write("{}\n".format(list_pts))
        file.close()

        all_points_1.append(list_pts)

        list_pts = []

    for i in range(len(all_points_0)):
        all_points.append(all_points_0[i] + all_points_1[i])

    file = open('{}/points_all.txt'.format(DIR_dest), 'w')
    for i in range(len(all_points)):
        file.write("{}\n".format(all_points[i]))
    file.close()

    return all_points_0, all_points_1, all_points


def createNewFileSagittal(plan='Sagittal'):
    all_points = []

    ds = openFile(plan)

    list_pts = []

    for i in range(len(ds)):
        lpts = ds[i].split(';')
        lpts.pop(-1)

        for j in range(len(lpts)):
            spts = (lpts[j].split(','))

            tupla = (int(spts[0][1:]), int(spts[1][1:-1]))
            list_pts.append(tupla)

        file = open('{}/points.txt'.format(DIR_dest), 'a')
        file.write("{}\n".format(list_pts))
        file.close()

        all_points.append(list_pts)

        list_pts = []

    return all_points


def createMaskCoronal(points_lung_left, points_lung_right, number):
    mask = np.zeros((256, 256, 3), np.uint8)

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

    cv2.imwrite('{}/maskIM ({}).png'.format(DIR_dest, number), mask)


def createMaskSagittal(points, number):
    mask = np.zeros((256, 256, 3), np.uint8)

    i = 0
    for point in points:
        if i == len(points) - 1:
            i = 0
            cv2.line(
                mask, points[i], points[len(points) - 1], (0, 0, 255), 1)

        cv2.line(
            mask, points[i], points[i + 1], (0, 0, 255), 1)

        i = i + 1

    cv2.imwrite('{}/maskIM ({}).png'.format(DIR_dest, number), mask)


if __name__ == '__main__':
    patient = 'Iwasawa'
    # plan = 'Coronal'
    # plan = 'Sagittal'
    plan = sys.argv[1]
    # sequence = raw_input("Sequence: ")
    sequence = sys.argv[2]

    DIR_origin = '/home/handrey/Documents/UDESC/ASM/Segmentadas/{}/{}/{}'\
        .format(patient, plan, sequence)
    DIR_dest = '/home/handrey/Documents/UDESC/ASM/Mascaras/PNG/{}/{}/{}'\
        .format(patient, plan, sequence)

    if plan == "Coronal":
        pts_left, pts_right, all_points = createNewFileCoronal()

        for i in range(50):
            createMaskCoronal(pts_left[i], pts_right[i], i + 1)
    else:
        pts = createNewFileSagittal()

        for i in range(50):
            createMaskSagittal(pts[i], i + 1)
