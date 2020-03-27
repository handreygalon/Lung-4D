#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from operator import itemgetter


def readPoints():
    """ Read points from diaphragmatic surface segmented manually """
    if plan == 'Coronal':
        if side == 0:
            dataset =\
                open('{}/{}/{}/{}_L/points.txt'.format(
                    DIR_MAN_DIAHPRAGM_MASKS,
                    patient,
                    plan,
                    sequence,
                    side), 'r').read().split('\n')

            del dataset[50:]
        elif side == 1:
            dataset =\
                open('{}/{}/{}/{}_R/points.txt'.format(
                    DIR_MAN_DIAHPRAGM_MASKS,
                    patient,
                    plan,
                    sequence,
                    side), 'r').read().split('\n')

            del dataset[50:]
    else:
        dataset =\
            open('{}/{}/{}/{}/points.txt'.format(
                DIR_MAN_DIAHPRAGM_MASKS,
                patient,
                plan,
                sequence), 'r').read().split('\n')

        del dataset[50:]

    all_points, ltuples = list(), list()

    for i in range(len(dataset)):
        lpts = dataset[i].replace('), ', ');')[1:-1].split(';')

        for j in range(len(lpts)):
            pts = lpts[j].split(',')
            tupla = (int(pts[0][1:]), int(pts[1][1:-1]))
            ltuples.append(tupla)

        all_points.append(ltuples)
        ltuples = []

    return all_points


def viewDiaphragmPoints(points, imgnumber):
    img = cv2.imread('{}/{}/{}/{}/IM ({}).jpg'.format(DIR_JPG, patient, plan, sequence, imgnumber), cv2.IMREAD_COLOR)

    for i in range(len(points)):
        if i < 4:
            cv2.circle(img, points[i], 1, yellow, -1)
        else:
            cv2.circle(img, points[i], 1, green, -1)
        # if i <= 24:
        #     cv2.circle(img, points[i], 1, yellow, -1)
        # elif i > 24 and i < 35:  # Diaphragm region
        #     cv2.circle(img, points[i], 1, green, -1)
        # else:
        #     cv2.circle(img, points[i], 1, red, -1)

    if show == 1:
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    DIR_JPG = os.path.expanduser("~/Documents/UDESC/JPG")
    DIR_ASM_LUNG_MASKS_JPG = os.path.expanduser("~/Documents/UDESC/ASM/Mascaras/JPG")
    DIR_MAN_DIAHPRAGM_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/diaphragm")

    yellow = (0, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)

    patient = 'Iwasawa'
    plan = 'Coronal'
    sequence = 5
    side = 1
    show = 1
    save = 0

    lds_lung = readPoints()

    for i in range(50):
        viewDiaphragmPoints(lds_lung[i], i + 1)
