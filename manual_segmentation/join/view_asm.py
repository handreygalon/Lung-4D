#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2


def openASMFile():
    """ Open the right file from images of lungs segmented by ASM """

    if plan == 'Coronal':
        if side == 0:
            left_dataset =\
                open('{}/{}/{}/{}_L/Pontos.txt'.format(
                    DIR_ASM_LUNG_MASKS_JPG,
                    patient,
                    plan,
                    sequence,
                    side), 'r').read().split('\n')

            del left_dataset[50:]

            return left_dataset
        else:
            right_dataset =\
                open('{}/{}/{}/{}_R/Pontos.txt'.format(
                    DIR_ASM_LUNG_MASKS_JPG,
                    patient,
                    plan,
                    sequence,
                    side), 'r').read().split('\n')

            del right_dataset[50:]

            return right_dataset
    else:
        dataset =\
            open('{}/{}/{}/{}/Pontos.txt'.format(
                DIR_ASM_LUNG_MASKS_JPG,
                patient,
                plan,
                sequence), 'r').read().split('\n')

        return dataset


def readASMFile(option=False):
    """ Read points from lung's contour segmented by ASM """

    dataset = openASMFile()
    # print('{} ({})\n'.format(dataset, len(dataset)))

    currentpts, allpts = list(), list()

    for i in range(len(dataset)):
        lpts = dataset[i].split(';')
        lpts.pop(-1)

        for j in range(len(lpts)):
            str_pts = (lpts[j].split(','))

            tupla = (int(str_pts[0][1:]), int(str_pts[1][1:-1]))
            currentpts.append(tupla)

        allpts.append(currentpts)

        currentpts = []

    # print("{} ({})\n".format(allpts, len(allpts)))

    return allpts


def viewASMPoints(points, imgnumber):
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


if __name__ == '__main__':
    DIR_JPG = os.path.expanduser("~/Documents/UDESC/JPG")
    DIR_ASM_LUNG_MASKS_JPG = os.path.expanduser("~/Documents/UDESC/ASM/Mascaras/JPG")

    patient = 'Iwasawa'
    plan = 'Sagittal'
    sequence = 13
    side = 0
    show = 1

    lds_lung = readASMFile(option=False)

    for i in range(50):
        viewASMPoints(lds_lung[i], i + 1)
