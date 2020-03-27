#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
from constant import *


class JoinMasks(object):

    """ It replaces points of the diaphragmatic region segmented by the ASM
        by points of the diaphragmatic region suggested manually """

    def __init__(self, patient, plan, sequence, side, show, save):
        """
        Parameters
        ----------
        show: bool
            Shows the result image if set to 'True', or does not show, otherwise
        save: bool
            Save the result image if set to 'True', or does not save, otherwise
        """
        self.patient = patient
        self.plan = plan
        self.sequence = sequence
        self.side = side
        self.show = show
        self.save = save

    def openASMFile(self):
        """ Open the right file from images of lungs segmented by ASM """
        if self.plan == 'Coronal':
            if side == 0:
                left_dataset =\
                    open('{}/{}/{}/{}_L/Pontos.txt'.format(
                        DIR_ASM_LUNG_MASKS_JPG,
                        self.patient,
                        self.plan,
                        self.sequence,
                        self.side), 'r').read().split('\n')

                del left_dataset[50:]

                return left_dataset
            else:
                right_dataset =\
                    open('{}/{}/{}/{}_R/Pontos.txt'.format(
                        DIR_ASM_LUNG_MASKS_JPG,
                        self.patient,
                        self.plan,
                        self.sequence,
                        self.side), 'r').read().split('\n')

                del right_dataset[50:]

                return right_dataset
        else:
            dataset =\
                open('{}/{}/{}/{}/Pontos.txt'.format(
                    DIR_ASM_LUNG_MASKS_JPG,
                    self.patient,
                    self.plan,
                    self.sequence), 'r').read().split('\n')

            return dataset

    def createFile(self, option=False):
        """ Read points from lung's contour segmented by ASM """
        if self.side == 0:
            filename = 'points_left.txt'
            lungside = 'L'
        else:
            filename = 'points_right.txt'
            lungside = 'R'

        dataset = self.openASMFile()
        # print('{} ({})\n'.format(dataset, len(dataset)))

        currentpts, allpts = list(), list()

        for i in range(len(dataset)):
            lpts = dataset[i].split(';')
            lpts.pop(-1)

            for j in range(len(lpts)):
                str_pts = (lpts[j].split(','))

                tupla = (int(str_pts[0][1:]), int(str_pts[1][1:-1]))
                currentpts.append(tupla)

            if option:
                file = open(
                    '{}/{}/{}/{}_{}/{}'.format(
                        DIR_ASM_LUNG_MASKS_JPG,
                        self.patient,
                        self.plan,
                        self.sequence,
                        lungside,
                        filename), 'a')
                file.write("{}\n".format(currentpts))
                file.close()

            allpts.append(currentpts)

            currentpts = []

        # print("{} ({})\n".format(allpts, len(allpts)))

        return allpts

    def readPoints(self):
        """ Read points from diaphragmatic surface segmented manually """
        if self.plan == 'Coronal':
            if self.side == 0:
                dataset =\
                    open('{}/{}/{}/{}_L/points.txt'.format(
                        DIR_MAN_DIAHPRAGM_MASKS,
                        self.patient,
                        self.plan,
                        self.sequence,
                        self.side), 'r').read().split('\n')

                del dataset[50:]
            elif self.side == 1:
                dataset =\
                    open('{}/{}/{}/{}_R/points.txt'.format(
                        DIR_MAN_DIAHPRAGM_MASKS,
                        self.patient,
                        self.plan,
                        self.sequence,
                        self.side), 'r').read().split('\n')

                del dataset[50:]
        else:
            dataset =\
                open('{}/{}/{}/{}/points.txt'.format(
                    DIR_MAN_DIAHPRAGM_MASKS,
                    self.patient,
                    self.plan,
                    self.sequence), 'r').read().split('\n')

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

    def openFile(self):
        if self.plan == 'Coronal':
            if self.side == 0:
                left_dataset =\
                    open('{}/{}/{}/{}_L/points_left.txt'.format(
                        DIR_ASM_LUNG_MASKS_JPG,
                        self.patient,
                        self.plan,
                        self.sequence,
                        self.side), 'r').read().split('\n')

                del left_dataset[50:]

                return left_dataset
            else:
                right_dataset =\
                    open('{}/{}/{}/{}_R/points_right.txt'.format(
                        DIR_ASM_LUNG_MASKS_JPG,
                        self.patient,
                        self.plan,
                        self.sequence,
                        self.side), 'r').read().split('\n')

                del right_dataset[50:]

                return right_dataset
        else:
            dataset =\
                open('{}/{}/{}_R/{}/points.txt'.format(
                    DIR_ASM_LUNG_MASKS_JPG,
                    self.patient,
                    self.plan,
                    self.sequence), 'r').read().split('\n')

            return dataset

    def createMask(self, points, imgnumber, option=False):
        if self.plan == 'Coronal':
            if self.side == 0:
                DIR = '{}/{}/{}/{}_L'.format(DIR_ASM_MAN_LUNG_MASK, self.patient, self.plan, self.sequence)
            else:
                DIR = '{}/{}/{}/{}_R'.format(DIR_ASM_MAN_LUNG_MASK, self.patient, self.plan, self.sequence)
        else:
            DIR = '{}/{}/{}/{}'.format(DIR_ASM_MAN_LUNG_MASK, self.patient, self.plan, self.sequence)

        if option:
            mask = cv2.imread(
                '{}/{}/{}/{}/IM ({}).jpg'.format(
                    DIR_JPG, self.patient, self.plan, self.sequence, imgnumber),
                cv2.IMREAD_COLOR)

        else:
            mask = np.zeros((256, 256, 3), np.uint8)

        i = 0
        for point in points:
            if i == len(points) - 1:
                i = 0
                cv2.line(
                    mask,
                    points[i],
                    points[len(points) - 1],
                    (0, 0, 255), 1)

            cv2.line(
                mask,
                points[i],
                points[i + 1],
                (0, 0, 255), 1)

            i = i + 1

        # print('{}/maskIM ({}).png'.format(DIR, imgnumber))
        if self.show:
            cv2.imshow('Maks', mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.save:
            cv2.imwrite('{}/maskIM ({}).png'.format(DIR, imgnumber), mask)

    def viewPoints(self, points, imgnumber):
        img = cv2.imread(
            '{}/{}/{}/{}/IM ({}).jpg'.format(
                DIR_JPG, self.patient, self.plan, self.sequence, imgnumber),
            cv2.IMREAD_COLOR)

        for i in range(len(points)):
            if i <= 24:
                cv2.circle(img, points[i], 1, (0, 255, 255), -1)
            elif i > 24 and i < 35:  # Diaphragm region
                cv2.circle(img, points[i], 1, (0, 255, 0), -1)
            else:
                cv2.circle(img, points[i], 1, (0, 0, 255), -1)

        if self.show:
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if self.save:
            if self.plan == 'Coronal':
                if self.side == 0:
                    DIR = '{}/{}/{}/{}_L'.format(DIR_ASM_MAN_LUNG_MASK, self.patient, self.plan, self.sequence)
                else:
                    DIR = '{}/{}/{}/{}_R'.format(DIR_ASM_MAN_LUNG_MASK, self.patient, self.plan, self.sequence)
            else:
                DIR = '{}/{}/{}/{}'.format(DIR_ASM_MAN_LUNG_MASK, self.patient, self.plan, self.sequence)

            cv2.imwrite('{}/pointsIM ({}).png'.format(DIR, imgnumber), img)

    def buildFile(self, asmpts, manualpts, createtxt=False):
        """ Replacees diaphragm's points (obtatined by ASM) with the points obtained through manual segmentation """

        del asmpts[50:]

        # diaphragm_pts = list()
        for i in range(len(asmpts)):
            # print("{}\n".format(asmpts[i]))

            # diaphragm_pts = asmpts[i][25:35]
            # print("{}\n".format(diaphragm_pts))
            # print("{}\n".format(manualpts[i]))

            for j in range(len(manualpts[0])):
                asmpts[i][j + 25] = manualpts[i][j]

            # print("{}\n".format(asmpts[i]))
            # c = raw_input("?")

        if createtxt:
            if self.plan == 'Coronal':
                if self.side == 0:
                    DIR = '{}/{}/{}/{}_L'.format(DIR_ASM_MAN_LUNG_MASK, self.patient, self.plan, self.sequence)
                else:
                    DIR = '{}/{}/{}/{}_R'.format(DIR_ASM_MAN_LUNG_MASK, self.patient, self.plan, self.sequence)
            else:
                DIR = '{}/{}/{}/{}'.format(DIR_ASM_MAN_LUNG_MASK, self.patient, self.plan, self.sequence)

            for i in range(len(asmpts)):
                file = open('{}/points.txt'.format(DIR), 'a')
                file.write("{}\n".format(asmpts[i]))
                file.close()

        return asmpts


if __name__ == '__main__':
    patient = 'Iwasawa'
    plan = 'Sagittal'
    # plan = sys.argv[1]
    sequence = 6
    side = 0  # 0 - left | 1 - right
    imgnum = 1
    show = True
    save = False

    manager = JoinMasks(patient, plan, sequence, side, show, save)

    # lds = manager.openASMFile()

    lds_lung = manager.createFile(option=False)
    # print("{} ({}) - ({})\n".format(lds_lung, len(lds_lung), len(lds_lung[0])))

    lds_diaphragm = manager.readPoints()
    # print("{} ({}) - ({})\n".format(lds_diaphragm, len(lds_diaphragm), len(lds_diaphragm[0])))

    '''
    for i in range(50):
        manager.viewPoints(lds_lung[i], i + 1)
        manager.createMask(lds_lung[i], i + 1, True)
    '''

    lungpts = manager.buildFile(lds_lung, lds_diaphragm, True)

    for i in range(50):
        manager.viewPoints(lungpts[i], i + 1)
        manager.createMask(lungpts[i], i + 1, True)
