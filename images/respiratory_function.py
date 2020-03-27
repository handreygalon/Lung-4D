#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from operator import itemgetter


class ExtractRespiratoryPattern(object):
    def __init__(self, patient, plan):
        self.patient = patient
        self.plan = plan

    def extract(self, filename):
        print(filename)

        img = cv2.imread('{}'.format(filename), 0)

        color_pts = np.argwhere(img > 70)

        """
        Sorts the pixels according to their x coordenate.
        Obs: np.argwhere inverts x and y, it's like (y, x), because of it,
        the parameter of itemgetter is 1 (to get x coordinate)
        """
        lcolor_pts = sorted(color_pts.tolist(), key=itemgetter(1))

        # Reverse the coordinates and store the result in lordered_pts list
        lordered_pts = []
        for j in range(len(lcolor_pts)):
            lordered_pts.append(lcolor_pts[j][::-1])

        """
        Convert pixels coordinates into a tuples and check which column
        has pixels that corresponding to diaphragmatic level
        Obs. There are some columns that doesnt have any pixel that
        correpond to diaphragmatic level.
        """
        # Columns that have a pixel corresponding diaphragmatic level
        lcolumn_available = []
        for j in range(len(lordered_pts)):
            lordered_pts[j] = tuple(lordered_pts[j])
            lcolumn_available.append(lordered_pts[j][0])
        lcolumn_available = list(set(lcolumn_available))
        # print("Ordered points: ", lordered_pts)
        # print("Columns available: ", lcolumn_available)

        """
        If there are no pixel that corresponding diaphragmatic level in the
        first column, assign to it the value of the second y coordinate
        """
        if lcolumn_available[0] is not 0:
            y = max(
                [x for x in lordered_pts if x[0] == lcolumn_available[0]],
                key=itemgetter(1))[1]
            lordered_pts.insert(0, (0, y))
            lcolumn_available.insert(0, 0)

        """
        If there are no pixel that corresponding diaphragmatic level in the
        last column, assign to it the value of the penultimate y coordinate
        available
        """
        if lcolumn_available[-1] is not 49:
            lordered_pts.append(
                (49, lordered_pts[len(lcolumn_available)][1]))
            lcolumn_available.append(49)

        """
        Get the biggest y value in each column that represents the
        diaphragmatic level
        """
        column = 0
        lcolumn = []
        ldiaphragm_pts = []
        for j in range(50):
            # Get the column's points
            lcolumn = [x for x in lordered_pts if x[0] == column]
            # print('{}: {}'.format(j, lcolumn))

            if len(lcolumn) > 0:
                ldiaphragm_pts.append(
                    max(lcolumn, key=itemgetter(1)))  # Get the biggest y
            else:
                # Get the y value from the previous column
                lcolumn_available.insert(column, column)
                ldiaphragm_pts.append((column, ldiaphragm_pts[-1][1]))
            column += 1
        lcolumn = []

        # Draw diaphragmatic level
        diaphragmatic_lvl = np.zeros((256, 50, 3), np.uint8)
        j = 0
        while(j < len(lcolumn_available) - 1):
            cv2.line(
                diaphragmatic_lvl,
                ldiaphragm_pts[j], ldiaphragm_pts[j + 1],
                (0, 0, 255), 1)
            j = j + 1

        print("Diaphragmatic's points: ", ldiaphragm_pts)
        cv2.imshow('Diaphragmatic level', diaphragmatic_lvl)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite('img2DST_diaphragm_man.png', diaphragmatic_lvl)

        # file = open('{}/{}/{}/points.txt'.format(DIR_2DST_Diaphragm, self.patient, self.plan), 'a')
        # file.write("{}:{}\n".format(files[i], ldiaphragm_pts))
        # file.close()

        # return ldiaphragm_pts

        img2DST_RM = cv2.imread('img2DSTrm.jpg', 1)
        j = 0
        while(j < len(lcolumn_available) - 1):
            cv2.line(
                img2DST_RM,
                ldiaphragm_pts[j], ldiaphragm_pts[j + 1],
                (0, 0, 255), 1)
            j = j + 1
        cv2.imshow('Diaphragmatic level', img2DST_RM)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('img2DST_diaphragm_rm_man.png', img2DST_RM)


if __name__ == '__main__':
    patient = 'Iwasawa'
    plan = 'Sagittal'

    respiratory_func = ExtractRespiratoryPattern(patient, plan)

    filename = 'img2DSTman.png'
    respiratory_func.extract(filename)
