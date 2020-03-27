#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from operator import itemgetter
import os


class ExtractRespiratoryFunction(object):

    """

    Extract diaphragmatic respiratory function from the masks created
    using the ASM algorithm

    Parameters:
    -----------
    patient: string
        Patient's name

    plan: string
        Plan in which the images were extracted - must be:
        'Coronal' or 'Sagittal'

    """

    def __init__(self, patient, plan):
        self.patient = patient
        self.plan = plan
        self.DIR2DST =\
            '/home/handrey/Documents/UDESC/DICOM/2DST_Masks/{}/{}'\
            .format(self.patient, self.plan)
        self.DIR_resp_func =\
            '/home/handrey/Documents/UDESC/DICOM/{}/2DST_diaphragm/{}'\
            .format(self.patient, self.plan)

    def get2DSTs(self):
        files = [name for name in os.listdir(self.DIR2DST)]
        return files

    def extract(self, filename):
        """

        Function to extract respiratory function of 2DST images

        Parameters:
        -----------
        filename: string
            2DST image's name with extension

        """

        img = cv2.imread('{}/{}'.format(self.DIR2DST, filename), 0)

        """
        Find the indices of array elements that are non-zero, i.e,
        find the pixels' positions that represents the respiratory
        functions (he pixels in the respiratory function are brighter).
        """
        color_pts = np.argwhere(img > 70)
        """
        Sorts the pixels according to their x coordenate.
        Obs: np.argwhere inverts x and y, it's like (y, x), because of it,
        the parameter of itemgetter is 1 (to get x coordinate)
        """
        lcolor_pts = sorted(color_pts.tolist(), key=itemgetter(1))

        """
        If there is no pixel representing the respiratory function
        (ie, lighter pixel) it creates an empty image (without any
        respiratory function)
        """
        if len(lcolor_pts) == 0:
            diaphragmatic_lvl = np.zeros((256, 50, 3), np.uint8)

            cv2.imwrite('{}/{}'.format(
                self.DIR_resp_func, filename), diaphragmatic_lvl)

            ldiaphragm_pts = []

            return ldiaphragm_pts

        lordered_pts = []
        # Reverse the coordinates and store the result in lordered_pts list
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
            lordered_pts.append((49, lordered_pts[len(lcolumn_available)][1]))
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

        lcolumn_available = []

        # cv2.imshow('Diaphragmatic level', diaphragmatic_lvl)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('{}/{}'.format(
        #     self.DIR_resp_func, filename), diaphragmatic_lvl)

        return ldiaphragm_pts


"""
if __name__ == '__main__':
    # plan = raw_input("Which plan, Coronal or Sagittal? ")
    # patient = raw_input("Patient's name? ")
    plan = 'Coronal'
    patient = 'Matsushita'
    respiratory_func = ExtractRespiratoryFunction(patient, plan)

    # files = respiratory_func.get2DSTs()
    ldiaphragm_pts = respiratory_func.extract('c-9-176;s-14-115.png')
    print(ldiaphragm_pts)
    print(len(ldiaphragm_pts))
"""
