#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.filters import gaussian
from skimage.segmentation import active_contour
# from skimage.color import rgb2gray

import extract_resp_func


class Smooth(object):

    """

    Smooth respiratory pattern (diaphragmatic respiratory function)
    using Active contour model (Snakes)

    Parameters:
    -----------
    patient: string
        Patient's name

    plan: string
        Plan in which the images were extracted - must be:
        'Coronal' or 'Sagittal'

    References
    ----------
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/active_contour_model.py
    http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour

    """

    def __init__(self, patient, plan):
        self.patient = patient
        self.plan = plan
        self.DIR_2DST_Masks =\
            '/home/handrey/Documents/UDESC/DICOM/2DST_Masks/{}/{}'\
            .format(self.patient, self.plan)

        self.DIR_2DST_DICOM =\
            '/home/handrey/Documents/UDESC/DICOM/2DST_DICOM/{}/{}'\
            .format(self.patient, self.plan)

    def get2DSTs(self):
        """

        Get files' names

        Returns
        -------
        files: list
            List of string with name of each image in the directory

        """
        files = [name for name in os.listdir(self.DIR_2DST_Masks)]
        return files

    def snake2DST(self, filename):
        """
        Apply snake algorithm to smooth respiratory function

        Parameters
        ----------
        filname: string
            Name of 2DST file to smooth the respiratory function

        Returns
        -------
        snake: list
            Optimised snake, respiratory function smoothed

        """
        img = cv2.imread('{}/{}'.format(self.DIR_2DST_DICOM, filename), 0)

        # Recover respiratory function extracted from 2DST images using masks
        respiratory_func = extract_resp_func\
            .ExtractRespiratoryFunction(self.patient, self.plan)
        # ldiaphragm_pts = respiratory_func.extract('c-5-90;s-6-102.png')
        ldiaphragm_pts = respiratory_func.extract('c-5-72;s-3-102.png')
        x = np.asarray([_x[0] for _x in ldiaphragm_pts])
        y = np.asarray([_y[1] for _y in ldiaphragm_pts])
        # init = np.array([x, y]).T
        init = np.column_stack((x, y))

        # xp = [(0, 177), (2, 181), (3, 183), (5, 179), (8, 176),
        #       (10, 176), (12, 178), (13, 178), (16, 177), (18, 175),
        #       (21, 176), (23, 178), (25, 179), (27, 177), (29, 176),
        #       (31, 178), (33, 181), (35, 179), (37, 177), (39, 176),
        #       (42, 176), (44, 178), (45, 181), (47, 182), (49, 179)]
        # xp = [(0, 170), (1, 175), (2, 181), (3, 183), (4, 181),
        #       (5, 179), (6, 178), (7, 177), (8, 176), (9, 176),
        #       (10, 176), (11, 177), (12, 178), (13, 178), (14, 178),
        #       (15, 177), (16, 177), (17, 176), (18, 175), (19, 175),
        #       (20, 176), (21, 176), (22, 178), (23, 178), (24, 179),
        #       (25, 179), (26, 178), (27, 177), (28, 177), (29, 176),
        #       (30, 177), (31, 178), (32, 179), (33, 181), (34, 180),
        #       (35, 179), (36, 178), (37, 177), (38, 177), (39, 176),
        #       (40, 176), (41, 176), (42, 176), (43, 177), (44, 178),
        #       (45, 181), (46, 181), (47, 182), (48, 175), (49, 170)]
        # x = np.asarray([x[0] for x in xp])
        # y = np.asarray([y[1] for y in xp])

        # Create a horizontal line as initial snake
        # x = np.linspace(0, 49, 50)     # x.shape = (50,)
        # y = np.linspace(160, 160, 50)  # y.shape = (50,)
        # init = np.array([x, y]).T      # init.shape = (50, 2)

        # Snake algorithm
        snake = active_contour(
            image=gaussian(img, 1),
            snake=init,
            bc='fixed',
            alpha=0.001, beta=1.0, w_line=-5, w_edge=70, gamma=0.5,
            max_px_move=0.5, convergence=0.1, max_iterations=2500)

        # Round points found by active_contour function
        x_axis = map(lambda x: int(round(x)), snake[:, 0].tolist())
        y_axis = map(lambda y: int(round(y)), snake[:, 1].tolist())

        # List of tuples of round points that represents respiratory function
        ldiaphragm_pts = [(x_axis[i], y_axis[i]) for i in range(len(x_axis))]
        print(ldiaphragm_pts)

        # Plot the snake (result)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=1)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()

        # OpenCV
        # initial_curve = init.tolist()
        # for i in range(len(ldiaphragm_pts)):
        #     img[row, column[i]] = (255, 255, 0)
        # cv2.imshow('2DST', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('{}.jpg'.format(column), res)

        return ldiaphragm_pts

    def draw_line(self, pts):
        """

        Create an image with only respiratory function

        Parameters:
        -----------
        pts: list
            List of tuples with points that represents respiratory function

        """
        diaphragmatic_lvl = np.zeros((256, 50, 3), np.uint8)
        j = 0
        while(j < len(pts) - 1):
            cv2.line(
                diaphragmatic_lvl,
                pts[j], pts[j + 1],
                (0, 0, 255), 1)
            j = j + 1
        cv2.imshow('Diaphragmatic level', diaphragmatic_lvl)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite('name.png', diaphragmatic_lvl)


if __name__ == '__main__':
    # plan = raw_input("Which plan, Coronal or Sagittal? ")
    # patient = raw_input("Patient's name? ")
    plan = 'Sagittal'
    patient = 'Matsushita'

    resp_func = Smooth(patient, plan)

    limages = resp_func.get2DSTs()

    # Coronal
    # alpha=0.001, beta=1.0, w_line=-5, w_edge=70, gamma=0.5
    # filename = 'c-4-96;s-7-98.jpg'  # y = 170
    # filename = 'c-4-103;s-8-98.jpg'  # y = 170

    # Sagittal
    # filename = 'c-5-90;s-6-102.jpg'
    # filename = 'c-6-96;s-7-105.jpg'
    filename = 'c-5-72;s-3-102.jpg'

    # filename = 'c-5-90;s-6-102.jpg' #  Exemplo Relatorio
    # filename = 'c-17-72;s-3-142.jpg'
    ldiaphragm_pts = resp_func.snake2DST(filename)

    # resp_func.draw_line(ldiaphragm_pts)
