#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from geomdl import BSpline
from geomdl import utilities
from util.constant import *


class SmoothRespiratoryPattern(object):
    """
    Smooth respiratory pattern (diaphragmatic respiratory function)
    using Active contour model (Snakes)

    Parameters
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

    def get2DSTs(self):
        """
        Get files' names

        Returns
        -------
        files: list
            List of string with name of each image in the directory
        """
        files = [name for name in os.listdir('{}/{}/{}'.format(DIR_2DST_Mask, self.patient, self.plan))]
        return files

    def respiratory_pattern(self, filename):
        """
        Function to extract respiratory function of 2DST images

        Parameters:
        -----------
        filename: string
            2DST image's name with extension
        """
        img = cv2.imread('{}/{}/{}/{}'.format(DIR_2DST_Mask, self.patient, self.plan, filename), 0)

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

        lcolumn_available = []

        return ldiaphragm_pts

    def save_respiratory_pattern(self, filename, points):
        """
        Creates a blank image and draw only the respiratory pattern

        Parameters
        ----------
        points: list of tuples (each tuple is a point)
            List of diaphragm's points
        """

        # Draw diaphragmatic level
        diaphragm_pattern = np.zeros((256, 50, 3), np.uint8)
        j = 0
        # while(j < len(lcolumn_available) - 1):
        while(j < 49):
            cv2.line(
                diaphragm_pattern,
                points[j], points[j + 1],
                (0, 0, 255), 1)
            j = j + 1

        cv2.imshow('Diaphragmatic level', diaphragm_pattern)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('{}/{}/{}/{}'.format(
            DIR_2DST_Smooth, self.patient, self.plan, filename), diaphragm_pattern)

    def smooth_snake(self, filename):
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
        img = cv2.imread('{}/{}/{}/{}'.format(DIR_2DST_DICOM, self.patient, self.plan, filename), 0)

        # Recover respiratory function extracted from 2DST images using masks
        ldiaphragm_pts = self.respiratory_pattern(filename)
        # print("Original diaphragm points: {}".format(ldiaphragm_pts))
        # print("")

        x = np.asarray([_x[0] for _x in ldiaphragm_pts])
        y = np.asarray([_y[1] for _y in ldiaphragm_pts])
        # init = np.array([x, y]).T
        init = np.column_stack((x, y))

        # Snake algorithm - http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour
        # snake = active_contour(
        #     image=gaussian(img, 1),
        #     snake=init,
        #     bc='fixed',
        #     alpha=0.001, beta=1.0, w_line=-5, w_edge=70, gamma=0.5,
        #     max_px_move=0.5, convergence=0.1, max_iterations=2500)
        snake = active_contour(
            image=gaussian(img, 5),
            snake=init,
            bc='fixed',
            alpha=0.001, beta=0.01, w_line=-5, w_edge=70, gamma=0.5,
            max_px_move=0.5, convergence=0.1, max_iterations=2500)
        # print("Original snake points: ",
        #       [(round(snake[:, 0].tolist()[i], 2),
        #         round(snake[:, 1].tolist()[i], 2))
        #        for i in range(len(snake[:, 0].tolist()))])

        # Round points found by active_contour function
        x_axis = list(map(lambda x: int(round(x)), snake[:, 0]))
        y_axis = list(map(lambda y: int(round(y)), snake[:, 1]))

        # List of tuples of round points that represents respiratory function
        ldiaphragm_pts_smoothed = [(x_axis[i], y_axis[i]) for i in range(len(x_axis))]
        # print("Smoothed diaphragm points: {}".format(ldiaphragm_pts_smoothed))

        # Plot the snake (result)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()

        # fig, ax = plt.subplots(figsize=(6, 5))
        # ax.imshow(img, cmap=plt.cm.gray)
        # ax.plot(snake[:, 0], snake[:, 1], '-b', lw=1)
        # ax.set_xticks([]), ax.set_yticks([])
        # ax.axis([0, img.shape[1], img.shape[0], 0])
        # plt.show()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=1)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()

        return ldiaphragm_pts, ldiaphragm_pts_smoothed

    def smooth_bspline(self, filename):
        """
        Smooth respiratory function using B-spline curve

        Parameters
        ----------
        filname: string
            Name of 2DST file to smooth the respiratory function

        Returns
        -------
        snake: list
            Optimised B-spline curve (respiratory function smoothed)
        """

        # Recover respiratory function extracted from 2DST images using masks
        ldiaphragm_pts = [list(item) for item in self.respiratory_pattern(filename)]
        # print("Original diaphragm points: {}".format(ldiaphragm_pts))

        # img = cv2.imread('{}/{}/{}/{}'.format(DIR_2DST_DICOM, self.patient, self.plan, filename), 1)
        # diaphragm_mask = np.zeros((256, 50, 3), np.uint8)
        # img_mask = img.copy()
        # i = 0
        # while(i < len(ldiaphragm_pts) - 1):
        #     cv2.line(
        #         img_mask,
        #         ldiaphragm_pts[i], ldiaphragm_pts[i + 1],
        #         (255, 0, 0), 1)
        #     i = i + 1
        # cv2.imshow('Diaphragmatic level', img_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('name.png', img_mask)

        # Try to load the visualization module
        try:
            render_curve = True
            from geomdl.visualization import VisMPL
        except ImportError:
            render_curve = False

        # Create a B-Spline curve instance
        curve = BSpline.Curve()

        # Set evaluation delta
        curve.delta = 0.0002

        """ Set up curve """
        # Set control points
        controlpts = ldiaphragm_pts
        convert_controlpts = map(lambda x: 255 - x[1], controlpts)
        for i in range(len(controlpts)):
            controlpts[i][1] = convert_controlpts[i]

        curve.ctrlpts = controlpts

        # Set curve degree
        curve.degree = 3

        # Auto-generate knot vector
        curve.knotvector =\
            utilities.generate_knot_vector(
                curve.degree, len(curve.ctrlpts))

        # Evaluate curve
        curve.evaluate()

        # Draw the control point polygon and the evaluated curve
        if render_curve:
            vis_comp = VisMPL.VisCurve2D()
            curve.vis = vis_comp
            curve.render()

        # return ldiaphragm_pts, ldiaphragm_pts_smoothed

    def draw_line(self, diaphragm_pts_mask, diaphragm_pts_smoothed, filename):
        """
        Create an image with only respiratory function

        Parameters:
        -----------
        pts: list
            List of tuples with points that represents respiratory function

        filename: string
            Name of 2DST file to smooth the respiratory function
        """
        img = cv2.imread('{}/{}/{}/{}'.format(DIR_2DST_DICOM, self.patient, self.plan, filename), 1)

        # diaphragm_mask = np.zeros((256, 50, 3), np.uint8)
        img_mask = img.copy()
        i = 0
        while(i < len(diaphragm_pts_mask) - 1):
            cv2.line(
                img_mask,
                diaphragm_pts_mask[i], diaphragm_pts_mask[i + 1],
                (255, 0, 0), 1)
            i = i + 1
        # cv2.imshow('Diaphragmatic level', diaphragm_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('name.png', diaphragm_mask)

        # diaphragm_smoothed = np.zeros((256, 50, 3), np.uint8)
        img_smoothed = img.copy()
        j = 0
        while(j < len(diaphragm_pts_smoothed) - 1):
            cv2.line(
                img_smoothed,
                diaphragm_pts_smoothed[j], diaphragm_pts_smoothed[j + 1],
                (0, 255, 0), 1)
            j = j + 1

        # image = img.copy()
        # k = 0
        # while(k < len(diaphragm_pts_smoothed) - 1):
        #     cv2.line(
        #         image,
        #         diaphragm_pts_mask[k], diaphragm_pts_mask[k + 1],
        #         (255, 0, 0), 1)
        #     cv2.line(
        #         image,
        #         diaphragm_pts_smoothed[k], diaphragm_pts_smoothed[k + 1],
        #         (0, 255, 0), 1)
        #     k = k + 1

        titles = ['Original', 'Mask', 'Snake']
        # images = [img, diaphragm_mask, diaphragm_smoothed, image]
        images = [img, img_mask, img_smoothed]
        for i in range(3):
            plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')  # Line / Column
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == '__main__':
    try:
        patient = 'Iwasawa'
        plan = 'Sagittal'

        txtargv2 = '-patient={}'.format(patient)
        txtargv3 = '-plan={}'.format(plan)

        if len(sys.argv) > 1:
            txtargv2 = sys.argv[1]
            if len(sys.argv) > 2:
                txtargv3 = sys.argv[2]

        txtargv = '{}|{}'.format(txtargv2, txtargv3)

        if txtargv.find('-patient') != -1:
            txttmp = txtargv.split('-patient')[1]
            txttmp = txttmp.split('=')[1]
            patient = txttmp.split('|')[0]

        if txtargv.find('-plan') != -1:
            txttmp = txtargv.split('-plan')[1]
            txttmp = txttmp.split('=')[1]
            plan = txttmp.split('|')[0]
    except ValueError:
        print("""
        Examples of use:

        $ python {} -patient=Iwasawa -plan=Sagittal

        Parameters:

        patient = Iwasawa -> Patient's name
        plan = Sagittal
        """.format(sys.argv[0]))
        exit()

    resp_pattern = SmoothRespiratoryPattern(patient, plan)

    files = resp_pattern.get2DSTs()

    for file in files:
        ldiaphragm_pts, ldiaphragm_pts_smoothed =\
            resp_pattern.smooth_snake(file)
        # resp_pattern.draw_line(
        #     ldiaphragm_pts, ldiaphragm_pts_smoothed, file)
        # resp_pattern.save_respiratory_pattern(
        #     file, ldiaphragm_pts_smoothed)

    # resp_pattern.smooth_bspline('c-9-70;s-1-134.png')

    # ldiaphragm_pts, ldiaphragm_pts_smoothed = resp_pattern.smooth_snake('c-9-70;s-1-134.png')
    # print(ldiaphragm_pts)
    # print(len(ldiaphragm_pts))

    # resp_pattern.save_respiratory_pattern('c-9-70;s-1-134.png', ldiaphragm_pts_smoothed)

    # resp_pattern.draw_line(ldiaphragm_pts, ldiaphragm_pts_smoothed, 'c-9-70;s-1-134.png')
