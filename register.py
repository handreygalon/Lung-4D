#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
from operator import itemgetter

from util.constant import *
from plot_lung import Reconstruction


class Register(object):
    def __init__(self, patient, lung, corsequences, lsagsequences, rsagsequences):
        self.patient = patient
        self.lung = lung
        self.corsequences = corsequences
        self.lsagsequences = lsagsequences
        self.rsagsequences = rsagsequences
        self.pc = Reconstruction(patient)

        self.matDL = None  # Matrix with diaphragmatic level info
        self.matRP = None  # Matrix with respiratory phase info

        """
        matRegistration: Matrix that contains registration information of each crossing position
        The vertical and horizontal lines represent the coronal and sagittal images respectively
        """
        self.matRegistration = None
        self.blue = 1
        self.green = 2
        self.yellow = 3

        # Read the file which contains the names of the images referring to respiratory patterns
        self.dataset = open('mapping.txt', 'r').read().split('\n')
        self.dataset.pop(-1)
        # print("Dataset: {} ({})\n".format(self.dataset, len(self.dataset)))

        if self.lung == 0:
            # Create a list of which sequences are available in the sagittal and coronal planes
            self.cor_sequences = list(set([int(i.split('-')[1]) for i in self.dataset]))
            self.sag_sequences = list(set([int(i.split('-')[3]) for i in self.dataset]))
            self.sag_sequences = self.sag_sequences[:len(self.lsagsequences)]
            # print("Coronal sequences: {} ({})".format(self.cor_sequences, len(self.cor_sequences)))
            # print("Sagittal sequences: {} ({}) \n ".format(self.sag_sequences, len(self.sag_sequences)))

            # Create list with the positions of intersections between the coronal and sagittal plans
            self.cor_columns =\
                sorted(list(set([int(i.split(';')[0].split('-')[2]) for i in self.dataset])))
            self.cor_columns = self.cor_columns[:len(self.lsagsequences)]
            self.sag_columns =\
                sorted(list(set([int(i.split(';')[1].split('-')[2]) for i in self.dataset])))
            # print("Coronal columns: {} ({})".format(self.cor_columns, len(self.cor_columns)))
            # print("Sagittal columns: {} ({})\n".format(self.sag_columns, len(self.sag_columns)))

            # Matrices' dimensions
            self.matrows = len(self.cor_columns)
            self.matcols = len(self.sag_columns)
            # print("Matrix's rows number: {}\n".format(self.matrows))
            # print("Matrix's columns number: {}".format(self.matcols))
        else:
            # Create a list of which sequences are available in the sagittal and coronal planes
            self.cor_sequences = list(set([int(i.split('-')[1]) for i in self.dataset]))
            self.sag_sequences = list(set([int(i.split('-')[3]) for i in self.dataset]))
            self.sag_sequences = self.sag_sequences[-(len(self.rsagsequences)):]
            # print("Coronal sequences: {} ({})".format(self.cor_sequences, len(self.cor_sequences)))
            # print("Sagittal sequences: {} ({}) \n ".format(self.sag_sequences, len(self.sag_sequences)))

            # Create list with the positions of intersections between the coronal and sagittal plans
            self.cor_columns =\
                sorted(list(set([int(i.split(';')[0].split('-')[2]) for i in self.dataset])))
            self.cor_columns = self.cor_columns[-(len(self.rsagsequences)):]
            self.sag_columns =\
                sorted(list(set([int(i.split(';')[1].split('-')[2]) for i in self.dataset])))
            # print("Coronal columns: {} ({})".format(self.cor_columns, len(self.cor_columns)))
            # print("Sagittal columns: {} ({})\n".format(self.sag_columns, len(self.sag_columns)))

            # Matrices' dimensions
            self.matrows = len(self.cor_columns)
            self.matcols = len(self.sag_columns)
            # print("Matrix's rows number: {}\n".format(self.matrows))
            # print("Matrix's columns number: {}".format(self.matcols))

    def pattern_coronal(self, file):
        """
        Extract respiratory pattern from coronal plan

        Parameters
        ----------
        file: string
            Name of image that have the only respiratory diaphragmatic pattern

        Return
        ------
        lpts: list
            List of tuples that have all points of respiratory diaphragmatic
            pattern
        """
        img = cv2.imread('{}/{}/Coronal/{}'.format(DIR_2DST_Diaphragm, self.patient, file), 0)

        """ Find the indices of array elements that are non-zero
            Find the pixels' positions that represents the respiratory function
            The pixels in the respiratory function are brighter """
        pts = np.argwhere(img > 70).tolist()

        """ When argwhere was use, the coordinate x and y was inverted,
            because of it, its necessary to reverse the coordinates and store
            the result in ordered_pts list """
        ordered_pts = [pts[i][::-1] for i in range(len(pts))]

        # Sorts the pixels according to their x coordenate
        return sorted(ordered_pts, key=itemgetter(0))

    def pattern_sagittal(self, file):
        img = cv2.imread('{}/{}/Sagittal/{}'.format(DIR_2DST_Diaphragm, self.patient, file), 0)

        pts = np.argwhere(img > 70).tolist()

        ordered_pts = [pts[i][::-1] for i in range(len(pts))]

        return sorted(ordered_pts, key=itemgetter(0))

    def diaphragmatic_level_coronal(self, pts):
        """
        Extract only diaphragmatic level, i.e., get the value of y coordinate
        of each column of 2DST image

        Parameters
        ----------
        pts: list
            List of coronal points (pixel's coordinate) that represents the
            respiratory pattern

        Return
        ------
        diaphragmatic_lvl: list
            List of lists. Each column has a list of points that represents
            diaphragmatic level, because sometimes, the curve that represents
            the respiratory pattern occupy more than one pixel on each column
        """

        # List with just the y coordenate (represents the diaphragmatic level)
        diaphragmatic_lvl = []

        column = 0
        # Each column has a list that represents the diaphragmatic level
        for i in range(50):
            lcolumn_lvl = [lvl[1] for lvl in pts if lvl[0] == column]
            diaphragmatic_lvl.append(lcolumn_lvl)
            column += 1

        return diaphragmatic_lvl

    def diaphragmatic_level_sagittal(self, pts):
        diaphragmatic_lvl = []

        column = 0
        for i in range(50):
            lcolumn_lvl = [lvl[1] for lvl in pts if lvl[0] == column]
            diaphragmatic_lvl.append(lcolumn_lvl)
            column += 1

        return diaphragmatic_lvl

    def respiratory_phase_coronal(self, diaphragmatic_lvl):
        """
        Initially a classification of respiratory phases is made, by the
        difference between the current and the previous diaphragmatic level.
        If the difference is positive, the individual is in inspiration,
        otherwise, it is in expiration

        Parameters
        ----------
        diaphragmatic_lvl: list
            List of lists thar represents the diaphragmatic level of
            each column on 2DST image

        Return: list
            List of respiratory phase of each image (each column on 2DST
            image)
            0 - Expiration
            1 - Inspiration
        """
        def mean(numbers):
            """
            How there are more than one pixel in the same column that represent
            the diaphragmatic lvl, its necessary to calc the mean of the values

            Parameters
            ----------
            numbers: list
                List that contain all the column points that represent the
                diaphragmatic level
            """
            return float(sum(numbers)) / float(len(numbers))

        lphase = []

        """ Compare the current diaphragmatic level to the previous one.
            The first level take the same respiratory phase that the second,
            because there is no previous level to compare """
        if mean(diaphragmatic_lvl[1]) - mean(diaphragmatic_lvl[0]) >= 0:
            lphase.append(1)
        else:
            lphase.append(0)

        """ The others level are extracted by comparing the current diaphragmatic
            level to the previous one """
        for lvl in range(len(diaphragmatic_lvl) - 1):
            if mean(diaphragmatic_lvl[lvl + 1]) -\
               mean(diaphragmatic_lvl[lvl]) >= 0:
                lphase.append(1)
            else:
                lphase.append(0)

        return lphase

    def respiratory_phase_sagittal(self, diaphragmatic_lvl):
        def mean(numbers):
            return float(sum(numbers)) / float(len(numbers))

        lphase = []

        if mean(diaphragmatic_lvl[1]) - mean(diaphragmatic_lvl[0]) >= 0:
            lphase.append(1)
        else:
            lphase.append(0)

        for lvl in range(len(diaphragmatic_lvl) - 1):
            if mean(diaphragmatic_lvl[lvl + 1]) -\
               mean(diaphragmatic_lvl[lvl]) >= 0:
                lphase.append(1)
            else:
                lphase.append(0)

        return lphase

    def read_points(self, plan, sequence, side=0, imgnumber=1):
        """ Read points from txt file

        Parameters
        ----------
        plan: string
            Represents from which plane the txt file belongs. Must be 'Coronal' or 'Sagittal'
        sequence: int
            Represents from which sequence the txt file belongs
        side: int
            In case the file belongs to the coronal plan: 0 - left lung, 1 - right lung
        imgnumber: int
            Represents the image (instant)

        Return
        ------
        Points representing the lung contour of a specific respiratory time """
        if plan == 'Coronal':
            if self.lung == 0:
                dataset =\
                    open('{}/{}/{}/{}_L/points.txt'.format(
                        DIR_MAN_LUNG_MASKS,
                        self.patient,
                        plan,
                        sequence,
                        self.lung), 'r').read().split('\n')

                del dataset[50:]
            elif self.lung == 1:
                dataset =\
                    open('{}/{}/{}/{}_R/points.txt'.format(
                        DIR_MAN_LUNG_MASKS,
                        self.patient,
                        plan,
                        sequence,
                        self.lung), 'r').read().split('\n')

                del dataset[50:]
        else:
            dataset =\
                open('{}/{}/{}/{}/points.txt'.format(
                    DIR_MAN_LUNG_MASKS,
                    self.patient,
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

        return all_points[imgnumber - 1]  # -1

    def convert_points(self, step, rootsequence, currentsequence, imgnum):
        """ Converts 2D points to 3D space

        Parameters
        ----------
        step: int
            Represents the registration step that is being performed
        rootsequence: int
            Represents the coronal root sequence
        currentsequence: int
            Represents the current sequence being parsed
        imgnum: int
            Represents the image number of the current sequence

        Return
        ------
        X: list
            Represents a list of all points in x coordinates
        Y: list
            Represents a list of all points in y coordinates
        Z: list
            Represents a list of all points in z coordinates
        lpts: list
            Represents a list of (x, y, z) points """

        if step == 1:
            lpts = self.read_points(plan='Coronal', sequence=rootsequence, side=self.lung, imgnumber=imgnum)
            # print("Image's points: {} ({})\n".format(lpts, len(lpts)))

            # X, Y, Z = self.pc.point3D(plan='Coronal', sequence=currentsequence, imgnum=imgnum, pts=lpts)
            X, Y, Z = self.pc.point3D(plan='Coronal', sequence=rootsequence, imgnum=imgnum, pts=lpts)

        elif step == 2:
            lpts = self.read_points(plan='Sagittal', sequence=currentsequence, side=self.lung, imgnumber=imgnum)

            X, Y, Z = self.pc.point3D(plan='Sagittal', sequence=currentsequence, imgnum=imgnum, pts=lpts)

        elif step == 3:
            lpts = self.read_points(plan='Coronal', sequence=currentsequence, side=self.lung, imgnumber=imgnum)

            X, Y, Z = self.pc.point3D(plan='Coronal', sequence=currentsequence, imgnum=imgnum, pts=lpts)

        else:
            print("Invalid step value (must be between 1 and 3).")

        # Create a list of the points
        lpts = []
        for i in range(len(X)):
            pt = (X[i], Y[i], Z[i])
            lpts.append(pt)
        # print('Points: {} ({})\n'.format(lpts, len(lpts)))

        def createFile(step, currentsequence, imgnum, points):
            # Saves register information about an respiration instant in a .txt file
            instant_information = list()
            instant_information.append(self.patient)
            if step == 1 or step == 3:
                instant_information.append('Coronal')
            else:
                instant_information.append('Sagittal')
            instant_information.append(currentsequence)
            instant_information.append(imgnum)
            instant_information.append(step)
            instant_information.append(lpts)
            file = open('{}/{}-{}.txt'.format(DIR_RESULT, rootsequence, imgnumber), 'a')  # Coronal sequence (root) - image's number (instant)
            file.write("{}\n".format(instant_information))
            file.close()

        if step == 1 or step == 2:
            createFile(step, currentsequence, imgnum, lpts)
        else:
            if currentsequence != rootsequence:
                createFile(step, currentsequence, imgnum, lpts)

        # return lpts
        return X, Y, Z, lpts

    def first_step(self, plan, sequence, imgnum):
        """ Root image (coronal) is defined

        Parameters:
        ----------
        plan: string
            For now, must be coronal. Represents plan of the root image
        sequence: int
            Represents the sequence number of root image
        imgnum: int
            Image's number of the selected sequence (represents the respiratory instant)
        Retrun
        ------
        dlvl: list
            List of integers of the diaphragmatic level at the crossing points
        rphase: list
            List of integers (0 or 1) representing the respiratory phase associated with
            the root image
         """

        def max_diaphragmatic_level(levels):
            """
            As a column in the 2DST image may have more than one point representing the
            diaphragmatic level, this function retrieves only the biggest point, which
            will represent the diaphragmatic level

            Parameters
            ----------
            levels: list
                List of lists of int that represents the diafragmatics levels
            """
            # print(levels)
            # print(len(levels))
            return [max(x) for x in levels]

        # Define the dimension of the matrixes
        mat_dim = (len(self.sag_sequences), len(self.cor_sequences))
        self.matRegistration = np.zeros(mat_dim)
        self.matX = np.zeros(mat_dim)
        self.matY = np.zeros(mat_dim)
        self.matZ = np.zeros(mat_dim)
        self.matDL = np.zeros(mat_dim)
        self.matRP = np.zeros(mat_dim)

        # Respiratory patterns linked to the sequence containing the root image
        lpatterns = [x for x in self.dataset if int(x.split('-')[1]) == sequence]
        # print('Patterns coronal: {} ({})\n'.format(lpatterns, len(lpatterns)))

        # Represents the column in the register matrix that represents the sequence used
        column = self.cor_sequences.index(sequence)
        # print('Column: {}\n'.format(column))

        """ Retrieves the diaphragmatic levels of the root image (coronal plan) and stores
            them in the correct positions of the register matrix """
        for i in range(self.matrows):
            # print('{}.png\n'.format(lpatterns[i]))

            pts_pattern = self.pattern_coronal('{}.png'.format(lpatterns[i]))
            # print("{} ({})\n".format(pts_pattern, len(pts_pattern)))
            diaph_lvl = self.diaphragmatic_level_coronal(pts_pattern)
            max_diaph_lvl = max_diaphragmatic_level(diaph_lvl)
            resp_phase = self.respiratory_phase_coronal(diaph_lvl)

            self.matDL[i, column] = max_diaph_lvl[imgnum - 1]
            self.matRP[i, column] = resp_phase[i]
            self.matRegistration[i, column] = self.blue

        # print("(Step 1) Diaphragmatic level matrix:\n{}\n".format(self.matDL))
        # print("(Step 1) Registration matrix:\n{}\n".format(self.matRegistration))
        # print("(Step 1) Respiratory phase:\n{}\n".format(self.matRP))

        dlvl = [self.matDL[i, column] for i in range(len(self.sag_sequences))]
        rphase = [int(self.matRP[i, column]) for i in range(len(self.sag_sequences))]
        # print("(Step 1) DL: {}\n".format(dlvl))
        # print("(Step 1) RP: {}\n".format(rphase))

        return dlvl, rphase

    def second_step(self, plan, dlvl_root_img, rphase_root_img, root_sequence, sag_sequence):
        """ It finds sagittal images that have the same diaphragmatic level as the coronal root sequence

        Parameters:
        ----------
        plan: string
            Must be sagittal. Represents plan orthogonal to the root image
        dlvl_root_img: list
            List of integers representing the diaphragmatic level at crossing points
        rphase_root_img: list
            List of integers representing the repiratpry phase at crossing points
        root_sequence: int
            Represents which sequence belong the root image
        sag_sequence: int
            Sagittal sequence being analyzed

        Retrun
        ------
        imgnum: int
            imgnum is the index of the image in the sequence. It's need to add 1 to get the real
            number of the image, bacause the index starts in 0.
        dlvl: list
            List of integers of the diaphragmatic level at the crossing points
        rphase: list
            List of integers (0 or 1) representing the respiratory phase associated with
            the sagittal image registered with the root coronal image """

        # Sagittal respiratory patterns associated with sagittal sequence analyzed
        lpatterns =\
            [x for x in self.dataset if int(x.split('-')[3]) == self.sag_sequences[self.sag_sequences.index(sag_sequence)]]
        # print('(Step 2) Sagittal patterns: {} ({})\n'.format(lpatterns, len(lpatterns)))

        # Respiratory patterns linked to the sequence containing the root image
        pattern = [p for p in lpatterns if int(p.split('-')[1]) == root_sequence][0]
        # print("(Step 2) Pattern: {}".format(pattern))

        """ Get the diaphragmatic level of each image of the analyzed sagittal sequence
            that crosses the coronal root image """
        pts_pattern = self.pattern_sagittal('{}.png'.format(pattern))
        diaph_lvl = [max(x) for x in self.diaphragmatic_level_sagittal(pts_pattern)]
        resp_phase = self.respiratory_phase_sagittal(self.diaphragmatic_level_sagittal(pts_pattern))
        # print("(Step 2) Diaphragmatic level: {} ({})\n".format(diaph_lvl, len(diaph_lvl)))
        # print("(Step 2) Respiratory phase: {} ({})\n".format(resp_phase, len(resp_phase)))

        """ Check register condition: 1) If there is same diaphragmatic level """
        index_imgs_registered = list()  # Store index of the sagittal registered images
        for index, i in enumerate(diaph_lvl):
            if i == dlvl_root_img[self.sag_sequences.index(sag_sequence)]:
                index_imgs_registered.append(index)
        # print("(Step 2) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))

        # If there is no registered image, do the second attempt
        if len(index_imgs_registered) == 0:
            """ If there are no records, there is the possibility of an error in the segmentation
                (which is done manually), then analyzing the respiratory phase is added in one unit
                the value of the diaphragmatic level (inspiration case) or subtracted one unit
                (expiration case) """

            index_imgs_registered =\
                self.second_step_second_attempt(
                    diaph_lvl=diaph_lvl,
                    dlvl_root_img=dlvl_root_img,
                    rphase_root_img=rphase_root_img,
                    sag_sequence=sag_sequence,
                    option=True)

            if len(index_imgs_registered) == 0:
                return -1, -1, -1

        # Get first sagittal image that was registered with root image
        imgnum = index_imgs_registered[0]
        # print("(Step 2) Image index: {}\n".format(imgnum))

        """ Represents the row in the register matrix that represents the sequence used
            It's used to populate the registration matrix correctly """
        row = self.sag_sequences.index(sag_sequence)
        # print('(Step 2) Row: {}\n'.format(row))

        for i in range(self.matcols):
            pts_pattern = self.pattern_sagittal('{}.png'.format(lpatterns[i]))
            # if len(pts_pattern) == 0:
            #     print("{}".format(lpatterns[i]))
            diaph_lvl = [max(x) for x in self.diaphragmatic_level_sagittal(pts_pattern)]
            resp_phase =\
                self.respiratory_phase_sagittal(self.diaphragmatic_level_sagittal(pts_pattern))

            if self.matDL[row, i] == 0.0:
                self.matDL[row, i] = diaph_lvl[imgnum]
                self.matRegistration[row, i] = self.green
                self.matRP[row, i] = resp_phase[i]

        # print("(Step 2) Diaphragmatic level matrix:\n{}\n".format(self.matDL))
        # print("(Step 2) Registration matrix:\n{}\n".format(self.matRegistration))
        # print("(Step 2) Respiratory phase:\n{}\n".format(self.matRP))

        imgnum = imgnum + 1
        dlvl = [self.matDL[row, i] for i in range(len(self.cor_sequences))]
        rphase = [int(self.matRP[row, i]) for i in range(len(self.cor_sequences))]

        return imgnum, dlvl, rphase

    def second_step_second_attempt(self, diaph_lvl, dlvl_root_img, rphase_root_img, sag_sequence, option=False):
        """
        If there are no registers, there is the possibility of an error in the segmentation
        (which is done manually), then analyzing the respiratory phase is added in one unit
        the value of the diaphragmatic level (inspiration case) or subtracted one unit
        (expiration case)

        Parameters
        ----------
        diaph_lvl: list
            Diaphragmatic level of the 50 images of the analyzed sagittal sequence that crosses the coronal root image
        dlvl_root_img:
            Diaphragmatic level of the root instant
        rphase_root_img:
            Respiratory phase of the root instant
        sag_sequence: int
            Sagittal sequence being analyzed
        option: bool
            If True, check images to register up to 3 difference units
            If False, check images to register up to 1 difference units """
        print("Second attempt (Second Step)")
        # print("DL: {} ({})".format(diaph_lvl, len(diaph_lvl)))

        index_imgs_registered = list()  # Store index of the sagittal registered images
        variation_lvl = 2  # Variation of the diaphragmatic level

        if option:
            for lvl in range(1, variation_lvl):
                # Current coronal (root) diaphragmatic level being analyzed
                diaph_lvl_root = dlvl_root_img[self.sag_sequences.index(sag_sequence)]
                resp_phase_root = rphase_root_img[self.sag_sequences.index(sag_sequence)]
                # print("DL: {} ({})".format(diaph_lvl_root, resp_phase_root))
                if resp_phase_root == 1:
                    diaph_lvl_root += lvl
                else:
                    diaph_lvl_root -= lvl
                # print("Diaph. lvl: {}\n".format(diaph_lvl_root))

                for index, i in enumerate(diaph_lvl):
                    if i == diaph_lvl_root:
                        index_imgs_registered.append(index)
                # print("(Step 2 - Second attempt) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))
                # c = raw_input("?")
                if len(index_imgs_registered) > 0:
                    break

            # if still there is no image registered, invert the calc of the diaphragmatic level and try again
            if len(index_imgs_registered) == 0:
                for lvl in range(1, variation_lvl):
                    # Current coronal (root) diaphragmatic level being analyzed
                    diaph_lvl_root = dlvl_root_img[self.sag_sequences.index(sag_sequence)]
                    resp_phase_root = rphase_root_img[self.sag_sequences.index(sag_sequence)]
                    # print("-DL: {} ({})".format(diaph_lvl_root, resp_phase_root))
                    if resp_phase_root == 1:
                        diaph_lvl_root -= lvl
                    else:
                        diaph_lvl_root += lvl
                    # print("-Diaph. lvl: {}\n".format(diaph_lvl_root))

                    for index, i in enumerate(diaph_lvl):
                        if i == diaph_lvl_root:
                            index_imgs_registered.append(index)
                    # print("(Step 2 - Second attempt inv) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))
                    # c = raw_input("?")
                    if len(index_imgs_registered) > 0:
                        break

        else:
            # Current coronal (root) diaphragmatic level being analyzed
            diaph_lvl_root = dlvl_root_img[self.sag_sequences.index(sag_sequence)]
            resp_phase_root = rphase_root_img[self.sag_sequences.index(sag_sequence)]
            # print("DL: {} ({})".format(diaph_lvl_root, resp_phase_root))
            if resp_phase_root == 1:
                diaph_lvl_root += 1
            else:
                diaph_lvl_root -= 1
            # print("Diaph. lvl: {}\n".format(diaph_lvl_root))

            for index, i in enumerate(diaph_lvl):
                if i == diaph_lvl_root:
                    index_imgs_registered.append(index)
            # print("(Step 2 - Second attempt) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))

        # if len(index_imgs_registered) == 0:
        return index_imgs_registered

    def get_root_sagittal_sequence(self, dlvl_sag_img, rphase_sag_img, cor_sequence, sag_sequence):
        """ Calculate the root sagittal sequence (the sagittal sequence that had more registers)

        Parameters
        ----------
        dlvl_sag_img:
            List of integers representing the diaphragmatic level at crossing points
        rphase_sag_img:
            List of integers representing the repiratpry phase at crossing points
        cor_sequence: int
            Coronal sequence being analyzed
        sag_sequence:
            Sagittal sequence being analyzed

        Return
        ------
        index_imgs_registered: list
            List with the images registered """
        lpatterns = [x for x in self.dataset if int(x.split('-')[1]) == cor_sequence]

        pattern = [p for p in lpatterns if int(p.split('-')[3]) == sag_sequence][0]
        # print("Pattern: {}".format(pattern))

        pts_pattern = self.pattern_coronal('{}.png'.format(pattern))
        diaph_lvl = [max(x) for x in self.diaphragmatic_level_coronal(pts_pattern)]
        resp_phase = self.respiratory_phase_coronal(self.diaphragmatic_level_coronal(pts_pattern))

        index_imgs_registered = list()  # Store index of the coronal registered images
        for index, i in enumerate(diaph_lvl):
            if i == dlvl_sag_img[self.cor_sequences.index(cor_sequence)]:
                index_imgs_registered.append(index)
        # print("(Step 3) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))

        for index, i in enumerate(resp_phase):
            if index in index_imgs_registered:
                if resp_phase[index] != rphase_sag_img[self.cor_sequences.index(cor_sequence)]:
                    index_imgs_registered.remove(index)
        # print("(Step 3) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))

        return index_imgs_registered

    def third_step(self, plan, dlvl_sag_img, rphase_sag_img, cor_sequence, sag_sequence):
        """
        Represents the column in the register matrix that represents the coronal sequence used.
        It's used to populate the registration matrix correctly

        Parameters
        ----------
        plan: string
            Must be sagittal. Represents plan orthogonal to the root image

        dlvl_sag_img: list
            List of integers representing the diaphragmatic level at crossing points

        rphase_sag_img: list
            List of integers representing the repiratpry phase at crossing points

        cor_sequence: int
            Represents the coronal sequences parallel to the root coronal sequence

        sag_sequence: int
            Represents which sequence belong the sagittal image (Root sagittal sequence)

        Retrun
        ------
        imgnum: int
            imgnum is the index of the image in the sequence. It's need to add 1 to get the real
            number of the image, bacause the index starts in 0.

        dlvl: list
            List of integers of the diaphragmatic level at the crossing points

        rphase: list
            List of integers (0 or 1) representing the respiratory phase associated with
            the sagittal image registered with the root coronal image
         """
        column = self.cor_sequences.index(cor_sequence)
        # print('(Step 3) Column: {}\n'.format(column))

        # Respiratory patterns linked to the sequence containing the root image
        lpatterns = [x for x in self.dataset if int(x.split('-')[1]) == cor_sequence]
        # print('(Step 3) Patterns coronal: {} ({})\n'.format(lpatterns, len(lpatterns)))

        # Respiratory patterns linked to the sequence containing the root sagittal image
        pattern = [p for p in lpatterns if int(p.split('-')[3]) == sag_sequence][0]
        # print("(Step 3) Pattern: {}\n".format(pattern))
        # print("(Step 3) DL: {}\n".format(dlvl_sag_img))
        # print("(Step 3) DL[pos]: {}\n".format(dlvl_sag_img[column]))

        """ Get the diaphragmatic level of each image of the analyzed coronal sequence (parallel
            to the root coronal sequence) that crosses the sagittal image registered in the second
            step """
        pts_pattern = self.pattern_coronal('{}.png'.format(pattern))
        diaph_lvl = [max(x) for x in self.diaphragmatic_level_coronal(pts_pattern)]
        resp_phase = self.respiratory_phase_coronal(self.diaphragmatic_level_coronal(pts_pattern))
        # print("DL sag: {}\n".format(dlvl_sag_img))
        # print("DL cor: {}\n".format(diaph_lvl))

        """ Check register condition:
            1) If there is same diaphragmatic level """
        index_imgs_registered = list()  # Store index of the coronal registered images
        for index, i in enumerate(diaph_lvl):
            if i == dlvl_sag_img[column]:
                index_imgs_registered.append(index)
        # print("(Step 3) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))

        """ Check register condition:
            2) If the instants are in the same respiratory phase """
        for index, i in enumerate(resp_phase):
            if index in index_imgs_registered:
                if resp_phase[index] != rphase_sag_img[column]:
                    index_imgs_registered.remove(index)
        # print("(Step 3) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))
        # c = raw_input("?")

        # If there is no registered image
        if len(index_imgs_registered) == 0:
            # return -1, -1, -1

            index_imgs_registered =\
                self.third_step_second_attempt(
                    diaph_lvl=diaph_lvl,
                    dlvl_sag_img=dlvl_sag_img,
                    rphase_sag_img=rphase_sag_img,
                    cor_sequence=cor_sequence,
                    option=True)

            if len(index_imgs_registered) == 0:
                return -1, -1, -1

        # Get first sagittal image that was registered with root image
        if len(index_imgs_registered) > 0:
            imgnum = index_imgs_registered[0]
        # print("(Step 3) Imagem: {}\n".format(imgnum))
        # print("(Step 3) DL[pos]: {}\n".format(diaph_lvl[imgnum]))

        # Fills the matrices
        for i in range(self.matrows):
            pts_pattern = self.pattern_coronal('{}.png'.format(lpatterns[i]))
            diaph_lvl = [max(x) for x in self.diaphragmatic_level_coronal(pts_pattern)]
            resp_phase =\
                self.respiratory_phase_sagittal(self.diaphragmatic_level_sagittal(pts_pattern))

            """ Update matrices """
            if self.matRegistration[i, column] == 0.0 and len(index_imgs_registered) > 0:
                self.matDL[i, column] = diaph_lvl[imgnum]
                self.matRegistration[i, column] = self.yellow
                self.matRP[i, column] = resp_phase[i]

        # print("(Step 3) Registration matrix:\n{}\n".format(self.matRegistration))
        # print("(Step 3) Diaphragmatic level matrix:\n{}\n".format(self.matDL))
        # print("(Step 3) Respiratory phase:\n{}\n".format(self.matRP))

        imgnum = imgnum + 1
        dlvl = [self.matDL[i, column] for i in range(len(self.sag_sequences))]
        rphase = [int(self.matRP[i, column]) for i in range(len(self.sag_sequences))]

        return imgnum, dlvl, rphase

    def third_step_second_attempt(self, diaph_lvl, dlvl_sag_img, rphase_sag_img, cor_sequence, option=False):
        """
        If there are no records, there is the possibility of an error in the segmentation
        (which is done manually), then analyzing the respiratory phase is added in one unit
        the value of the diaphragmatic level (inspiration case) or subtracted one unit
        (expiration case)

        Parameters
        ----------
        diaph_lvl: list
            Diaphragmatic level of the 50 images of the analyzed sagittal sequence that crosses the coronal root image
        dlvl_sag_img:
            Diaphragmatic level of the root instant
        rphase_sag_img:
            Respiratory phase of the root instant
        cor_sequence: int
            Coronal sequence being analyzed
        option: bool
            If True, check images to register up to 3 difference units
            If False, check images to register up to 1 difference units """
        print("Second attempt (Third Step)")

        index_imgs_registered = list()  # Store index of the sagittal registered images
        variation_lvl = 2  # Variation of the diaphragmatic level

        if option:
            for lvl in range(1, variation_lvl):
                # Current coronal (root) diaphragmatic level being analyzed
                diaph_lvl_root = dlvl_sag_img[self.cor_sequences.index(cor_sequence)]
                resp_phase_root = rphase_sag_img[self.cor_sequences.index(cor_sequence)]
                # print("DL: {} ({})".format(diaph_lvl_root, resp_phase_root))
                if resp_phase_root == 1:
                    diaph_lvl_root += lvl
                else:
                    diaph_lvl_root -= lvl
                # print("Diaph. lvl: {}\n".format(diaph_lvl_root))

                for index, i in enumerate(diaph_lvl):
                    if i == diaph_lvl_root:
                        index_imgs_registered.append(index)
                # print("(Step 3 - Second attempt) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))
                # c = raw_input("?")
                if len(index_imgs_registered) > 0:
                    break

                # if still there is no image registered, invert the calc of the diaphragmatic level and try again
                if len(index_imgs_registered) == 0:
                    for lvl in range(1, variation_lvl):
                        # Current coronal (root) diaphragmatic level being analyzed
                        diaph_lvl_root = dlvl_sag_img[self.cor_sequences.index(cor_sequence)]
                        resp_phase_root = rphase_sag_img[self.cor_sequences.index(cor_sequence)]
                        # print("DL: {} ({})".format(diaph_lvl_root, resp_phase_root))
                        if resp_phase_root == 1:
                            diaph_lvl_root -= lvl
                        else:
                            diaph_lvl_root += lvl
                        # print("Diaph. lvl: {}\n".format(diaph_lvl_root))

                        for index, i in enumerate(diaph_lvl):
                            if i == diaph_lvl_root:
                                index_imgs_registered.append(index)
                        # print("(Step 3 - Second attempt) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))
                        # c = raw_input("?")
                        if len(index_imgs_registered) > 0:
                            break

        else:
            # Current sagittal (root) diaphragmatic level being analyzed
            diaph_lvl_root = dlvl_sag_img[self.cor_sequences.index(cor_sequence)]
            resp_phase_root = rphase_sag_img[self.cor_sequences.index(cor_sequence)]
            # print("DL: {} ({})".format(diaph_lvl_root, resp_phase_root))
            if resp_phase_root == 1:
                diaph_lvl_root += 1
            else:
                diaph_lvl_root -= 1
            # print("Diaph. lvl: {}\n".format(diaph_lvl_root))

            for index, i in enumerate(diaph_lvl):
                if i == diaph_lvl_root:
                    index_imgs_registered.append(index)
            # print("(Step 3 - Second attempt) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))

        return index_imgs_registered


def calculate_points(patient, side, root_sequence, root_img, csequences, lssequences, rssequences):
    reg = Register(patient, side, csequences, lssequences, rssequences)
    root_plan = 'Coronal'

    if side == 0:
        ssequences = lssequences  # Left
    else:
        ssequences = rssequences  # Right

    points2plot = list()

    """ -------------------------------- First step -------------------------------- """
    print('(Step 1) Root: Image: {} - Sequence: {}\n'.format(root_img, root_sequence))
    points2plot_st = list()
    dl_st, rp_st =\
        reg.first_step(
            plan=root_plan,
            sequence=root_sequence,
            imgnum=root_img)
    xpts_st, ypts_st, zpts_st, lpts_st =\
        reg.convert_points(
            step=1,
            rootsequence=root_sequence,
            currentsequence=root_sequence,
            imgnum=root_img)
    # print("Points: {} ({})\n".format(lpts_st, len(lpts_st)))
    # points2plot.append(lpts_st)
    points2plot_st.append(lpts_st)

    """ -------------------------------- Second step -------------------------------- """
    ldl_nd, lrp_nd, points2plot_nd = list(), list(), list()
    lindex_reg_img_nd = []  # images' indexes that were registered in the second step

    for i in range(len(ssequences)):
        img_reg_nd, dl_nd, rp_nd =\
            reg.second_step(
                plan='Sagittal',
                dlvl_root_img=dl_st,
                rphase_root_img=rp_st,
                root_sequence=root_sequence,
                sag_sequence=ssequences[i])
        ldl_nd.append(dl_nd)
        lrp_nd.append(rp_nd)

        if img_reg_nd != -1:
            lindex_reg_img_nd.append(img_reg_nd)

            xpts_nd, ypts_nd, zpts_st, lpts_nd =\
                reg.convert_points(
                    step=2,
                    rootsequence=root_sequence,
                    currentsequence=ssequences[i],
                    imgnum=img_reg_nd)

            points2plot_nd.append(lpts_nd)
            # points2plot.append(lpts_nd)
        else:
            lindex_reg_img_nd.append(0)  # Became -1 in the next line instruction

        # print("(Step 2) Registered image: {}\n".format(img_reg_nd))

    lindex_reg_img_nd = list(map(lambda item: item - 1, lindex_reg_img_nd))
    print("(Step 2) Reg. images: {}\n".format(lindex_reg_img_nd))

    '''
    ssequence = 1
    img_reg_nd, dl_nd, rp_nd =\
        reg.second_step(
            plan='Sagittal',
            dlvl_root_img=dl_st,
            rphase_root_img=rp_st,
            root_sequence=root_sequence,
            sag_sequence=ssequence)

    if img_reg_nd != -1:
        xpts_nd, ypts_nd, zpts_st, lpts_nd =\
            reg.convert_points(
                step=2,
                rootsequence=root_sequence,
                currentsequence=ssequence,
                imgnum=img_reg_nd)
    print("(Step 2) Registered image: {} - Sequence: {}\n".format(img_reg_nd, ssequence))
    points2plot.append(lpts_nd)
    '''

    """ ------------------- Calculate the root sagittal sequence ------------------- """
    lreg_img, sagsequences = list(), list()
    for s in range(len(ssequences)):
        dl_nd = ldl_nd[ssequences.index(ssequences[s])]
        # print("*" * 50)
        # print(">>>> Sagittal sequence: {}, DL: {}\n".format(ssequences[s], dl_nd))
        rp_nd = lrp_nd[ssequences.index(ssequences[s])]

        if dl_nd != -1:
            for c in range(len(csequences)):
                # print(">>>> Coronal sequence: {}\n".format(csequences[c]))
                reg_img =\
                    reg.get_root_sagittal_sequence(
                        dlvl_sag_img=dl_nd,
                        rphase_sag_img=rp_nd,
                        cor_sequence=csequences[c],
                        sag_sequence=ssequences[s])

                lreg_img.append(reg_img)
                # print("Registered images: {} ({})\n".format(lreg_img, len(lreg_img)))

            # Join the lists
            flatten = list(itertools.chain.from_iterable(lreg_img))
            lreg_img = []
            # print("(Root Sag.) {}, ({})\n".format(flatten, len(flatten)))
            # c = raw_input("{} ... ".format(len(flatten)))
            sagsequences.append(flatten)
        else:
            sagsequences.append([])
    # print("(Root Sag.) {} ({})".format(sagsequences, len(sagsequences)))

    # Get the index that had more registers
    reg_index = sagsequences.index(max(sagsequences, key=lambda coll: len(coll)))
    root_ssequence = ssequences[reg_index]
    print("Root sagittal sequence: {}\n".format(root_ssequence))

    """ ------------------ Calculate the roots sagittal sequences ------------------ """
    roots_sequences = list()  # List with the root sagittal sequences in order of registers (highest to lowest)
    for i in range(8):
        # print(i + 1)
        # print("{}\n".format(i + 1, sagsequences[0]))
        # print("{}\n".format(i + 1, sagsequences[1]))
        # print("{}\n".format(i + 1, sagsequences[2]))
        # print("{}\n".format(i + 1, sagsequences[3]))
        # print("{}\n".format(i + 1, sagsequences[4]))
        # print("{}\n".format(i + 1, sagsequences[5]))
        # print("{}\n".format(i + 1, sagsequences[6]))
        # print("{}\n".format(i + 1, sagsequences[7]))

        if all(len(sagsequences[j]) == 0 for j in range(len(sagsequences))):
            # roots_sequences.append(-1)
            break
        else:
            reg_index = sagsequences.index(max(sagsequences, key=lambda coll: len(coll)))
            sagsequences[reg_index] = []
            roots_sequences.append(ssequences[reg_index])
        # c = raw_input("?")
    print("Roots sagittal sequences: {}\n".format(roots_sequences))

    """ -------------------------------- Third step -------------------------------- """
    lindex_reg_img_rd = list()  # images' indexes that were registered in the third step
    points2plot_rd = list()
    dl_nd = ldl_nd[ssequences.index(root_ssequence)]
    rp_nd = lrp_nd[ssequences.index(root_ssequence)]

    # if dl_nd != -1:
    for i in range(len(csequences)):
        # print(csequences[i])
        # if csequences[i] != root_sequence:
        img_reg_rd, dl_rd, rp_rd =\
            reg.third_step(
                plan='Coronal',
                dlvl_sag_img=dl_nd,
                rphase_sag_img=rp_nd,
                cor_sequence=csequences[i],
                sag_sequence=root_ssequence)
        if img_reg_rd != -1:
            lindex_reg_img_rd.append(img_reg_rd)

            xpts_rd, yprs_rd, zpts_rd, lpts_rd =\
                reg.convert_points(
                    step=3,
                    rootsequence=root_sequence,
                    currentsequence=csequences[i],
                    imgnum=img_reg_rd)

            points2plot_rd.append(lpts_rd)
            # points2plot.append(lpts_rd)
        else:
            lindex_reg_img_rd.append(0)

    lindex_reg_img_rd = list(map(lambda item: item - 1, lindex_reg_img_rd))
    print("(Step 3) Reg. images: {}\n".format(lindex_reg_img_rd))

    """ ------------------- Create list with all the 3D points ------------------- """
    all3Dpoints, ptsrd = list(), points2plot_rd.copy()
    # Adding points from 1st step
    for i in range(len(points2plot_st)):
        all3Dpoints.append(points2plot_st[i])

    # Adding points from 2nd step
    for i in range(len(points2plot_nd)):
        all3Dpoints.append(points2plot_nd[i])

    # Adding points from 3rd step
    for i in range(len(ptsrd)):
        if i != csequences.index(rootsequence):  # Adds the points of the third step of the register (except the points of the coronal root image)
            all3Dpoints.append(ptsrd[i])
    ptsrd.pop(csequences.index(rootsequence))  # Delete the points of coronal root image

    all3Dpoints = list(itertools.chain.from_iterable(all3Dpoints))

    """ ------------------- Adding points in a list to plot them -------------------
    Do what Tsuzuki, M. S. G. et al. "Animated solid model of the lung constructed
    from unsynchronized MR sequential images". In: Computer-Aided Design. Vol. 41,
    pp. 573 - 585, 2009. does in figure 10.
    """

    yminmax = list()
    for i in range(len(points2plot_rd)):
        yminmax.append(points2plot_rd[i][0][1])
    ymin, ymax = min(yminmax), max(yminmax)  # ymin, yman: y value relative to first and last coronal sequence used

    delitems = list()  # Points to be deleted
    for i in range(len(points2plot_nd)):
        # print("Points[{}]: {} ({})\n".format(i, points2plot_nd[i], len(points2plot_nd[i])))
        for j in range(len(points2plot_nd[i])):
            if points2plot_nd[i][j][1] > ymin and points2plot_nd[i][j][1] < ymax:
                delitems.append(points2plot_nd[i][j])

        # print("Deleted: {} ({})\n".format(delitems, len(delitems)))
        points = points2plot_nd[i][:]  # Need to make a copy of the list not to change the indexes of the items in the deletion
        for element in points2plot_nd[i]:  # Scans the list by checking for the same items
            if element in delitems:  # If there is, delete item from the copied list...
                points.remove(element)
        points2plot_nd[i] = points  # Update original list
        # print("Points[{}]: {} ({})\n".format(i, points2plot_nd[i], len(points2plot_nd[i])))

    # Adding points from 1st step
    for i in range(len(points2plot_st)):
        points2plot.append(points2plot_st[i])

    # Adding points from 2nd step
    for i in range(len(points2plot_nd)):
        points2plot.append(points2plot_nd[i])

    # Adding points from 3rd step
    for i in range(len(points2plot_rd)):
        if i != csequences.index(rootsequence):  # Adds the points of the third step of the record (except the points of the coronal root image)
            points2plot.append(points2plot_rd[i])
    points2plot_rd.pop(csequences.index(rootsequence))  # Delete the points of coronal root image
    # points2plot.append(points2plot_rd[0])
    # points2plot.append(points2plot_rd[-1])

    points2plot = list(itertools.chain.from_iterable(points2plot))

    return lpts_st, points2plot_nd, points2plot_rd, points2plot, all3Dpoints


def both_lungs(patient, root_sequence, root_img, csequences, lssequences, rssequences):
    lpts_step1, lpts_step2, lpts_step3, lpts, lall3Dpoints =\
        calculate_points(
            patient=patient,
            side=0,
            root_sequence=root_sequence,
            root_img=root_img,
            csequences=csequences,
            lssequences=lssequences,
            rssequences=rssequences)

    rpts_step1, rpts_step2, rpts_step3, rpts, rall3Dpoints =\
        calculate_points(
            patient=patient,
            side=1,
            root_sequence=root_sequence,
            root_img=root_img,
            csequences=csequences,
            lssequences=lssequences,
            rssequences=rssequences)

    return lpts_step1, lpts_step2, lpts_step3, lpts, lall3Dpoints, rpts_step1, rpts_step2, rpts_step3, rpts, rall3Dpoints


def execute(optplot, patient, side, root_sequence, root_img, csequences, lssequences, rssequences):
    pc = Reconstruction(patient)

    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.set_xlabel('X axis')
    axes.set_ylabel('Y axis')
    axes.set_zlabel('Z axis')

    """
    optplot = 0 - Plot the point cloud using matplotlib library
    optplot = 1 - Plot wireframe using matplotlib
    optplot = 2 - Plot the point cloud using plotly library
    optplot = 3 - Plot the surface of the lung using alpha-shape by plotly library
    optplot = 4 - Plot the point cloud and surface using plotly library
    """

    if side == 0 or side == 1:
        pts_step1, pts_step2, pts_step3, pts, all3Dpoints =\
            calculate_points(
                patient=patient,
                side=side,
                root_sequence=root_sequence,
                root_img=root_img,
                csequences=csequences,
                lssequences=lssequences,
                rssequences=rssequences)

        if optplot == 0:
            pc.plot3D(pts=pts, fig=figure, ax=axes, howplot='dots', dots=1)
        elif optplot == 1:
            pc.plot3D(pts=pts, howplot='wireframe')
        elif optplot == 2:
            if side == 0:
                pc.pointCloud(left_pts=all3Dpoints)
            else:
                pc.pointCloud(right_pts=all3Dpoints)
        elif optplot == 3:
            if side == 0:
                pc.alphaShape(left_pts=all3Dpoints, alpha=5, opacity=1.0)
            else:
                pc.alphaShape(right_pts=all3Dpoints, alpha=5, opacity=1.0)
        elif optplot == 4:
            if side == 0:
                pc.alphaShape_pointCloud(left_pts=all3Dpoints, alpha=5, opacity=1.0)
            else:
                pc.alphaShape_pointCloud(right_pts=all3Dpoints, alpha=5, opacity=1.0)
        elif optplot == -1:
            pc.plotColor3D(
                pts_step1=pts_step1,
                pts_step2=list(itertools.chain.from_iterable(pts_step2)),
                pts_step3=list(itertools.chain.from_iterable(pts_step3)))
        else:
            print("Invalid plot option!")
    elif side == 2:
        lpts_step1, lpts_step2, lpts_step3, lpts, lall3Dpoints, rpts_step1, rpts_step2, rpts_step3, rpts, rall3Dpoints =\
            both_lungs(
                patient=patient,
                root_sequence=root_sequence,
                root_img=root_img,
                csequences=csequences,
                lssequences=lssequences,
                rssequences=rssequences)

        if optplot == 0:
            pc.plotLungs3D(
                all_points=lpts + rpts,
                left_pts=lpts,
                right_pts=rpts,
                fig=figure,
                axes=axes,
                howplot='dots',
                dots=1)

        elif optplot == 2:
            pc.pointCloud(left_pts=lpts, right_pts=rpts)

        elif optplot == 3:
            pc.alphaShape(left_pts=lpts, right_pts=rpts, alpha=6, opacity=1.0)

        elif optplot == 4:
            pc.alphaShape_pointCloud(left_pts=lpts, right_pts=rpts, alpha=6, opacity=1.0)

        else:
            pc.plotLungs3D(
                all_points=lpts + rpts,
                left_pts=lpts,
                right_pts=rpts,
                fig=figure,
                axes=axes,
                howplot='wireframe')
    else:
        print("Invalid option!")


if __name__ == '__main__':
    try:
        mode = 0  # 0 - Points cloud | 1 - Wireframe
        patient = 'Matsushita'  # 'Iwasawa'
        rootsequence = 14  # 9
        side = 1  # 0 - left | 1 - right | 2 - Both
        imgnumber = 1  # Instant
        # Iwasawa
        # coronalsequences = '4, 5, 6, 7, 8, 9, 10, 11, 12'
        # leftsagittalsequences = '1, 2, 3, 4, 5, 6, 7, 8'
        # rightsagittalsequences = '12, 13, 14, 15, 16, 17'
        # Matsushita
        coronalsequences = '4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21'  # 3
        leftsagittalsequences = '2, 3, 4, 5, 6, 7'  # 8, 9
        rightsagittalsequences = '10, 11, 12, 13, 14, 15'  # 16, 17
        leftalpha = 6
        rightalpha = 6
        opacity = 1.0

        txtargv2 = '-mode={}'.format(mode)
        txtargv3 = '-patient={}'.format(patient)
        txtargv4 = '-rootsequence={}'.format(rootsequence)
        txtargv5 = '-side={}'.format(side)
        txtargv6 = '-imgnumber={}'.format(imgnumber)
        txtargv7 = '-coronalsequences={}'.format(coronalsequences)
        txtargv8 = '-leftsagittalsequences={}'.format(leftsagittalsequences)
        txtargv9 = '-rightsagittalsequences={}'.format(rightsagittalsequences)

        if len(sys.argv) > 1:
            txtargv2 = sys.argv[1]
            if len(sys.argv) > 2:
                txtargv3 = sys.argv[2]
                if len(sys.argv) > 3:
                    txtargv4 = sys.argv[3]
                    if len(sys.argv) > 4:
                        txtargv5 = sys.argv[4]
                        if len(sys.argv) > 5:
                            txtargv6 = sys.argv[5]
                            if len(sys.argv) > 6:
                                txtargv7 = sys.argv[6]
                                if len(sys.argv) > 7:
                                    txtargv8 = sys.argv[7]
                                    if len(sys.argv) > 8:
                                        txtargv9 = sys.argv[8]

        txtargv =\
            '{}|{}|{}|{}|{}|{}|{}|{}'.format(
                txtargv2,
                txtargv3,
                txtargv4,
                txtargv5,
                txtargv6,
                txtargv7,
                txtargv8,
                txtargv9,)

        if txtargv.find('-mode') != -1:
            txttmp = txtargv.split('-mode')[1]
            txttmp = txttmp.split('=')[1]
            mode = int(txttmp.split('|')[0])

        if txtargv.find('-patient') != -1:
            txttmp = txtargv.split('-patient')[1]
            txttmp = txttmp.split('=')[1]
            patient = txttmp.split('|')[0]

        if txtargv.find('-rootsequence') != -1:
            txttmp = txtargv.split('-rootsequence')[1]
            txttmp = txttmp.split('=')[1]
            rootsequence = int(txttmp.split('|')[0])

        if txtargv.find('-side') != -1:
            txttmp = txtargv.split('-side')[1]
            txttmp = txttmp.split('=')[1]
            side = int(txttmp.split('|')[0])

        if txtargv.find('-imgnumber') != -1:
            txttmp = txtargv.split('-imgnumber')[1]
            txttmp = txttmp.split('=')[1]
            imgnumber = int(txttmp.split('|')[0])

        if txtargv.find('-coronalsequences') != -1:
            txttmp = txtargv.split('-coronalsequences')[1]
            txttmp = txttmp.split('=')[1]
            txttmp = txttmp.split('|')[0]
            txttmp = txttmp.split(',')
            if len(coronalsequences) > 0:
                coronalsequences = [int(x) for x in txttmp]
            else:
                coronalsequences = []

        if txtargv.find('-leftsagittalsequences') != -1:
            txttmp = txtargv.split('-leftsagittalsequences')[1]
            txttmp = txttmp.split('=')[1]
            txttmp = txttmp.split('|')[0]
            txttmp = txttmp.split(',')
            if len(leftsagittalsequences) > 0:
                leftsagittalsequences = [int(x) for x in txttmp]
            else:
                leftsagittalsequences = []

        if txtargv.find('-rightsagittalsequences') != -1:
            txttmp = txtargv.split('-rightsagittalsequences')[1]
            txttmp = txttmp.split('=')[1]
            txttmp = txttmp.split('|')[0]
            txttmp = txttmp.split(',')
            if len(rightsagittalsequences) > 0:
                rightsagittalsequences = [int(x) for x in txttmp]
            else:
                rightsagittalsequences = []

    except ValueError:
        print(
            """
            Example of use:\n

            $ python {} -mode=0 -patient=Iwasawa -rootsequence=9 -side=0 -imgnumber=1
            -coronalsequences=4,5,6,7,8,9,10,11,12
            -leftsagittalsequences=1,2,3,4,5,6,7,8
            -rightsagittalsequences=12,13,14,15,16,17
            """.format(sys.argv[0]))
        exit()

    execute(
        optplot=mode,
        patient=patient,
        side=side,
        root_sequence=rootsequence,
        root_img=imgnumber,
        csequences=coronalsequences,
        lssequences=leftsagittalsequences,
        rssequences=rightsagittalsequences)
