#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
from operator import itemgetter

from util.constant import *
from plot_diaphragm import PointCloud


class Register(object):

    """
    The multiple register is performed for all coronal-sagittal intersections.
    Initially, a coronal image is defined as root.

    First of all, the temporal registration is performed with all crossing
    sagittal sequences and then, all the temporal sagittal sequences (except
    the root) are all registered considering the temporal registration
    determined in the previous step.

    There are three cases:
    1) Positions in which the temporary register could not be found;
    2) Positions in which the temporary register is unique;
    3) Positions where there are multiple temporal registers.

    References
    ----------
    Abe, L. I.; Tsuzuki, M. S. G.; Chirinos, J. M. M.; Martins, T. C.;
    Gotoh, T.; Kagei, S.; Iwasawa, T.; Silva, A. G.; Rosso Jr., R. S. U.
    "Diaphragmatic Surface Reconstrution from Massive Temporal Registration
    of Orthogonal MRI Sequences". In: The International Federation of
    Automatic Control. Vol. 47, No. 3, pp. 3.569 - 3.574, 2014.
    """

    def __init__(self, patient, lung, corsequences, lsagsequences, rsagsequences):
        self.patient = patient
        self.lung = lung
        self.corsequences = corsequences
        self.lsagsequences = lsagsequences
        self.rsagsequences = rsagsequences
        self.pc = PointCloud(patient)

        self.matDL = None  # Matrix with diaphragmatic level info
        self.matRP = None  # Matrix with respiratory phase info

        # Matrix that represents (x, y, z) positions on the 3D space
        self.matX = None  # Matrix representing X value of each point after converting to 3D
        self.matY = None  # Matrix representing Y value of each point after converting to 3D
        self.matZ = None  # Matrix representing Z value of each point after converting to 3D

        """
        matRegistration: Matrix that contains registration information of each crossing position
        The vertical and horizontal lines represent the coronal and sagittal images respectively

        Colors meaning:
        1) Blue: Positions where the diaphragmatic level and the respiratory phase were determined
           for the coronal root image (Abe (2013) page 44)
        2) Green: Positions where the diaphragmatic level and the respiratory phase were determined
           for each moment where the temporal register occurred (Abe (2013) page 46)
        3) Red: Positions where a temporal record was found in the third step (Abe (2013) page 46)
        4) Yellow: Similar to green dots (Abe (2013) page 46)
        5) Purple: Diaphragmatic level is indirectly defined as the mean between the levels present
           in the images that intersect at this position (Abe (2013) page 48)
        6) Gray: Positions that do not have a diaphragmatic level. These points shall be determined
           by linear interpolation between adjacent positions (Abe (2013) page 48)
        """
        self.matRegistration = None  # Values of the x-axis of the intersection between plans
        self.blue = 1
        self.green = 2
        self.red = 3
        self.yellow = 4
        self.purple = 5
        self.gray = 6

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

        '''
        # Create a list of which sequences are available in the sagittal and coronal planes
        self.cor_sequences = list(set([int(i.split('-')[1]) for i in self.dataset]))
        self.sag_sequences = list(set([int(i.split('-')[3]) for i in self.dataset]))
        print("Coronal sequences: {} ({})".format(self.cor_sequences, len(self.cor_sequences)))
        print("Sagittal sequences: {} ({}) \n ".format(self.sag_sequences, len(self.sag_sequences)))

        # Create list with the positions of intersections between the coronal and sagittal plans
        self.cor_columns =\
            sorted(list(set([int(i.split(';')[0].split('-')[2]) for i in self.dataset])))
        self.sag_columns =\
            sorted(list(set([int(i.split(';')[1].split('-')[2]) for i in self.dataset])))
        print("Coronal columns: {} ({})".format(self.cor_columns, len(self.cor_columns)))
        print("Sagittal columns: {} ({})\n".format(self.sag_columns, len(self.sag_columns)))

        # Matrices' dimensions
        self.matrows = len(self.cor_columns)
        self.matcols = len(self.sag_columns)
        # print("Matrix's rows number: {}\n".format(self.matrows))
        # print("Matrix's columns number: {}".format(self.matcols))
        '''

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

    def convert_points(self, step, root_sequence, sequence, imgnum):
        """
        Convert 2D points to 3D (Create a list with points of the diaphragmatic surface)
        Obs. This function must be call only after populate the matrixes (matDL and
        matRegistration)

        Parameters
        ----------
        step: int
            Represents the step of the multiple registration process
            Step 1: Establish respiratory instant (defines sequence and coronal root image);
            Step 2: Check register of sagittal images that intersect with the coronal root image;
            Step 3: Check the register of the coronal images (belonging to the parallel sequences
                    containing the root image) that cross the sagittal image recorded in the
                    second step.

        root_sequence: int
            Represents which sequence belong the root image

        sequence: int
            Current sequence being parsed

        imgnum: int
            Image's number registered

        Return
        ------
        lpts: list
            List of tuples with points that represents the diaphragmatic surface
        """
        # print("Root coronal sequence: {}\n".format(root_sequence))
        # print("Root index: {}\n".format(self.cor_sequences.index(root_sequence)))
        root_index = self.cor_sequences.index(root_sequence)

        # Convert bidimensional numpy array to list of lists
        lmatDL = self.matDL.tolist()

        lpts = []  # list of points that represents the diaphragmatic surface

        # Extract diaphragmatic surface points
        if step == 1:
            """ Walks over the matrix of diaphragmatic levels (row, column)
                and if the item is not 0, add to the list"""
            for i in range(len(lmatDL)):  # Rows
                for j in range(len(lmatDL[0])):  # Columns
                    if lmatDL[i][j] != 0.0:
                        pt = (self.cor_columns[i], int(lmatDL[i][j]))
                        lpts.append(pt)
            # print('(Step 1) {} ({})\n'.format(lpts, len(lpts)))
        elif step == 2:
            idx = self.sag_sequences.index(sequence)  # Index of the coronal sequence containing the root image
            """ Walks over the matrix of diaphragmatic levels (column)
                and if the item is not 0, add to the list"""
            for i in range(len(lmatDL[0])):  # Columns
                pt = (self.sag_columns[i], int(lmatDL[idx][i]))
                lpts.append(pt)
            # print('(Step 2) {} ({})\n'.format(lpts, len(lpts)))
        else:
            # Index of the sagittal sequence containing the image registered in the second step
            idx = self.cor_sequences.index(sequence)
            """ Walks over the matrix of diaphragmatic levels (row)
                and if the item is not 0, add to the list"""
            for i in range(len(lmatDL)):  # Rows
                pt = (self.cor_columns[i], int(lmatDL[i][idx]))
                lpts.append(pt)
            # print('(Step 3) {} ({})\n'.format(lpts, len(lpts)))

        """ Converts 2D points to 3D space and stores the values of (x, y, z) in the variables X, Y, Z
            respectively and fills the matrices """
        if step == 1:
            X, Y, Z = self.pc.point3D(plan='Coronal', sequence=sequence, imgnum=imgnum, pts=lpts)

            # Fills the matrices with the values of the coordinates
            index = self.cor_sequences.index(sequence)  # Index of coronal sequence analyzed
            for i in range(len(self.sag_sequences)):
                self.matX[i, index] = X[i]
                self.matY[i, index] = Y[i]
                self.matZ[i, index] = Z[i]
        elif step == 2:
            X, Y, Z = self.pc.point3D(plan='Sagittal', sequence=sequence, imgnum=imgnum, pts=lpts)

            # Fills the matrices with the values of the coordinates
            index = self.sag_sequences.index(sequence)  # index of sagittal sequence analyzed
            for i in range(len(self.cor_sequences)):  # - 1
                if i != root_index:
                    self.matX[index, i] = X[i]
                    self.matY[index, i] = Y[i]
                    self.matZ[index, i] = Z[i]

            # Remove item referring to the position of the coronal root sequence so as not to overlap two points
            del X[root_index]
            del Y[root_index]
            del Z[root_index]
        else:
            X, Y, Z = self.pc.point3D(plan='Coronal', sequence=sequence, imgnum=imgnum, pts=lpts)

            # Fills the matrices with the values of the coordinates
            index = self.cor_sequences.index(sequence)  # Index of coronal sequence analyzed
            for i in range(len(self.sag_sequences)):
                self.matX[i, index] = X[i]
                self.matY[i, index] = Y[i]
                self.matZ[i, index] = Z[i]

        # print('X: {}\n'.format(X))
        # print('Y: {}\n'.format(Y))
        # print('Z: {}\n'.format(Z))

        # print('{}\n'.format(self.matX))
        # print('{}\n'.format(self.matY))
        # print('{}\n'.format(self.matZ))

        # Create a list of the points
        lpts = []
        for i in range(len(X)):
            pt = (X[i], Y[i], Z[i])
            lpts.append(pt)
        # print('Points: {} ({})\n'.format(lpts, len(lpts)))

        return X, Y, Z, lpts

    def first_step(self, plan, sequence, imgnum):
        """
        First step in the process to register images between coronal and sagittal sequences.
        Root image (coronal) is defined and the register matrix is built (now only with data
        from the root image)

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
        # print('Index: {}\n'.format(column))

        """ Retrieves the diaphragmatic levels of the root image (coronal plan) and stores
            them in the correct positions of the register matrix """
        for i in range(self.matrows):
            # print('{}.png'.format(lpatterns[i]))

            pts_pattern = self.pattern_coronal('{}.png'.format(lpatterns[i]))
            diaph_lvl = self.diaphragmatic_level_coronal(pts_pattern)
            max_diaph_lvl = max_diaphragmatic_level(diaph_lvl)
            resp_phase = self.respiratory_phase_coronal(diaph_lvl)

            self.matDL[i, column] = max_diaph_lvl[imgnum]
            self.matRP[i, column] = resp_phase[i]
            self.matRegistration[i, column] = self.blue

        # print("(Step 1) Diaphragmatic level matrix:\n{}\n".format(self.matDL))
        # print("(Step 1) Registration matrix:\n{}\n".format(self.matRegistration))
        # print("(Step 1) Respiratory phase:\n{}\n".format(self.matRP))

        dlvl = [self.matDL[i, column] for i in range(len(self.sag_sequences))]
        rphase = [int(self.matRP[i, column]) for i in range(len(self.sag_sequences))]
        # print("(Step 1) DL: {}\n".format(dlvl))

        return dlvl, rphase

    def second_step(self, plan, dlvl_root_img, rphase_root_img, root_sequence, sag_sequence):
        """
        Second step in the process to register images between coronal and sagittal sequences.

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
        # Sagittal respiratory patterns associated with sagittal sequence analyzed
        lpatterns =\
            [x for x in self.dataset if int(x.split('-')[3]) == self.sag_sequences[self.sag_sequences.index(sag_sequence)]]
        # print('(Step 2) Sagittal patterns: {} ({})\n'.format(lpatterns, len(lpatterns)))

        # Respiratory patterns linked to the sequence containing the root image
        pattern = None
        for p in lpatterns:
            if int(p.split('-')[1]) == root_sequence:
                pattern = p
        # print("(Step 2) Pattern: {}".format(pattern))
        # print("(Step 2) DL: {}".format(dlvl_root_img))
        # print("(Step 2) DL[{}]: {}".format(self.sag_sequences.index(sag_sequence), dlvl_root_img[self.sag_sequences.index(sag_sequence)]))
        # print("(Step 2) RP: {}".format(rphase_root_img))
        # print("(Step 2) RP[{}]: {}\n".format(self.sag_sequences.index(sag_sequence), rphase_root_img[self.sag_sequences.index(sag_sequence)]))

        """ Get the diaphragmatic level of each image of the analyzed sagittal sequence
            that crosses the coronal root image """
        pts_pattern = self.pattern_sagittal('{}.png'.format(pattern))
        diaph_lvl = [max(x) for x in self.diaphragmatic_level_sagittal(pts_pattern)]
        resp_phase = self.respiratory_phase_sagittal(self.diaphragmatic_level_sagittal(pts_pattern))
        # print("(Step 2) Diaphragmatic level: {} ({})\n".format(diaph_lvl, len(diaph_lvl)))
        # print("(Step 2) Respiratory phase: {} ({})\n".format(resp_phase, len(resp_phase)))

        """ Check register condition:
            1) If there is same diaphragmatic level """
        index_imgs_registered = list()  # Store index of the sagittal registered images
        for index, i in enumerate(diaph_lvl):
            if i == dlvl_root_img[self.sag_sequences.index(sag_sequence)]:
                index_imgs_registered.append(index)
        # print("(Step 2) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))

        # TODO - Uncomment this for release
        '''
        """ Check register condition:
            2) If the instants are in the same respiratory phase """
        for index, i in enumerate(resp_phase):
            if index in index_imgs_registered:
                if resp_phase[index] != rphase_root_img[self.sag_sequences.index(sag_sequence)]:
                    index_imgs_registered.remove(index)
        # print("(Step 2) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))
        '''

        # If there is no registered image
        if len(index_imgs_registered) == 0:

            """ If there are no records, there is the possibility of an error in the segmentation
                (which is done manually), then analyzing the respiratory phase is added in one unit
                the value of the diaphragmatic level (inspiration case) or subtracted one unit
                (expiration case) """
            # Current coronal (root) diaphragmatic level being analyzed
            '''
            diaph_lvl_root = dlvl_root_img[self.sag_sequences.index(sag_sequence)]
            resp_phase_root = rphase_root_img[self.sag_sequences.index(sag_sequence)]
            print("Diaph. lvl: {}\n".format(diaph_lvl_root))
            print("Resp. phase: {}\n".format(resp_phase_root))
            if resp_phase_root == 1:
                diaph_lvl_root += 1
            else:
                diaph_lvl_root -= 1
            print("Diaph. lvl: {}\n".format(diaph_lvl_root))

            index_imgs_registered = list()  # Store index of the sagittal registered images
            for index, i in enumerate(diaph_lvl):
                if i == diaph_lvl_root:
                    index_imgs_registered.append(index)
            print("(Step 2 - Second attempt) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))

            if len(index_imgs_registered) == 0:
                return -1, -1, -1
            '''
            return -1, -1, -1

        # Get first sagittal image that was registered with root image
        imgnum = index_imgs_registered[0]
        # print("(Step 2) Imagem: {}\n".format(imgnum))

        """ Represents the row in the register matrix that represents the sequence used
            It's used to populate the registration matrix correctly """
        row = self.sag_sequences.index(sag_sequence)
        # print('(Step 2) Row: {}\n'.format(row))

        for i in range(self.matcols):
            pts_pattern = self.pattern_sagittal('{}.png'.format(lpatterns[i]))
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
        # print("(Step 2) DL: {}\n".format(dlvl))

        return (imgnum, dlvl, rphase)

    def third_step(self, plan, dlvl_sag_img, rphase_sag_img, root_cor_sequence, cor_sequence, sag_sequence):
        """
        For each result of the second step of the algorithm a serach is made again finding
        temporal registers in coronal images parallel to those of the root sequence

        Parameters
        ----------
        plan: string
            Must be sagittal. Represents plan orthogonal to the root image

        dlvl_sag_img: list
            List of integers representing the diaphragmatic level at crossing points

        rphase_sag_img: list
            List of integers representing the repiratpry phase at crossing points

        root_cor_sequence: int
            Represents which sequence belong the root image

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
        """ Represents the column in the register matrix that represents the coronal sequence used.
            It's used to populate the registration matrix correctly """
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
        # print("(Step 3) Diaphragmatic level: {} ({})\n".format(diaph_lvl, len(diaph_lvl)))
        # print("(Step 3) Respiratory phase: {} ({})\n".format(resp_phase, len(resp_phase)))

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

        # If there is no registered image
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
            # resp_phase =\
            #     self.respiratory_phase_sagittal(self.diaphragmatic_level_sagittal(pts_pattern))

            """ By analyzing a green points:
                - Red points: Appears if a temporal register is found """
            if self.matRegistration[i, column] == 0.0 and len(index_imgs_registered) > 0:
                self.matDL[i, column] = diaph_lvl[imgnum]
                self.matRegistration[i, column] = self.yellow
                self.matRP[i, column] = resp_phase[i]

            # elif self.matRegistration[i, column] == 2.0 and len(index_imgs_registered) > 0:
            #     self.matRegistration[i, column] = self.red

            # print("(Step 3) Diaphragmatic level matrix:\n{}\n".format(self.matDL))
            # print("(Step 3) Registration matrix:\n{}\n".format(self.matRegistration))
            # print("(Step 3 Respiratory phase:\n{}\n".format(self.matRP))

        # print("(Step 3) Diaphragmatic level matrix:\n{}\n".format(self.matDL))
        # print("(Step 3) Registration matrix:\n{}\n".format(self.matRegistration))
        # print("(Step 3) Respiratory phase:\n{}\n".format(self.matRP))

        imgnum = imgnum + 1
        dlvl = [self.matDL[i, column] for i in range(len(self.sag_sequences))]
        rphase = [int(self.matRP[i, column]) for i in range(len(self.sag_sequences))]
        # print("(Step 3) Registered mage: {}\n".format(imgnum))
        # print("(Step 3) DL: {}\n".format(dlvl))

        return imgnum, dlvl, rphase

    def red_points(self, plan, dlvl_sag_img, rphase_sag_img, root_cor_sequence, cor_sequence, sag_sequence):
        """ Check condition for red point
            Analyzing a green point, a red point appears if there is a temporal register with the sagittal and
            coronal image (Abe (2013) page 46) """
        # print("dlvl_sag_img: {}".format(dlvl_sag_img))
        # print("sag_sequence: {}".format(sag_sequence))
        # print("cor_sequence: {}".format(cor_sequence))
        column = self.cor_sequences.index(cor_sequence)
        row = self.sag_sequences.index(sag_sequence)

        lpatterns = [x for x in self.dataset if int(x.split('-')[1]) == cor_sequence]

        pattern = [p for p in lpatterns if int(p.split('-')[3]) == sag_sequence][0]

        pts_pattern = self.pattern_coronal('{}.png'.format(pattern))
        diaph_lvl = [max(x) for x in self.diaphragmatic_level_coronal(pts_pattern)]
        resp_phase = self.respiratory_phase_coronal(self.diaphragmatic_level_coronal(pts_pattern))

        index_imgs_registered = list()  # Store index of the coronal registered images
        for index, i in enumerate(diaph_lvl):
            if i == dlvl_sag_img[self.cor_sequences.index(cor_sequence)]:
                index_imgs_registered.append(index)
        # print("(Red) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))

        for index, i in enumerate(resp_phase):
            if index in index_imgs_registered:
                if resp_phase[index] != rphase_sag_img[self.cor_sequences.index(cor_sequence)]:
                    index_imgs_registered.remove(index)
        # print("(Red) Index of registered images: {} ({})\n".format(index_imgs_registered, len(index_imgs_registered)))

        # print("lpatterns[sag_sequence]: {}, sag_sequence: {}".format(lpatterns[self.sag_sequences.index(sag_sequence)], sag_sequence))
        pts_pattern = self.pattern_coronal('{}.png'.format(lpatterns[self.sag_sequences.index(sag_sequence)]))
        diaph_lvl = [max(x) for x in self.diaphragmatic_level_coronal(pts_pattern)]

        """ If analyzed point is green and there are at least one image registered in this
            position, the diaphragmatic level and respiratory phase is equal and only
            registration matrix is update """
        if self.matRegistration[row, column] == 2.0 and len(index_imgs_registered) > 0:
            # print("{}x{}\n".format(row, column))
            # print("matDL: {}, DL: {}\n".format(self.matDL[i, column], diaph_lvl[imgnum]))
            self.matRegistration[row, column] = self.red

        # print("(Red) Diaphragmatic level matrix:\n{}\n".format(self.matDL))
        # print("(Red) Registration matrix:\n{}\n".format(self.matRegistration))
        # print("(Red) Respiratory phase:\n{}\n".format(self.matRP))

    def yellow_points(self, plan, dlvl_sag_img, rphase_sag_img, root_cor_sequence, cor_sequence, sag_sequence):
        """ Yellow points
            Similar to green points (Abe (2013) page 46)
            Calculate the root sagittal sequence (the sagittal sequence that had more registers)

        Parameters
        ----------
        Same as third_step() function

        Return
        ------
        index_imgs_registered: list
            List of integer with all the image's indexes registered in the second step of temporal register
        """
        # print(">>>> Sag. Seq: {}".format(sag_sequence))
        # print(">>>> Cor. Seq: {}".format(cor_sequence))

        lpatterns = [x for x in self.dataset if int(x.split('-')[1]) == cor_sequence]

        # column = self.cor_sequences.index(cor_sequence)

        pattern = [p for p in lpatterns if int(p.split('-')[3]) == sag_sequence][0]

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

    def purple_points(self, sag_reg_img, cor_reg_img, cor_sequence, sag_sequence):
        """ Purple points:
            Positions where the diaphragmatic level is defined indirectly, as the mean of the levels present in
            the two images that intersect in this position.
            Update Z matrix with the mean of the diaphragmatic levels of the coronal and sagittal images

            Parameters
            ----------
            sag_reg_img: list
                List of the sagittal images registered with the root coronal image

            cor_reg_img: list
                List of the coronal images registered with the root sagittal image (Define previously)

            sag_sequence: int
                Sagittal sequence analyzed

            cor_sequence: int
                Coronal sequence analyzed
        """
        column = self.cor_sequences.index(cor_sequence)
        row = self.sag_sequences.index(sag_sequence)
        # print("{} x {}\n".format(row, column))
        # self.matRegistration[7, 1] = 2.0  # Just for tests

        """ If analyzed point is green and there is no image registered in this position, the diaphragmatic
            level is the levels' average of the two images that intersect in the analyzed position """
        if self.matRegistration[row, column] == 2.0:
            # print("Sag. sequence: {}".format(self.sag_sequences[row]))
            # print("Cor. sequence: {}".format(self.cor_sequences[column]))
            # print(">>>> Sag. Seq: {}".format(sag_sequence))
            # print(">>>> Cor. Seq: {}".format(cor_sequence))
            # print(">>>> Reg. Sag. Img.: {}".format(sag_reg_img))
            # print(">>>> Reg. Cor. Img.: {}\n".format(cor_reg_img))

            # Get the sagittal and coronal images' indexes in the position analyzed
            sag_index_img = sag_reg_img[self.cor_sequences.index(cor_sequence)]
            cor_index_img = cor_reg_img[self.sag_sequences.index(sag_sequence)]
            print("(Purple) Reg. sag. img: {}".format(sag_index_img))
            print("(Purple) Reg. cor. img: {}\n".format(cor_index_img))

            # Get the respiratory pattern (sagittal and coronal) associated in the position analyzed
            pattern =\
                [p for p in self.dataset
                    if int(p.split('-')[1]) == self.cor_sequences[column] and
                    int(p.split('-')[3]) == self.sag_sequences[row]][0]
            # print(pattern)

            # Get the diphragmatics levels from the coronal respiratory pattern
            cor_pts_pattern = self.pattern_coronal('{}.png'.format(pattern))
            cor_diaph_lvl = [max(x) for x in self.diaphragmatic_level_coronal(cor_pts_pattern)]
            # Get the diphragmatics levels from the sagittal respiratory pattern
            sag_pts_pattern = self.pattern_sagittal('{}.png'.format(pattern))
            sag_diaph_lvl = [max(x) for x in self.diaphragmatic_level_sagittal(sag_pts_pattern)]
            # print("DL -> Cor: {} and Sag: {}\n".format(cor_diaph_lvl[sag_index_img], sag_diaph_lvl[cor_index_img]))

            # Calculate the mean of the diaphragmatics levels
            meanDL = round((cor_diaph_lvl[sag_index_img] + sag_diaph_lvl[cor_index_img]) / 2, 2)
            # print('Mean Diaphragm level: {}\n'.format(meanDL))

            # Update regiatration matrix and diaphragmatic level matrix
            # print("(Purple) Diaphragmatic level matrix:\n{}\n".format(self.matDL))
            # print("(Purple) Registration matrix:\n{}\n".format(self.matRegistration))
            self.matDL[row, column] = meanDL
            self.matRegistration[row, column] = self.purple
            # print("(Purple) Diaphragmatic level matrix:\n{}\n".format(self.matDL))
            # print("(Purple) Registration matrix:\n{}\n".format(self.matRegistration))

            # Build the points to convert
            # print("Cor. Col: {}".format(self.cor_columns))
            # print("Sag. Col: {}".format(self.sag_columns))
            # Use coronal image
            # print("{} -> ({}, {})".format(cor_reg_img[row], self.cor_columns[row], meanDL))
            lpts = [(self.cor_columns[row], meanDL)]
            # Convert point to 3D
            X, Y, Z = self.pc.point3D(plan='Sagittal', sequence=sag_sequence, imgnum=sag_reg_img[column], pts=lpts)
            # Use sagittal image
            # print("{} -> ({}, {})".format(sag_reg_img[column], self.sag_columns[column], meanDL))
            # lpts = [(self.sag_columns[column], meanDL)]
            # # Convert point to 3D
            # X, Y, Z = self.pc.point3D(plan='Sagittal', sequence=sag_sequence, imgnum=sag_reg_img[column], pts=lpts)

            # Update Z matrix
            # print("(Purple) Z: \n{}\n".format(self.matZ))
            # print(self.matZ[row, column])
            self.matZ[row, column] = Z[0]
            # print("(Purple) Z: \n{}\n".format(self.matZ))

            # c = raw_input("?")

    def gray_points(self, sag_sequence, cor_sequence):
        """ Positions that do not have a diaphragmatic level. These points shall be determined by linear
            interpolation between adjacent positions (Abe (2013) page 48) """

        # python register.py -side=1 -imgnumber=9 # right lung (user view)

        def diff_from_zero(value):
            return value != 0

        column = self.cor_sequences.index(cor_sequence)
        row = self.sag_sequences.index(sag_sequence)

        neighbor_up = row - 1
        neighbor_down = row + 1
        neighbor_right = column + 1
        neighbor_left = column - 1
        neighors_values = list()

        # print(">>>> Sag. Seq: {}".format(sag_sequence))
        # print(">>>> Cor. Seq: {}".format(cor_sequence))
        # print("{} x {}\n".format(row, column))

        numrows = self.matRegistration.shape[0]
        numcols = self.matRegistration.shape[1]
        # column = self.cor_sequences.index(cor_sequence)
        # row = self.sag_sequences.index(sag_sequence)

        if self.matRegistration[row, column] == 0.0:
            # print("X: \n{}\n".format(self.matX))
            # print("Y: \n{}\n".format(self.matY))
            # print("Z: \n{}\n".format(self.matZ))
            # print("DL: \n{}\n".format(self.matDL))

            if row == 0 and column == 0:
                # print("Top left corner")
                # print("Neighbor right: {}".format(self.matX[row][neighbor_right]))
                # print("Neighbor down: {}\n".format(self.matX[neighbor_down][column]))

                neighors_values.append(self.matDL[row][neighbor_right])
                neighors_values.append(self.matDL[neighbor_down][column])
                # Get only neighbors that value is different from zero
                neighors_values = list(filter(diff_from_zero, neighors_values))
                # Calculates the mean of the values of the neighboring points that are different from zero
                meanDL = int(sum(neighors_values) / len(neighors_values))
                # Coordinate of 2D point to be converted to 3D space
                lpts = [(self.cor_columns[row], meanDL)]
                # print("Points: {}\n".format(lpts))
                # Converto 2D points to 3D space
                X, Y, Z = self.pc.point3D(plan='Coronal', sequence=cor_sequence, imgnum=1, pts=lpts)
                # print("Z: {}\n".format(Z))
                # Update the matrices in the position analyzed
                self.matX[row, column] = X[0]
                self.matY[row, column] = Y[0]
                self.matZ[row, column] = Z[0]
                neighors_values = []

                '''
                # Z
                neighors_values.append(self.matZ[row][neighbor_right])
                neighors_values.append(self.matZ[neighbor_down][column])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                mean = sum(neighors_values) / len(neighors_values)
                # print("Neighbors values: {}\n".format(neighors_values))
                # print("Mean: {}\n".format(mean))
                self.matZ[row][column] = mean
                # print("Current position: {}\n".format(self.matX[row][column]))
                neighors_values = []
                '''

                self.matRegistration[row][column] = self.gray

            elif row == 0 and column == (numcols - 1):
                # print("Top right corner")
                # print("Neighbor left: {}".format(self.matX[row][neighbor_left]))
                # print("Neighbor down: {}\n".format(self.matX[neighbor_down][column]))

                neighors_values.append(self.matDL[row][neighbor_left])
                neighors_values.append(self.matDL[neighbor_down][column])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                meanDL = int(sum(neighors_values) / len(neighors_values))
                lpts = [(self.cor_columns[row], meanDL)]
                # print("Points: {}\n".format(lpts))
                X, Y, Z = self.pc.point3D(plan='Coronal', sequence=cor_sequence, imgnum=1, pts=lpts)
                # print("Z: {}\n".format(Z))
                self.matX[row, column] = X[0]
                self.matY[row, column] = Y[0]
                self.matZ[row, column] = Z[0]
                neighors_values = []

                '''
                # Z
                neighors_values.append(self.matZ[row][neighbor_left])
                neighors_values.append(self.matZ[neighbor_down][column])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                mean = sum(neighors_values) / len(neighors_values)
                self.matZ[row][column] = mean
                neighors_values = []
                '''

                self.matRegistration[row][column] = self.gray

            elif row == (numrows - 1) and column == 0:
                # print("Bottom left corner")
                # print("Neighbor right: {}".format(self.matX[row][neighbor_right]))
                # print("Neighbor up: {}\n".format(self.matX[neighbor_up][column]))

                neighors_values.append(self.matDL[row][neighbor_right])
                neighors_values.append(self.matDL[neighbor_up][column])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                meanDL = int(sum(neighors_values) / len(neighors_values))
                lpts = [(self.cor_columns[row], meanDL)]
                # print("Points: {}\n".format(lpts))
                X, Y, Z = self.pc.point3D(plan='Coronal', sequence=cor_sequence, imgnum=1, pts=lpts)
                # print("Z: {}\n".format(Z))
                self.matX[row, column] = X[0]
                self.matY[row, column] = Y[0]
                self.matZ[row, column] = Z[0]
                neighors_values = []

                '''
                # Z
                neighors_values.append(self.matZ[row][neighbor_right])
                neighors_values.append(self.matZ[neighbor_up][column])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                mean = sum(neighors_values) / len(neighors_values)
                self.matZ[row][column] = mean
                neighors_values = []
                '''

                self.matRegistration[row][column] = self.gray

            elif row == (numrows - 1) and column == (numcols - 1):
                # print("Bottom right column")
                # print("Neighbor left: {}".format(self.matX[row][neighbor_left]))
                # print("Neighbor up: {}\n".format(self.matX[neighbor_up][column]))

                neighors_values.append(self.matDL[row][neighbor_left])
                neighors_values.append(self.matDL[neighbor_up][column])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                meanDL = int(sum(neighors_values) / len(neighors_values))
                lpts = [(self.cor_columns[row], meanDL)]
                # print("Points: {}\n".format(lpts))
                X, Y, Z = self.pc.point3D(plan='Coronal', sequence=cor_sequence, imgnum=1, pts=lpts)
                # print("Z: {}\n".format(Z))
                self.matX[row, column] = X[0]
                self.matY[row, column] = Y[0]
                self.matZ[row, column] = Z[0]
                neighors_values = []

                '''
                # Z
                neighors_values.append(self.matZ[row][neighbor_left])
                neighors_values.append(self.matZ[neighbor_up][column])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                mean = sum(neighors_values) / len(neighors_values)
                self.matZ[row][column] = mean
                neighors_values = []
                '''

                self.matRegistration[row][column] = self.gray

            elif row == 0 and column > 0 and column < (numcols - 1):
                # print("Middle first line")
                # print("Neighbor right: {}".format(self.matX[row][neighbor_right]))
                # print("Neighbor left: {}".format(self.matX[row][neighbor_left]))
                # print("Neighbor down: {}\n".format(self.matX[neighbor_down][column]))

                neighors_values.append(self.matDL[row][neighbor_right])
                neighors_values.append(self.matDL[row][neighbor_left])
                neighors_values.append(self.matDL[neighbor_down][column])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                meanDL = int(sum(neighors_values) / len(neighors_values))
                lpts = [(self.cor_columns[row], meanDL)]
                # print("Points: {}\n".format(lpts))
                X, Y, Z = self.pc.point3D(plan='Coronal', sequence=cor_sequence, imgnum=1, pts=lpts)
                # print("Z: {}\n".format(Z))
                self.matX[row, column] = X[0]
                self.matY[row, column] = Y[0]
                self.matZ[row, column] = Z[0]
                neighors_values = []

                '''
                # Z
                neighors_values.append(self.matZ[row][neighbor_right])
                neighors_values.append(self.matZ[row][neighbor_left])
                neighors_values.append(self.matZ[neighbor_down][column])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                mean = sum(neighors_values) / len(neighors_values)
                self.matZ[row][column] = mean
                neighors_values = []
                '''

                self.matRegistration[row][column] = self.gray

            elif row == (numrows - 1) and column > 0 and column < (numcols - 1):
                # print("Middle last line")
                # print("Neighbor up: {}".format(self.matX[neighbor_up][column]))
                # print("Neighbor left: {}".format(self.matX[row][neighbor_left]))
                # print("Neighbor right: {}\n".format(self.matX[row][neighbor_right]))

                # print("{} -> {}\n".format(self.cor_columns, self.cor_columns[row]))
                neighors_values.append(self.matDL[neighbor_up][column])
                neighors_values.append(self.matDL[row][neighbor_left])
                neighors_values.append(self.matDL[row][neighbor_right])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                meanDL = int(sum(neighors_values) / len(neighors_values))
                lpts = [(self.cor_columns[row], meanDL)]
                # print("Points: {}\n".format(lpts))
                X, Y, Z = self.pc.point3D(plan='Coronal', sequence=cor_sequence, imgnum=1, pts=lpts)
                # print("Z: {}\n".format(Z))
                self.matX[row, column] = X[0]
                self.matY[row, column] = Y[0]
                self.matZ[row, column] = Z[0]
                neighors_values = []

                '''
                # Z
                neighors_values.append(self.matZ[neighbor_up][column])
                neighors_values.append(self.matZ[row][neighbor_left])
                neighors_values.append(self.matZ[row][neighbor_right])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                mean = sum(neighors_values) / len(neighors_values)
                self.matZ[row][column] = mean
                neighors_values = []
                '''

                self.matRegistration[row][column] = self.gray

            elif row > 0 and row < (numrows - 1) and column == 0:
                # print("Middle first column")
                # print("Neighbor up: {}".format(self.matX[neighbor_up][column]))
                # print("Neighbor down: {}".format(self.matX[neighbor_down][column]))
                # print("Neighbor right: {}\n".format(self.matX[row][neighbor_right]))

                neighors_values.append(self.matDL[neighbor_up][column])
                neighors_values.append(self.matDL[neighbor_down][column])
                neighors_values.append(self.matDL[row][neighbor_right])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                meanDL = int(sum(neighors_values) / len(neighors_values))
                lpts = [(self.cor_columns[row], meanDL)]
                # print("Points: {}\n".format(lpts))
                X, Y, Z = self.pc.point3D(plan='Coronal', sequence=cor_sequence, imgnum=1, pts=lpts)
                # print("Z: {}\n".format(Z))
                self.matX[row, column] = X[0]
                self.matY[row, column] = Y[0]
                self.matZ[row, column] = Z[0]
                neighors_values = []

                '''
                # Z
                neighors_values.append(self.matZ[neighbor_up][column])
                neighors_values.append(self.matZ[neighbor_down][column])
                neighors_values.append(self.matZ[row][neighbor_right])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                mean = sum(neighors_values) / len(neighors_values)
                self.matZ[row][column] = mean
                neighors_values = []
                '''

                self.matRegistration[row][column] = self.gray

            elif row > 0 and row < (numrows - 1) and column == (numcols - 1):
                # print("Middle last column")
                # print("Neighbor up: {}".format(self.matX[neighbor_up][column]))
                # print("Neighbor down: {}".format(self.matX[neighbor_down][column]))
                # print("Neighbor left: {}\n".format(self.matX[row][neighbor_left]))

                neighors_values.append(self.matDL[neighbor_up][column])
                neighors_values.append(self.matDL[neighbor_down][column])
                neighors_values.append(self.matDL[row][neighbor_left])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                meanDL = int(sum(neighors_values) / len(neighors_values))
                lpts = [(self.cor_columns[row], meanDL)]
                # print("Points: {}\n".format(lpts))
                X, Y, Z = self.pc.point3D(plan='Coronal', sequence=cor_sequence, imgnum=1, pts=lpts)
                # print("Z: {}\n".format(Z))
                self.matX[row, column] = X[0]
                self.matY[row, column] = Y[0]
                self.matZ[row, column] = Z[0]
                neighors_values = []

                '''
                # Z
                neighors_values.append(self.matZ[neighbor_up][column])
                neighors_values.append(self.matZ[neighbor_down][column])
                neighors_values.append(self.matZ[row][neighbor_left])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                mean = sum(neighors_values) / len(neighors_values)
                self.matZ[row][column] = mean
                neighors_values = []
                '''

                self.matRegistration[row][column] = self.gray

            else:
                # print("Middle all of it")
                # print("Neighbor up: {}".format(self.matX[neighbor_up][column]))
                # print("Neighbor down: {}".format(self.matX[neighbor_down][column]))
                # print("Neighbor left: {}".format(self.matX[row][neighbor_left]))
                # print("Neighbor right: {}\n".format(self.matX[row][neighbor_right]))

                # print("{} -> {}\n".format(self.cor_columns, self.cor_columns[row]))
                neighors_values.append(self.matDL[neighbor_up][column])
                neighors_values.append(self.matDL[neighbor_down][column])
                neighors_values.append(self.matDL[row][neighbor_left])
                neighors_values.append(self.matDL[row][neighbor_right])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                meanDL = int(sum(neighors_values) / len(neighors_values))
                lpts = [(self.cor_columns[row], meanDL)]
                # print("Points: {}\n".format(lpts))
                X, Y, Z = self.pc.point3D(plan='Coronal', sequence=cor_sequence, imgnum=1, pts=lpts)
                # print("Z: {}\n".format(Z))
                self.matX[row, column] = X[0]
                self.matY[row, column] = Y[0]
                self.matZ[row, column] = Z[0]
                neighors_values = []

                '''
                # Z
                neighors_values.append(self.matZ[neighbor_up][column])
                neighors_values.append(self.matZ[neighbor_down][column])
                neighors_values.append(self.matZ[row][neighbor_left])
                neighors_values.append(self.matZ[row][neighbor_right])
                neighors_values = list(filter(diff_from_zero, neighors_values))
                mean = sum(neighors_values) / len(neighors_values)
                # print("Neighbors values: {}\n".format(neighors_values))
                # print("Mean: {}\n".format(mean))
                self.matZ[row][column] = mean
                # print("Current position: {}\n".format(self.matX[row][column]))
                neighors_values = []
                '''

                self.matRegistration[row][column] = self.gray

            # c = raw_input("?")


def calculate_surface(patient, side, root_sequence, root_img, csequences, lssequences, rssequences):
    reg = Register(patient, side, csequences, lssequences, rssequences)
    root_plan = 'Coronal'

    if side == 0:
        ssequences = lssequences  # Left
    else:
        ssequences = rssequences  # Right

    ldl_nd, lrp_nd = list(), list()

    """ First Step
        dl = diaphragmatic level, rp = respiratory phase -> associated with coronal root image """
    print('(Step 1) Root: Image: {} - Sequence: {}\n'.format(root_img, root_sequence))
    dl_st, rp_st =\
        reg.first_step(
            plan=root_plan,
            sequence=root_sequence,
            imgnum=root_img)
    xpts_st, ypts_st, zpts_st, lpts_st =\
        reg.convert_points(
            step=1,
            root_sequence=root_sequence,
            sequence=root_sequence,
            imgnum=root_img)

    """ Second step """
    ldl_nd, lrp_nd = list(), list()
    lindex_reg_img_nd = []  # images' indexes that were registered in the second step

    for i in range(len(ssequences)):
        # print(ssequences[i])
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

            xpts_nd, ypts_nd, zpts_nd, lpts_nd =\
                reg.convert_points(
                    step=2,
                    root_sequence=root_sequence,
                    sequence=ssequences[i],
                    imgnum=img_reg_nd)
        else:
            lindex_reg_img_nd.append(0)  # Became -1 in the next line instruction

        # print("(Step 2) Registered image: {}\n".format(img_reg_nd))

    lindex_reg_img_nd = list(map(lambda item: item - 1, lindex_reg_img_nd))
    print("(Step 2) Reg. images: {}\n".format(lindex_reg_img_nd))
    # print("(Step 2) X: \n{}\n".format(reg.matX))
    # print("(Step 2) Y: \n{}\n".format(reg.matY))
    # print("(Step 2) Z: \n{}\n".format(reg.matZ))

    # Red points
    for s in range(len(ssequences)):
        dl_nd = ldl_nd[ssequences.index(ssequences[s])]
        # print(">>>> Sagittal sequence: {}, DL: {}\n".format(ssequences[s], dl_nd))
        rp_nd = lrp_nd[ssequences.index(ssequences[s])]

        if dl_nd != -1:
            for c in range(len(csequences)):
                # print(">>>> Coronal sequence: {}\n".format(csequences[c]))
                reg.red_points(
                    plan='Coronal',
                    dlvl_sag_img=dl_nd,
                    rphase_sag_img=rp_nd,
                    root_cor_sequence=root_sequence,
                    cor_sequence=csequences[c],
                    sag_sequence=ssequences[s])

    # Yellow points
    lreg_img, sagsequences = list(), list()
    for s in range(len(ssequences)):
        dl_nd = ldl_nd[ssequences.index(ssequences[s])]
        # print("*" * 40)
        # print(">>>> Sagittal sequence: {}, DL: {}\n".format(ssequences[s], dl_nd))
        rp_nd = lrp_nd[ssequences.index(ssequences[s])]

        if dl_nd != -1:
            for c in range(len(csequences)):
                # print(">>>> Coronal sequence: {}\n".format(csequences[c]))
                reg_img =\
                    reg.yellow_points(
                        plan='Coronal',
                        dlvl_sag_img=dl_nd,
                        rphase_sag_img=rp_nd,
                        root_cor_sequence=root_sequence,
                        cor_sequence=csequences[c],
                        sag_sequence=ssequences[s])

                lreg_img.append(reg_img)

            # Join the lists
            flatten = list(itertools.chain.from_iterable(lreg_img))
            # print("(Yellow) {}, ({})".format(flatten, len(flatten)))
            sagsequences.append(flatten)
        else:
            sagsequences.append([])
    # print("(Yellow) {} ({})".format(sagsequences, len(sagsequences)))
    # Get the index that had more registers
    reg_index = sagsequences.index(max(sagsequences, key=lambda coll: len(coll)))
    root_ssequence = ssequences[reg_index]
    print("Root sagittal sequence: {}\n".format(root_ssequence))

    """ Third step """
    lindex_reg_img_rd = []  # images' indexes that were registered in the third step
    dl_nd = ldl_nd[ssequences.index(root_ssequence)]
    rp_nd = lrp_nd[ssequences.index(root_ssequence)]

    # if dl_nd != -1:
    for i in range(len(csequences)):
        # print(csequences[i])
        img_reg_rd, dl_rd, rp_rd =\
            reg.third_step(
                plan='Coronal',
                dlvl_sag_img=dl_nd,
                rphase_sag_img=rp_nd,
                root_cor_sequence=root_sequence,
                cor_sequence=csequences[i],
                sag_sequence=root_ssequence)
        if img_reg_rd != -1:
            lindex_reg_img_rd.append(img_reg_rd)

            xpts_nd, ypts_nd, zpts_nd, lpts_nd =\
                reg.convert_points(
                    step=3,
                    root_sequence=root_sequence,
                    sequence=csequences[i],
                    imgnum=img_reg_rd)
        else:
            lindex_reg_img_rd.append(0)

    lindex_reg_img_rd = list(map(lambda item: item - 1, lindex_reg_img_rd))
    print("(Step 3) Reg. images: {}\n".format(lindex_reg_img_rd))

    # Purple points
    """
    for s in range(len(ssequences)):
        for c in range(len(csequences)):
            reg.purple_points(
                cor_reg_img=lindex_reg_img_nd,
                sag_reg_img=lindex_reg_img_rd,
                cor_sequence=csequences[c],
                sag_sequence=ssequences[s])
    """

    # Gray points
    for s in range(len(ssequences)):
        for c in range(len(csequences)):
            reg.gray_points(
                cor_sequence=csequences[c],
                sag_sequence=ssequences[s])

    print(reg.matRegistration)

    '''
    # Plot
    xpoints = reg.matX.tolist()
    ypoints = reg.matY.tolist()
    zpoints = reg.matZ.tolist()

    lpts = list()
    for i in range(len(xpoints)):
        for j in range(len(xpoints[0])):
            if xpoints[i][j] != 0.0 or ypoints[i][j] != 0.0 or zpoints[i][j] != 0.0:
                pt = (xpoints[i][j], ypoints[i][j], zpoints[i][j])
                lpts.append(pt)
    # print(lpts)

    pc.plot3D(pts=lpts, howplot=0, dots=0)
    '''

    return reg.matRegistration, reg.matX, reg.matY, reg.matZ


def both_diaphragmatic_surfaces(patient, root_sequence, root_img, csequences, lssequences, rssequences):
    # Left
    lmatLung, lmatX, lmatY, lmatZ =\
        calculate_surface(
            patient=patient,
            side=0,
            root_sequence=root_sequence,
            root_img=root_img,
            csequences=csequences,
            lssequences=lssequences,
            rssequences=rssequences)
    # Right
    rmatLung, rmatX, rmatY, rmatZ =\
        calculate_surface(
            patient=patient,
            side=1,
            root_sequence=root_sequence,
            root_img=root_img,
            csequences=csequences,
            lssequences=lssequences,
            rssequences=rssequences)

    return lmatX, lmatY, lmatZ, rmatX, rmatY, rmatZ


def execute(optplot, patient, side, root_sequence, root_img, csequences, lssequences, rssequences):
    """ Execute the program

    Parameters
    ----------
    optplot: int
        Execution option: 0 - Triangulation | 1 - B-spline | -1 - Both

    patient: string
        Patient's name

    side: int
        User view:
        0 - Left lung
        1 - Right lung
        2 - Both

    root_sequence: int
        Root coronal sequence

    root_img: int
        Root coronal image from root coronal sequence. Represents the respiratory instant

    csequences: list
        List of int with the coronal sequences available

    lssequences: list
        List of int with the sagittal sequences available from the left lung

    rssequences: list
        List of int with the sagittal sequences available from the right lung """

    pc = PointCloud(patient)

    if side == 0 or side == 1:
        matLung, matX, matY, matZ =\
            calculate_surface(
                patient, side, root_sequence, root_img, csequences, lssequences, rssequences)

        xpoints = matX.tolist()
        ypoints = matY.tolist()
        zpoints = matZ.tolist()

        pts = list()
        for i in range(len(xpoints)):
            for j in range(len(xpoints[0])):
                if xpoints[i][j] != 0.0 or ypoints[i][j] != 0.0 or zpoints[i][j] != 0.0:
                    pt = (xpoints[i][j], ypoints[i][j], zpoints[i][j])
                    pts.append(pt)
        # print(pts)

        if optplot == 0:
            pc.plot3D(pts=pts, howplot=0, dots=1)

        elif optplot == 1:
            pc.plotBsplineSurface(
                pts=pts,
                corsequences=csequences,
                lsagsequences=lssequences,
                rsagsequences=rssequences,
                degree=3,
                visualization_type=-1,
                side=side)

        elif optplot == 2:
            pc.plotNURBSSurface(
                pts=pts,
                corsequences=csequences,
                lsagsequences=lssequences,
                rsagsequences=rssequences,
                degree=3,
                visualization_type=-1,
                side=side)

        else:
            pc.plot3D(pts=pts, howplot=0, dots=0)

            pc.plotBsplineSurface(
                pts=pts,
                corsequences=csequences,
                lsagsequences=lssequences,
                rsagsequences=rssequences,
                degree=3,
                visualization_type=-1,
                side=side)

    else:
        lmatX, lmatY, lmatZ, rmatX, rmatY, rmatZ =\
            both_diaphragmatic_surfaces(
                patient, root_sequence, root_img, csequences, lssequences, rssequences)

        lxpoints = lmatX.tolist()
        lypoints = lmatY.tolist()
        lzpoints = lmatZ.tolist()

        lpts = list()
        for i in range(len(lxpoints)):
            for j in range(len(lxpoints[0])):
                if lxpoints[i][j] != 0.0 or lypoints[i][j] != 0.0 or lzpoints[i][j] != 0.0:
                    pt = (lxpoints[i][j], lypoints[i][j], lzpoints[i][j])
                    lpts.append(pt)

        rxpoints = rmatX.tolist()
        rypoints = rmatY.tolist()
        rzpoints = rmatZ.tolist()
        rpts = list()
        for i in range(len(rxpoints)):
            for j in range(len(rxpoints[0])):
                if rxpoints[i][j] != 0.0 or rypoints[i][j] != 0.0 or rzpoints[i][j] != 0.0:
                    pt = (rxpoints[i][j], rypoints[i][j], rzpoints[i][j])
                    rpts.append(pt)

        if optplot == 0:
            figure = plt.figure()
            axes = figure.add_subplot(111, projection='3d')
            axes.set_xlabel('X axis')
            axes.set_ylabel('Y axis')
            axes.set_zlabel('Z axis')

            pc.plotDiaphragm3D(
                allpts=lpts + rpts,
                leftpts=lpts,
                rightpts=rpts,
                triangulation=True,
                plot=True,
                figure=figure,
                axes=axes)

        elif optplot == 1:
            pc.plotBsplineSurfaces(
                left_pts=lpts,
                right_pts=rpts,
                corsequences=csequences,
                lsagsequences=lssequences,
                rsagsequences=rssequences,
                degree=3,
                visualization_type=-1)

        elif optplot == 2:
            pc.plotNURBSSurfaces(
                left_pts=lpts,
                right_pts=rpts,
                corsequences=csequences,
                lsagsequences=lssequences,
                rsagsequences=rssequences,
                degree=3,
                visualization_type=-1)

        else:
            figure = plt.figure()
            axes = figure.add_subplot(111, projection='3d')
            axes.set_xlabel('X axis')
            axes.set_ylabel('Y axis')
            axes.set_zlabel('Z axis')

            pc.plotDiaphragm3D(
                allpts=lpts + rpts,
                leftpts=lpts,
                rightpts=rpts,
                triangulation=True,
                plot=True,
                figure=figure,
                axes=axes)

            pc.plotBsplineSurfaces(
                left_pts=lpts,
                right_pts=rpts,
                corsequences=csequences,
                lsagsequences=lssequences,
                rsagsequences=rssequences,
                degree=3,
                visualization_type=-1)


if __name__ == '__main__':
    try:
        optmode = 0  # 0 - Triangulation | 1 - B-spline | 2 - NURBS | -1 - Both
        patient = 'Iwasawa'
        rootsequence = 9
        side = 0  # 0 - left | 1 - right
        imgnumber = 1  # Instant
        coronalsequences = '9, 10, 11, 12'
        leftsagittalsequences = '1, 2, 3, 4, 5, 6, 7, 8'
        rightsagittalsequences = '12, 13, 14, 15, 16, 17'  # '13, 14, 15, 16, 17'

        txtargv2 = '-mode={}'.format(optmode)
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
                txtargv9)

        if txtargv.find('-mode') != -1:
            txttmp = txtargv.split('-mode')[1]
            txttmp = txttmp.split('=')[1]
            optmode = int(txttmp.split('|')[0])

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

            $ python {} -mode=2 -patient=Iwasawa -rootsequence=9 -side=0 -imgnumber=1
            -coronalsequences=9,10,11
            -leftsagittalsequences=1,2,3,4,5,6,7,8
            -rightsagittalsequences=13,14,15,16,17
            """.format(sys.argv[0]))
        exit()

    execute(
        optplot=optmode,
        patient=patient,
        side=side,
        root_sequence=rootsequence,
        root_img=imgnumber,
        csequences=coronalsequences,
        lssequences=leftsagittalsequences,
        rssequences=rightsagittalsequences)
