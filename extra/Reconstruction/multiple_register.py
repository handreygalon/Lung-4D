#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from operator import itemgetter


class MultipleRegister(object):

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

    def __init__(self, patient):
        self.patient = patient
        self.DIR =\
            '/home/handrey/Documents/UDESC/DICOM/2DST_diaphragm/{}'\
            .format(self.patient)

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
        img = cv2.imread('{}/Coronal/{}'.format(self.DIR, file), 0)

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
        img = cv2.imread('{}/Sagittal/{}'.format(self.DIR, file), 0)

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

        """ Compare the current diaphragmatic level to the previous one
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

    def register_first_step(self, image, lvlCor, lvlSag, phaseCor, phaseSag):
        """
        Check diaphragmatic level and respiratory phase to do correct register

        1) Choose a coronal sequence
        2) Take an image as root (that represent the instant)
           Several sagittal sequences crosses the coronal image at
           intersection positions in 3D space
        3) Define the diaphragmatic level and respiratory phase to each
           position in 3D space to the coronal image
        4) The instant at which each sequence register the coronal image
           is determined

        Parameters
        ----------
        image: int
            Number of the root image (coronal)

        lvlCor: list
            List of lists of coronal plan. Each column has a list of points
            that represents diaphragmatic level, because sometimes, the curve
            that represents the respiratory pattern occupy more than one pixel
            on each column

        lvlSag: list
            List of lists of sagittal plan. (Same explanation as lvl_cor param)

        phaseCor: list
            List of respiratory phase of each image in coronal plan (each
            column on 2DST image) - (0 - Expiration | 1 - Inspiration)

        phaseSag: list
            List of respiratory phase of each image in sagittal plan (each
            column on 2DST image) - (0 - Expiration | 1 - Inspiration)

        Return
        ------
        llvl_sag: list
            List of sagittal images that have the same diaphragmatic level and
            respiratory phase that root coronal image
        """
        llvl_sag = []

        """
        Check for each coronal image, which sagittal images have the same
        diaphragmatic level

        llvl_sag: list of sagittal images that have the same diaphragmatic lvl
        that the root coronal image
        """
        # print(lvlCor[image])
        for i in range(len(lvlSag)):
            if [j for j in lvlCor[image] if j in lvlSag[i]]:
                llvl_sag.append(i)
        # print('{}: {}'.format(image, llvl_sag))

        """
        Check if the sagittal images that have the same diaphragmatic level
        that the coronal image have the same respiratory phase, if so,
        do the register, otherwise, delete from the list that have the
        sagittal images with the same diaphragmatic level that the
        coronal image
        """
        # print(phaseCor[image])
        i = 0
        lpop = []
        for val in llvl_sag:
            if phaseSag[val] != phaseCor[image]:
                lpop.append(val)
        # print(lpop)
        for item in lpop:
            if item in llvl_sag:
                llvl_sag.remove(item)
        # lpop = []
        # print('{}: {}'.format(image, llvl_sag))

        return llvl_sag

    def register_second_step(self, image, lvlCor, lvlSag, phaseCor, phaseSag):
        """
        Parameters
        ----------
        image: int
            Number of the root image (sagittal)

        lvlCor: list
            List of lists of coronal plan. Each column has a list of points
            that represents diaphragmatic level, because sometimes, the curve
            that represents the respiratory pattern occupy more than one pixel
            on each column

        lvlSag: list
            List of lists of sagittal plan. (Same explanation as lvl_cor param)

        phaseCor: list
            List of respiratory phase of each image in coronal plan (each
            column on 2DST image) - (0 - Expiration | 1 - Inspiration)

        phaseSag: list
            List of respiratory phase of each image in sagittal plan (each
            column on 2DST image) - (0 - Expiration | 1 - Inspiration)

        Return
        ------
        llvl_cor: list
            List of coronal images that have the same diaphragmatic level and
            respiratory phase that root sagittal image
        """
        llvl_cor = []
        for i in range(len(lvlCor)):
            if [j for j in lvlSag[image] if j in lvlCor[i]]:
                llvl_cor.append(i)
        # print('{}: {}'.format(lreg_sag[0], llvl_cor))

        i = 0
        lpop = []
        for val in llvl_cor:
            if phaseCor[val] != phaseSag[image]:
                lpop.append(val)
        for item in lpop:
            if item in lpop:
                if item in llvl_cor:
                    llvl_cor.remove(item)
        # print('{}: {}'.format(lreg_sag[0], llvl_cor))

        return llvl_cor


def first_step(register, ds, cor_sequences):
    """
    Get the available coronal sequences and images name that represents
    the respiratory pattern (referring to the same coronal sequence)
    """
    for i in range(len(cor_sequences)):
        def_cor_seq = cor_sequences[i]
        imgs2DST = [j for j in ds if j.split('-')[1] == str(def_cor_seq)]
        print('*' * 80)
        print('Coronal sequence: {}'.format(def_cor_seq))
        print('2DSTs: {}'.format(imgs2DST))

        """
        For each image referring to the respiratory pattern of a coronal
        sequence, get diaphragmatic levels and respiratory phases
        """
        for item_cor in imgs2DST:
            print('Pattern coronal: {}'.format(item_cor))

            pts_pattern_cor =\
                register.pattern_coronal('{}.png'.format(item_cor))

            if len(pts_pattern_cor) > 50:
                diaphragmatic_lvl_cor =\
                    register.diaphragmatic_level_coronal(pts_pattern_cor)
                respiratory_phase_cor =\
                    register.respiratory_phase_coronal(diaphragmatic_lvl_cor)

            """
            An image of the coronal sequence is used as a root to make the
            registration in the other sagittal sequences
            """
            # for image in range(50):
            for image in range(1):
                print("Image: {}".format(image))
                print('*' * 80)

                """
                Check the register to all sagittal sequences using the root img
                """
                for item_sag in imgs2DST:
                    print('Pattern sagittal: {}'.format(item_sag))

                    pts_pattern_sag =\
                        register.pattern_sagittal('{}.png'.format(item_sag))

                    if len(pts_pattern_sag) > 50:
                        diaphragmatic_lvl_sag =\
                            register.diaphragmatic_level_sagittal(
                                pts_pattern_sag)
                        respiratory_phase_sag =\
                            register.respiratory_phase_sagittal(
                                diaphragmatic_lvl_sag)

                        """
                        First step of register: With a root coronal image is
                        made the register with all sagittal sequences
                        """
                        reg_first_step = register.register_first_step(
                            image,
                            diaphragmatic_lvl_cor, diaphragmatic_lvl_sag,
                            respiratory_phase_cor, respiratory_phase_sag)

                        print('Registered: {}'.format(reg_first_step))
                        print("")
                        """
                        With each sagittal registered with root coronal image
                        is made the register with each one of them with coronal
                        sequences (except the coronal sequence with the root
                        image)
                        """
                        # second_step(
                        #     register, ds, reg_first_step,
                        #     cor_sequences, def_cor_seq, item_sag)
                    else:
                        reg_first_step = []

                        print('Registered: {}'.format(reg_first_step))
                        print("")

                    continuar = raw_input("Continue 1st step? ")
                    if continuar == 's':
                        pass
                    print("")
                # continuar = raw_input("Continue? ")
                # print("")


def second_step(register, ds, regFirstStep, corSeq, corSeqUsed, sagPattern):
    if corSeqUsed in corSeq:
        corSeq.remove(corSeqUsed)
    # print('\t{}'.format(corSeq))

    """
    Get sagittal pattern that have an image registered with coronal root image
    and get diaphragmatic levels and respiratory phases from that sagittal
    sequence
    """
    print('\t2nd Step: ')
    print('\tPattern sagittal (2nd step): {}'.format(sagPattern))

    pts_pattern_sag =\
        register.pattern_sagittal('{}.png'.format(sagPattern))

    if len(pts_pattern_sag) > 50:
        diaphragmatic_lvl_sag =\
            register.diaphragmatic_level_coronal(pts_pattern_sag)
        respiratory_phase_sag =\
            register.respiratory_phase_coronal(diaphragmatic_lvl_sag)

        """
        Get the available coronal sequences and images name that represents
        the respiratory pattern (referring to the same coronal sequence)
        """
        for i in range(len(corSeq)):
            def_cor_seq = corSeq[i]
            imgs2DST =\
                [j for j in ds if j.split('-')[1] == str(def_cor_seq)]
            # print('\tCoronal sequence: {}'.format(def_cor_seq))
            # print('\t2DSTs: {}'.format(imgs2DST))

            """
            The images of the sagittal sequence that were registered with
            coronal root image are use now as a root to make the
            registration in the other coronal sequences
            """
            for image in range(len(regFirstStep)):
                print("\tImage (2nd step): {}".format(regFirstStep[image]))
                print("\t" + "*" * 72)

                for item_cor in imgs2DST:
                    print('\tPattern coronal (2nd step): {}'.format(item_cor))

                    pts_pattern_cor =\
                        register.pattern_sagittal('{}.png'.format(item_cor))

                    if len(pts_pattern_cor) > 50:
                        diaphragmatic_lvl_cor =\
                            register.diaphragmatic_level_coronal(
                                pts_pattern_cor)
                        respiratory_phase_cor =\
                            register.respiratory_phase_sagittal(
                                diaphragmatic_lvl_cor)

                        """
                        Second step of register: With a root sagittal image
                        is made the register with all coronal sequences
                        """
                        reg_second_step = register.register_second_step(
                            regFirstStep[image],
                            diaphragmatic_lvl_cor, diaphragmatic_lvl_sag,
                            respiratory_phase_cor, respiratory_phase_sag)
                    else:
                        reg_second_step = []

                    print('\tRegistered: {}'.format(reg_second_step))
                    print("")

                    continuar = raw_input("Continue 2nd step? ")
                    if continuar == 's':
                        pass
                    print("")


if __name__ == '__main__':
    ds = open('../mapping.txt', 'r').read().split('\n')
    ds.pop(-1)

    cor_sequences = list(set([int(i.split('-')[1]) for i in ds]))
    sag_sequences = list(set([int(i.split('-')[3]) for i in ds]))

    # patient = 'Matsushita'
    patient = 'Iwasawa'
    register = MultipleRegister(patient)

    first_step(register, ds, cor_sequences)
