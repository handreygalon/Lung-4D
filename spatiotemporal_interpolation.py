#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pydicom
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from util.constant import *


class Interpolate(object):
    """ Spatio-temporal point interpolation """
    def __init__(self, patient, side, rootsequence, corsequences, lsagsequences, rsagsequences):
        self.patient = patient
        self.side = side
        self.rootsequence = rootsequence
        self.corsequences = corsequences
        self.lsagsequences = lsagsequences
        self.rsagsequences = rsagsequences

    def get_register_info(self, side, rootsequence, imgnumber):
        """
        Reads the text file and converts the data to the correct types

        Parameters
        ----------
        side: int
            Represents which lung will be analyzed (0 - left | 1 - right)
        rootsequence: int
            Represents the root coronal sequence
        imgnumber: int
            Represents the respiratory instant

        Returns
        -------
        information: list
            List with all the information about the register
        """
        if side == 0:
            dataset = open(
                '{}/{}-{}.txt'.format(
                    DIR_RESULT_INTERPOLATED_ST_LEFT, rootsequence, imgnumber),
                'r').read().split('\n')
        else:
            dataset = open(
                '{}/{}-{}.txt'.format(
                    DIR_RESULT_INTERPOLATED_ST_RIGHT, rootsequence, imgnumber),
                'r').read().split('\n')
        dataset.pop(-1)

        information, instant, points, tuples = list(), list(), list(), list()

        for i in range(len(dataset)):
            string = dataset[i].split('[')
            # Info
            basicinfo = string[1].replace(" ", "").split(',')
            basicinfo.pop(-1)
            instant.append(basicinfo[0][1:-1])
            instant.append(basicinfo[1][1:-1])
            instant.append(int(basicinfo[2]))
            instant.append(int(basicinfo[3]))
            instant.append(int(basicinfo[4]))
            # Points
            string2 = string[2].replace(']', '')
            string3 = string2.replace('), ', ');')
            string4 = string3.split(';')

            for j in range(len(string4)):
                pts = string4[j].split(',')
                tupla = (float(pts[0][1:]), float(pts[1]), float(pts[2][:-1]))
                tuples.append(tupla)

            points.append(tuples)

            instant.append(tuples)
            tuples = []
            information.append(instant)
            instant = []

        return information

    def point3D(self, plan, sequence, imgnumber, pts):
        xs, ys, zs = list(), list(), list()

        if imgnumber < 10:
            img = pydicom.dcmread("{}/{}/{}/{}/IM_0000{}.dcm".format(
                DIR_DICOM, self.patient, plan, sequence, imgnumber))
        else:
            img = pydicom.dcmread("{}/{}/{}/{}/IM_000{}.dcm".format(
                DIR_DICOM, self.patient, plan, sequence, imgnumber))

        xx_s7 = img.ImageOrientationPatient[0]
        xy_s7 = img.ImageOrientationPatient[1]
        xz_s7 = img.ImageOrientationPatient[2]

        yx_s7 = img.ImageOrientationPatient[3]
        yy_s7 = img.ImageOrientationPatient[4]
        yz_s7 = img.ImageOrientationPatient[5]

        delta_i_s7 = img.PixelSpacing[0]
        delta_j_s7 = img.PixelSpacing[1]

        sx_s7 = img.ImagePositionPatient[0]
        sy_s7 = img.ImagePositionPatient[1]
        sz_s7 = img.ImagePositionPatient[2]

        m = np.matrix([
                      [xx_s7 * delta_i_s7, yx_s7 * delta_j_s7, 0.0, sx_s7],
                      [xy_s7 * delta_i_s7, yy_s7 * delta_j_s7, 0.0, sy_s7],
                      [xz_s7 * delta_i_s7, yz_s7 * delta_j_s7, 0.0, sz_s7],
                      [0.0, 0.0, 0.0, 1.0]
                      ])

        for p in pts:
            m2 = np.matrix([
                           [p[0]],
                           [p[1]],
                           [0.0],
                           [1.0]
                           ])

            m_res = np.dot(m, m2)

            xs.append(m_res.item((0, 0)))
            ys.append(m_res.item((1, 0)))
            zs.append(m_res.item((2, 0)))

        return xs, ys, zs

    def plot3D(self, pts, interpolatepts=None, fig=None, ax=None, plot=True):
        point_size = 2

        xpts, ypts, zpts = list(), list(), list()
        for i in range(len(pts)):
            xpts.append(pts[i][0])
            ypts.append(pts[i][1])
            zpts.append(pts[i][2])

        X = np.asarray(xpts)
        Y = np.asarray(ypts)
        Z = np.asarray(zpts)

        ax.scatter(X, Y, Z, s=point_size, c='r', marker='o')

        if interpolatepts is not None and len(interpolatepts) == 60:
            interpolatex, interpolatey, interpolatez = list(), list(), list()
            for i in range(len(interpolatepts)):
                interpolatex.append(interpolatepts[i][0])
                interpolatey.append(interpolatepts[i][1])
                interpolatez.append(interpolatepts[i][2])
            interX = np.asarray(interpolatex)
            interY = np.asarray(interpolatey)
            interZ = np.asarray(interpolatez)

            ax.scatter(interX, interY, interZ, s=point_size, c='b', marker='^')
        elif interpolatepts is not None and len(interpolatepts) != 60:
            for i in range(len(interpolatepts)):
                interpolatex, interpolatey, interpolatez = list(), list(), list()
                for j in range(len(interpolatepts[i])):
                    interpolatex.append(interpolatepts[i][j][0])
                    interpolatey.append(interpolatepts[i][j][1])
                    interpolatez.append(interpolatepts[i][j][2])
                interX = np.asarray(interpolatex)
                interY = np.asarray(interpolatey)
                interZ = np.asarray(interpolatez)

                ax.scatter(interX, interY, interZ, s=point_size, c='b', marker='^')

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        if plot:
            plt.show()

    def get_silhouettes_positions(self):
        """
        Recovers the positions in the space of all the lung silhouettes. X - Sagittal | Y - Coronal

        Returns
        -------
        leftsagX: list
            X positions of left lung silhouettes
        rightsagX: list
            X positions of right lung silhouettes
        corY: list
            Y positions of lungs silhouettes
        """
        point2D = [(127, 127)]  # point in the middle of the image
        leftsagX, rightsagX, corY = list(), list(), list()

        # Get x values (sagittal)
        for i in range(len(self.lsagsequences)):
            X, Y, Z = self.point3D(
                plan='Sagittal',
                sequence=self.lsagsequences[i],
                imgnumber=1,
                pts=point2D)
            leftsagX.append(X)
        leftsagX = list(itertools.chain.from_iterable(leftsagX))
        # print("Left X: {} ({})".format(leftsagX, len(leftsagX)))

        for i in range(len(self.rsagsequences)):
            X, Y, Z = self.point3D(
                plan='Sagittal',
                sequence=self.rsagsequences[i],
                imgnumber=1,
                pts=point2D)
            rightsagX.append(X)
        rightsagX = list(itertools.chain.from_iterable(rightsagX))
        # print("Right X: {} ({})".format(rightsagX, len(rightsagX)))

        # Get y values (coronal)
        for i in range(len(self.corsequences)):
            X, Y, Z = self.point3D(
                plan='Coronal',
                sequence=self.corsequences[i],
                imgnumber=1,
                pts=point2D)
            corY.append(Y)
        corY = list(itertools.chain.from_iterable(corY))
        # print("Cor Y: {} ({})".format(corY, len(corY)))

        return leftsagX, rightsagX, corY

    def get_sequences(self, dataset, side):
        # Get all sequences
        if side == 0:
            allsagsequences = self.lsagsequences
        else:
            allsagsequences = self.rsagsequences
        allcorsequences = self.corsequences

        # Get the available sagittal sequences
        ssequences, csequences = list(), list()
        for i in range(len(dataset)):
            if dataset[i][4] == 1:
                csequences.append(dataset[i][2])
            elif dataset[i][4] == 2:
                ssequences.append(dataset[i][2])
            else:
                csequences.append(dataset[i][2])
        csequences = sorted(csequences)
        ssequences = sorted(ssequences)
        # print("All sagittal sequences: {} ({})".format(allsagsequences, len(allsagsequences)))
        # print("Available sagittal sequences: {} ({})".format(ssequences, len(ssequences)))
        # print("All coronal sequences: {} ({})".format(allcorsequences, len(allcorsequences)))
        # print("Available coronal sequences: {} ({})".format(csequences, len(csequences)))

        return allsagsequences, ssequences, allcorsequences, csequences

    def get_points(self, dataset):
        """
        Get only the points of all silhouettes of the current sequence being analyzed

        Parameters
        ----------
        dataset: list
            Contains all the information about the current sequence

        Returns
        -------
        pts1: list
            List of points referring to step 1 in the register
        pts2: list
            List of points referring to step 2 in the register
        pts3: list
            List of points referring to step 3 in the register
        pts: list
            List with all the points (pts1 + pts2 + pts3)
        """
        pts1, pts2, pts3, pts = list(), list(), list(), list()
        for i in range(len(dataset)):
            if dataset[i][4] == 1:
                pts1.append(dataset[i][5])
            elif dataset[i][4] == 2:
                pts2.append(dataset[i][5])
            else:
                pts3.append(dataset[i][5])

        # Union all points - use in this way to plot
        points1 = pts1.copy()
        points2 = pts2.copy()
        points3 = pts3.copy()
        points1 = list(itertools.chain.from_iterable(points1))
        points2 = list(itertools.chain.from_iterable(points2))
        points3 = list(itertools.chain.from_iterable(points3))
        pts.append(points1)
        pts.append(points2)
        pts.append(points3)
        pts = list(itertools.chain.from_iterable(pts))

        return pts1, pts2, pts3, pts

    def get_columns(self, points1, points2, points3):
        """
        Retrieves the positions of all the sequences and of those that are available, i.e.,
        disregarding the positions that did not generate registered

        Parameters
        ----------

        Returns
        -------
        leftsagX: list
            Columns of the left lung sequences (Sagittal)
        rightsagX: list
            Columns of the right lung sequences (Sagittal)
        corY: list
            Columns of the sequences (Coronal)
        sagavailableX: list
            Available lung sequences columns (Sagittal)
        coravailableY: list
            Available lung sequences columns (Coronal)
        sagdiff: list
            Missing columns (Sagittal)
        cordiff: list
            Missing columns (Coronal)
        """
        # Get all silhouettes positions (X values (sag) | Y values (cor))
        leftsagX, rightsagX, corY = self.get_silhouettes_positions()
        # sagX = leftsagX + rightsagX
        # print("Sagittal columns (X): {} ({})".format(sagX, len(sagX)))
        # print("Coronal columns (Y): {} ({})\n".format(corY, len(corY)))

        # Get the silhouettes available
        sagavailableX, coravailableY = list(), list()
        coravailableY.append(points1[0][0][1])
        for i in range(len(points2)):
            sagavailableX.append(points2[i][0][0])
        for i in range(len(points3)):
            coravailableY.append(points3[i][0][1])
        sagavailableX = sorted(sagavailableX)
        coravailableY = sorted(coravailableY)
        # print("Sagittal available columns (X): {} ({})".format(sagavailableX, len(sagavailableX)))
        # print("Coronal available columns (Y): {} ({})\n".format(coravailableY, len(coravailableY)))

        # Is there any missing silhouette?
        if side == 0:
            sagdiff = sorted(list(set(leftsagX) - set(sagavailableX)))
        else:
            sagdiff = sorted(list(set(rightsagX) - set(sagavailableX)))
        cordiff = sorted(list(set(corY) - set(coravailableY)))
        # print("Difference Sagittal: {}".format(sagdiff))
        # print("Difference Coronal: {}\n".format(cordiff))

        return leftsagX, rightsagX, corY, sagavailableX, coravailableY, sagdiff, cordiff

    def check_temporal_availability(self, dataset, side, rootsequence, imgnumber, sagcoldiff, corcoldiff):
        """
        Checks whether at the anterior and/or posterior instant of breathing there is a sequence in the same
        position as the sequence that is missing

        Parameters
        ----------
        dataset: list
        side: int
        rootsequence: int
        imgnumber: int
        sagcoldiff: list
        corcoldiff: list

        Returns
        -------
        temporalavailability: dict
            key = position of missing sequence in the respiratory instant
            value = availability of the previous and posterior sequences
                -1: Neither the previous nor the posterior sequence are available
                0: Only the previous sequence is available
                1: Only the posterior sequence is available
                2: The previous and posterior sequences are available
        """

        def checkAvailability(registerstep, missingsequences, allsequences, availablesequences, imgnumber):
            """ """
            temporalavailability = {}

            for i in range(len(missingsequences)):
                if imgnumber == 1:
                    # First instant!
                    posterior_instant = imgnumber + 1

                    dataset_posterior = self.get_register_info(side, rootsequence, posterior_instant)

                    allsagsequences_posterior, availablesagsequences_posterior, allcorsequences_posterior, availablecorsequences_posterior =\
                        self.get_sequences(
                            dataset=dataset_posterior,
                            side=side)
                    # print("Posterior instant: {}".format(posterior_instant))
                    # print("Available posterior sagittal sequences: {} ({})".format(availablesagsequences_posterior, len(availablesagsequences_posterior)))
                    # print("Available posterior coronal sequences: {} ({})".format(allcorsequences_posterior, len(availablecorsequences_posterior)))

                    availablesagsequences_previous, availablecorsequences_previous =\
                        list(), list()

                elif imgnumber > 1 and imgnumber < 50:
                    # Middle instant!
                    previous_instant = imgnumber - 1
                    posterior_instant = imgnumber + 1

                    dataset_previous = self.get_register_info(side, rootsequence, previous_instant)
                    dataset_posterior = self.get_register_info(side, rootsequence, posterior_instant)

                    allsagsequences_previous, availablesagsequences_previous, allcorsequences_previous, availablecorsequences_previous =\
                        self.get_sequences(
                            dataset=dataset_previous,
                            side=side)
                    # print("Previous instant: {}".format(previous_instant))
                    # print("Available previous sagittal sequences: {} ({})".format(availablesagsequences_posterior, len(availablesagsequences_posterior)))
                    # print("Available previous coronal sequences: {} ({})".format(availablecorsequences_previous, len(availablecorsequences_previous)))

                    allsagsequences_posterior, availablesagsequences_posterior, allcorsequences_posterior, availablecorsequences_posterior =\
                        self.get_sequences(
                            dataset=dataset_posterior,
                            side=side)
                    # print("Posterior instant: {}".format(posterior_instant))
                    # print("Available posterior sagittal sequences: {} ({})".format(availablesagsequences_posterior, len(availablesagsequences_posterior)))
                    # print("Available posterior coronal sequences: {} ({})".format(allcorsequences_posterior, len(availablecorsequences_posterior)))

                elif imgnumber == 50:
                    # Last instant!
                    previous_instant = imgnumber - 1

                    dataset_previous = self.get_register_info(side, rootsequence, previous_instant)

                    allsagsequences_previous, availablesagsequences_previous, allcorsequences_previous, availablecorsequences_previous =\
                        self.get_sequences(
                            dataset=dataset_previous,
                            side=side)
                    # print("Previous instant: {}".format(previous_instant))
                    # print("Available previous sagittal sequences: {} ({})".format(availablesagsequences_posterior, len(availablesagsequences_posterior)))

                    availablesagsequences_posterior, availablecorsequences_posterior =\
                        list(), list()

                else:
                    print("Invalid image number!")

                # index = allsequences.index(missingsequences[i])
                # print("Missing sequence(s): [{}] = {}".format(index, missingsequences[i]))

                if registerstep == 2:
                    if missingsequences[i] in availablesagsequences_previous and\
                       missingsequences[i] in availablesagsequences_posterior:
                        # Same position as the missing sequence in the previous and posterior instant is avalable
                        temporalavailability[missingsequences[i]] = 2

                    elif missingsequences[i] in availablesagsequences_previous:
                        # Same position as the missing sequence in the posterior instant is avalable
                        temporalavailability[missingsequences[i]] = 0

                    elif missingsequences[i] in availablesagsequences_posterior:
                        # Same position as the missing sequence in the previous instant is avalable
                        temporalavailability[missingsequences[i]] = 1

                    else:
                        temporalavailability[missingsequences[i]] = -1

                else:
                    if missingsequences[i] in availablecorsequences_previous and\
                       missingsequences[i] in availablecorsequences_posterior:
                        # Same position as the missing sequence in the previous and posterior instant is avalable
                        temporalavailability[missingsequences[i]] = 2

                    elif missingsequences[i] in availablecorsequences_previous:
                        # Same position as the missing sequence in the posterior instant is avalable
                        temporalavailability[missingsequences[i]] = 0

                    elif missingsequences[i] in availablecorsequences_posterior:
                        # Same position as the missing sequence in the previous instant is avalable
                        temporalavailability[missingsequences[i]] = 1

                    else:
                        temporalavailability[missingsequences[i]] = -1

                # print("Temporal availability: {}\n".format(temporalavailability))
                # c = input("?")

            temporalavailability = collections.OrderedDict(sorted(temporalavailability.items()))
            # print("Temporal availability: {}".format(temporalavailability))
            # c = input("?")

            return temporalavailability

        allsagsequences, availablesagsequences, allcorsequences, availablecorsequences =\
            self.get_sequences(
                dataset=dataset,
                side=side)
        # print("All sagittal sequences: {}".format(allsagsequences))
        # print("Available sagittal sequences: {}".format(availablesagsequences))
        # print("All coronal sequences: {}".format(allcorsequences))
        # print("Available coronal sequences: {}".format(availablecorsequences))

        sagtemporalavailability, cortemporalavailability = {}, {}

        if len(sagcoldiff) > 0:
            # List of positions that have not been registered
            missing_sag_sequences = sorted(list(set(allsagsequences) - set(availablesagsequences)))
            print("Missing sagittal sequences (temporal): {} ({})\n".format(missing_sag_sequences, len(missing_sag_sequences)))

            sagtemporalavailability = checkAvailability(
                registerstep=2,
                missingsequences=missing_sag_sequences,
                allsequences=allsagsequences,
                availablesequences=availablesagsequences,
                imgnumber=imgnumber)
            # print("\nDictionary temporal availability: {}\n".format(sagtemporalavailability))

        if len(corcoldiff) > 0:
            # List of positions that have not been registered
            missing_cor_sequences = sorted(list(set(allcorsequences) - set(availablecorsequences)))
            print("Missing coronal sequences (temporal): {} ({})\n".format(missing_cor_sequences, len(missing_cor_sequences)))

            cortemporalavailability = checkAvailability(
                registerstep=3,
                missingsequences=missing_cor_sequences,
                allsequences=allcorsequences,
                availablesequences=availablecorsequences,
                imgnumber=imgnumber)
            # print("\nDictionary temporal availability: {}\n".format(cortemporalavailability))

        return sagtemporalavailability, cortemporalavailability

    def check_spatio_availability(self, dataset, side, rootsequence, imgnumber, sagcoldiff, corcoldiff):
        """
        Checks whether at the same instant there is a sequence in the previous and/or posterior position

        Parameters
        ----------
        dataset: list
        side: int
        rootsequence: int
        imgnumber: int
        sagcoldiff: list
        corcoldiff: list

        Returns
        -------
        spatioavailability: dict
            key = position of missing sequence in the respiratory instant
            value = availability of the previous and posterior sequences
                -1: Neither the previous nor the posterior sequence are available
                0: Only the previous sequence is available
                1: Only the posterior sequence is available
                2: The previous and posterior sequences are available
        """

        def checkAvailability(missingsequences, allsequences, availablesequences):
            """ Check availability of the missing sequence """
            spatioavailability = {}

            for i in range(len(missingsequences)):
                index = allsequences.index(missingsequences[i])
                # print("Missing sequence: [{}] = {}".format(index, missingsequences[i]))
                # print("All sequences: {} ({})".format(allsequences, len(allsequences)))
                # print("Available sequences: {} ({})".format(availablesequences, len(availablesequences)))

                if index == 0:
                    # First silhouette - Gets the position of the posterior sequence
                    previous_sequence = -1
                    posterior_sequence = allsequences[index + 1]
                    # print("Posterior sequence: {}".format(posterior_sequence))

                elif index == len(allsequences) - 1:
                    # Last silhouette - Gets the position of the previous sequence
                    previous_sequence = allsequences[index - 1]
                    posterior_sequence = -1
                    # print("Previous sequence: {}".format(previous_sequence))

                else:
                    # Middle silhouette
                    # Gets the position of the previous sequence
                    previous_sequence = allsequences[index - 1]
                    # Gets the position of the posterior sequence
                    posterior_sequence = allsequences[index + 1]

                # Check availability
                if previous_sequence in availablesequences and\
                        posterior_sequence in availablesequences:
                    # Previous and posterior sequence are available
                    spatioavailability[missingsequences[i]] = 2

                elif previous_sequence in availablesequences:
                    # Previous sequence is available
                    spatioavailability[missingsequences[i]] = 0

                elif posterior_sequence in availablesequences:
                    # Posterior sequence is available
                    spatioavailability[missingsequences[i]] = 1

                else:
                    spatioavailability[missingsequences[i]] = -1

                # print("Spatio availability: {}\n".format(spatioavailability))
            # print("Spatio availability: {}".format(spatioavailability))

            spatioavailability = collections.OrderedDict(sorted(spatioavailability.items()))

            return spatioavailability

        allsagsequences, availablesagsequences, allcorsequences, availablecorsequences =\
            self.get_sequences(
                dataset=dataset,
                side=side)
        # print("All sagittal sequences: {}".format(allsagsequences))
        # print("Available sagittal sequences: {}".format(availablesagsequences))
        # print("All coronal sequences: {}".format(allcorsequences))
        # print("Available coronal sequences: {}".format(availablecorsequences))

        sagspatioavailability, corspatioavailability = {}, {}

        if len(sagcoldiff) > 0:
            # List of positions that have not been registered
            missing_sag_sequences = sorted(list(set(allsagsequences) - set(availablesagsequences)))
            # print("Missing sagittal sequences (spatio): {} ({})\n".format(missing_sag_sequences, len(missing_sag_sequences)))

            sagspatioavailability = checkAvailability(
                missingsequences=missing_sag_sequences,
                allsequences=allsagsequences,
                availablesequences=availablesagsequences)
            # print("\nDictionary spatio availability: {}\n".format(sagspatioavailability))

        if len(corcoldiff) > 0:
            # List of positions that have not been registered
            missing_cor_sequences = sorted(list(set(allcorsequences) - set(availablecorsequences)))
            # print("Missing coronal sequences (spatio): {} ({})\n".format(missing_cor_sequences, len(missing_cor_sequences)))

            corspatioavailability = checkAvailability(
                missingsequences=missing_cor_sequences,
                allsequences=allcorsequences,
                availablesequences=availablecorsequences)
            # print("\nDictionary spatio availability: {}\n".format(corspatioavailability))

        return sagspatioavailability, corspatioavailability

    def check_availability(self, dataset, side, rootsequence, imgnumber, sagcoldiff, corcoldiff):
        dictsagspatioavailability, dictcorspatioavailability =\
            self.check_spatio_availability(
                dataset=dataset,
                side=side,
                rootsequence=rootsequence,
                imgnumber=imgnumber,
                sagcoldiff=sagcoldiff,
                corcoldiff=corcoldiff)

        dictsagtemporalavailability, dictcortemporalavailability =\
            self.check_temporal_availability(
                dataset=dataset,
                side=side,
                rootsequence=rootsequence,
                imgnumber=imgnumber,
                sagcoldiff=sagcoldiff,
                corcoldiff=corcoldiff)

        return dictsagspatioavailability, dictcorspatioavailability, dictsagtemporalavailability, dictcortemporalavailability

    def write_points(self, side, step, rootsequence, sequence, analyzedimage, points):
        # Saves register information about an respiration instant in a .txt file
        instant_information = list()

        instant_information.append(self.patient)

        if step == 2:
            instant_information.append('Sagittal')
        else:
            instant_information.append('Coronal')

        instant_information.append(sequence)
        imgnumber = 0
        instant_information.append(imgnumber)
        instant_information.append(step)
        instant_information.append(points)

        if side == 0:
            file = open('{}/{}-{}.txt'.format(DIR_RESULT_INTERPOLATED_ST_LEFT, rootsequence, analyzedimage), 'a')
        else:
            file = open('{}/{}-{}.txt'.format(DIR_RESULT_INTERPOLATED_ST_RIGHT, rootsequence, analyzedimage), 'a')
        file.write("{}\n".format(instant_information))
        file.close()

    def get_spatio_points(self, dataset, registerstep, side, sequence):
        """ """
        previous_points, posterior_points = list(), list()

        allsagsequences, availablesagsequences, allcorsequences, availablecorsequences =\
            self.get_sequences(
                dataset=dataset,
                side=side)
        # print("{} ({})".format(allsagsequences, len(allsagsequences)))
        # print("{} ({})".format(availablesagsequences, len(availablesagsequences)))

        if registerstep == 2:
            index = allsagsequences.index(sequence)
            # print("Sequence: [{}]={}".format(index, sequence))

            if index == 0:
                # First silhouette! Gets the position of the posterior sequence
                posterior_sequence = allsagsequences[index + 1]
                # print("Posterior sequence: {}".format(posterior_sequence))

                for i in range(len(dataset)):
                    if dataset[i][4] == 2 and dataset[i][2] == posterior_sequence:
                        posterior_points = dataset[i][5]

            elif index == len(allsagsequences) - 1:
                # Last silhouette! Gets the position of the previous sequence
                previous_sequence = allsagsequences[index - 1]
                # print("Previous sequence: {}".format(previous_sequence))

                for i in range(len(dataset)):
                    if dataset[i][4] == 2 and dataset[i][2] == previous_sequence:
                        previous_points = dataset[i][5]

            else:
                # Middle silhouette! Gets the position of the previous sequence
                previous_sequence = allsagsequences[index - 1]  # sequence - 1
                # print("Previous sequence: {}".format(previous_sequence))
                # index_previous_sequence = allsagsequences.index(previous_sequence)
                # print("Index previous sequence: {}".format(index_previous_sequence))

                # Gets the position of the posterior sequence
                posterior_sequence = allsagsequences[index + 1]  # sequence + 1
                # print("Posterior sequence: {}".format(posterior_sequence))
                # index_posterior_sequence = allsagsequences.index(posterior_sequence)
                # print("Index posterior sequence: {}".format(index_posterior_sequence))

                for i in range(len(dataset)):
                    if dataset[i][4] == 2 and dataset[i][2] == previous_sequence:
                        previous_points = dataset[i][5]
                for i in range(len(dataset)):
                    if dataset[i][4] == 2 and dataset[i][2] == posterior_sequence:
                        posterior_points = dataset[i][5]

        else:
            index = allcorsequences.index(sequence)
            # print("Sequence: [{}]={}".format(index, sequence))

            if index == 0:
                # First silhouette! Gets the position of the posterior sequence
                posterior_sequence = allcorsequences[index + 1]
                # print("Posterior sequence: {}".format(posterior_sequence))

                for i in range(len(dataset)):
                    if dataset[i][4] == 1 or dataset[i][4] == 3 and dataset[i][2] == posterior_sequence:
                        posterior_points = dataset[i][5]

            elif index == len(allcorsequences) - 1:
                # Last silhouette! Gets the position of the previous sequence
                previous_sequence = allcorsequences[index - 1]
                # print("Previous sequence: {}".format(previous_sequence))

                for i in range(len(dataset)):
                    if dataset[i][4] == 1 or dataset[i][4] == 3 and dataset[i][2] == previous_sequence:
                        previous_points = dataset[i][5]

            else:
                # Middle silhouette! Gets the position of the previous sequence
                previous_sequence = allcorsequences[index - 1]  # sequence - 1
                # print("Previous sequence: {}".format(previous_sequence))
                # index_previous_sequence = allcorsequences.index(previous_sequence)
                # print("Index previous sequence: {}".format(index_previous_sequence))

                # Gets the position of the posterior sequence
                posterior_sequence = allcorsequences[index + 1]  # sequence + 1
                # print("Posterior sequence: {}".format(posterior_sequence))
                # index_posterior_sequence = allcorsequences.index(posterior_sequence)
                # print("Index posterior sequence: {}".format(index_posterior_sequence))

                for i in range(len(dataset)):
                    if dataset[i][4] == 1 or dataset[i][4] == 3 and dataset[i][2] == previous_sequence:
                        previous_points = dataset[i][5]
                for i in range(len(dataset)):
                    if dataset[i][4] == 1 or dataset[i][4] == 3 and dataset[i][2] == posterior_sequence:
                        posterior_points = dataset[i][5]

        return previous_points, posterior_points

    def get_temporal_points(self, registerstep, side, rootsequence, imgnumber, sequence):
        """ """
        previous_points, posterior_points = list(), list()
        previous_instant = imgnumber - 1
        posterior_instant = imgnumber + 1
        # print("Previous instant: {}".format(previous_instant))
        # print("Posterior instant: {}".format(posterior_instant))

        if registerstep == 2:
            if imgnumber == 1:
                dataset_posterior = self.get_register_info(side, rootsequence, posterior_instant)

                for i in range(len(dataset_posterior)):
                    if dataset_posterior[i][4] == 2 and dataset_posterior[i][2] == sequence:
                        posterior_points = dataset_posterior[i][5]

            elif imgnumber > 1 and imgnumber < 50:
                dataset_previous = self.get_register_info(side, rootsequence, previous_instant)
                dataset_posterior = self.get_register_info(side, rootsequence, posterior_instant)

                for i in range(len(dataset_previous)):
                    if dataset_previous[i][4] == 2 and dataset_previous[i][2] == sequence:
                        previous_points = dataset_previous[i][5]

                for i in range(len(dataset_posterior)):
                    if dataset_posterior[i][4] == 2 and dataset_posterior[i][2] == sequence:
                        posterior_points = dataset_posterior[i][5]

            elif imgnumber == 50:
                dataset_previous = self.get_register_info(side, rootsequence, previous_instant)

                for i in range(len(dataset_previous)):
                    if dataset_previous[i][4] == 2 and dataset_previous[i][2] == sequence:
                        previous_points = dataset_previous[i][5]

            else:
                print("Invalid image number!")

        else:
            if imgnumber == 1:
                dataset_posterior = self.get_register_info(side, rootsequence, posterior_instant)

                for i in range(len(dataset_posterior)):
                    if dataset_posterior[i][4] == 1 or dataset_posterior[i][4] == 3 and dataset_posterior[i][2] == sequence:
                        posterior_points = dataset_posterior[i][5]

            elif imgnumber > 1 and imgnumber < 50:
                dataset_previous = self.get_register_info(side, rootsequence, previous_instant)
                dataset_posterior = self.get_register_info(side, rootsequence, posterior_instant)
                # print(dataset_previous[0])
                # print(dataset_posterior[0])

                for i in range(len(dataset_previous)):
                    # print(sequence)
                    # print(dataset_previous[i][2])
                    # print(dataset_previous[i][4])
                    if dataset_previous[i][2] == sequence and dataset_previous[i][4] == 1 or dataset_previous[i][4] == 3:
                        # print(dataset_previous[i][5])
                        previous_points = dataset_previous[i][5]

                for i in range(len(dataset_posterior)):
                    if dataset_posterior[i][4] == 1 or dataset_posterior[i][4] == 3 and dataset_posterior[i][2] == sequence:
                        posterior_points = dataset_posterior[i][5]

            elif imgnumber == 50:
                dataset_previous = self.get_register_info(side, rootsequence, previous_instant)

                for i in range(len(dataset_previous)):
                    if dataset_posterior[i][4] == 1 or dataset_posterior[i][4] == 3 and dataset_previous[i][2] == sequence:
                        previous_points = dataset_previous[i][5]

            else:
                print("Invalid image number!")

        return previous_points, posterior_points

    def interpolation(self, dataset, dsagspatioavailability, dcorspatioavailability, dsagtemporalavailability, dcortemporalavailability, rootsequence, imgnumber, side, sagcoldiff, corcoldiff, save=0):
        """
        -1: The previous and the posterior sequence are not available
        0: Only the previous sequence is available
        1: Only the posterior sequence is available
        2: The previous and posterior sequences are available
        """
        print("\nDictionary sagittal spatio availability: {} ({})".format(dsagspatioavailability, len(dsagspatioavailability)))
        print("Dictionary coronal spatio availability: {} ({})\n".format(dcorspatioavailability, len(dcorspatioavailability)))
        print("Dictionary sagittal temporal availability: {} ({})".format(dsagtemporalavailability, len(dsagtemporalavailability)))
        print("Dictionary coronal temporal availability: {} ({})\n".format(dcortemporalavailability, len(dcortemporalavailability)))

        interpolated_points = list()

        def interpolates(registerstep, missingsequences, missingcol, meanx=None, meany=None, meanz=None):
            if registerstep == 2:
                """ In the sagittal plane, the value of X is already known """
                # print("Missing sequence: {}".format(missingsequences))
                print("Missing column: {}".format(missingcol))
                X = missingcol

                for i in range(len(missingsequences)):
                    # Create a list of interpolated points in the position that there was no register
                    interpolated_pts = list()
                    for j in range(60):
                        interpolated_pts.append((X, meany[j], meanz[j]))

            else:
                """ In the coronal plane, the value of Y is already known """
                # print("Missing sequence: {}".format(missingsequences))
                print("Missing column: {}".format(missingcol))
                Y = missingcol

                for i in range(len(missingsequences)):
                    # Create a list of interpolated points in the position that there was no register
                    interpolated_pts = list()
                    for j in range(60):
                        interpolated_pts.append((meanx[j], Y, meanz[j]))

            return interpolated_pts

        if len(sagcoldiff) > 0:
            # interpolated_points = list()

            for (ks, vs), (kt, vt) in zip(dsagspatioavailability.items(), dsagtemporalavailability.items()):
                print("\n*** Sagittal *** ")
                print("Spatio -> {}: {}".format(ks, vs))
                print("Temporal -> {}: {}".format(kt, vt))

                if vs == 2 and vt == -1:
                    print("Spatio = 2 | Temporal = -1")

                    temporal_previous_points, temporal_posterior_points = list(), list()

                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    spatio_previous_ypts, spatio_previous_zpts = list(), list()
                    spatio_posterior_ypts, spatio_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_ypts.append(spatio_previous_points[i][1])
                        spatio_posterior_ypts.append(spatio_posterior_points[i][1])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])
                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2) / 2 for y1, y2 in zip(spatio_previous_ypts, spatio_posterior_ypts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(spatio_previous_zpts, spatio_posterior_zpts)]

                elif vs == 2 and vt == 0:
                    print("Spatio = 2 | Temporal = 0")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_ypts, spatio_previous_zpts = list(), list()
                    spatio_posterior_ypts, spatio_posterior_zpts = list(), list()
                    temporal_previous_ypts, temporal_previous_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_ypts.append(spatio_previous_points[i][1])
                        spatio_posterior_ypts.append(spatio_posterior_points[i][1])

                        temporal_previous_ypts.append(temporal_previous_points[i][1])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])
                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2 + y3) / 3 for y1, y2, y3 in zip(spatio_previous_ypts, spatio_posterior_ypts, temporal_previous_ypts)]
                    mean_zpoints = [(z1 + z2 + z3) / 3 for z1, z2, z3 in zip(spatio_previous_zpts, spatio_posterior_zpts, temporal_previous_zpts)]

                elif vs == 2 and vt == 1:
                    print("Spatio = 2 | Temporal = 1")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_ypts, spatio_previous_zpts = list(), list()
                    spatio_posterior_ypts, spatio_posterior_zpts = list(), list()
                    temporal_posterior_ypts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_ypts.append(spatio_previous_points[i][1])
                        spatio_posterior_ypts.append(spatio_posterior_points[i][1])

                        temporal_posterior_ypts.append(temporal_posterior_points[i][1])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])
                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2 + y3) / 3 for y1, y2, y3 in zip(spatio_previous_ypts, spatio_posterior_ypts, temporal_posterior_ypts)]
                    mean_zpoints = [(z1 + z2 + z3) / 3 for z1, z2, z3 in zip(spatio_previous_zpts, spatio_posterior_zpts, temporal_posterior_zpts)]

                elif vs == 2 and vt == 2:
                    print("Spatio = 2 | Temporal = 2")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_ypts, spatio_previous_zpts = list(), list()
                    spatio_posterior_ypts, spatio_posterior_zpts = list(), list()
                    temporal_previous_ypts, temporal_previous_zpts = list(), list()
                    temporal_posterior_ypts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_ypts.append(spatio_previous_points[i][1])
                        spatio_posterior_ypts.append(spatio_posterior_points[i][1])

                        temporal_previous_ypts.append(temporal_previous_points[i][1])
                        temporal_posterior_ypts.append(temporal_posterior_points[i][1])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])
                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])
                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2 + y3 + y4) / 4 for y1, y2, y3, y4 in zip(spatio_previous_ypts, spatio_posterior_ypts, temporal_previous_ypts, temporal_posterior_ypts)]
                    mean_zpoints = [(z1 + z2 + z3 + z4) / 4 for z1, z2, z3, z4 in zip(spatio_previous_zpts, spatio_posterior_zpts, temporal_previous_zpts, temporal_posterior_zpts)]

                elif vs == 1 and vt == -1:
                    print("Spatio = 1 | Temporal = -1")
                    temporal_previous_points, temporal_posterior_points = list(), list()

                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    spatio_posterior_ypts, spatio_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_posterior_ypts.append(spatio_posterior_points[i][1])

                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = spatio_posterior_ypts
                    mean_zpoints = spatio_posterior_zpts

                elif vs == 1 and vt == 0:
                    print("Spatio = 1 | Temporal = 0")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_posterior_ypts, spatio_posterior_zpts = list(), list()
                    temporal_previous_ypts, temporal_previous_zpts = list(), list()

                    for i in range(60):
                        spatio_posterior_ypts.append(spatio_posterior_points[i][1])

                        temporal_previous_ypts.append(temporal_previous_points[i][1])

                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2) / 2 for y1, y2 in zip(spatio_posterior_ypts, temporal_previous_ypts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(spatio_posterior_zpts, temporal_previous_zpts)]

                elif vs == 1 and vt == 1:
                    print("Spatio = 1 | Temporal = 1")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_posterior_ypts, spatio_posterior_zpts = list(), list()
                    temporal_posterior_ypts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_posterior_ypts.append(spatio_posterior_points[i][1])

                        temporal_posterior_ypts.append(temporal_posterior_points[i][1])

                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2) / 2 for y1, y2 in zip(spatio_posterior_ypts, temporal_posterior_ypts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(spatio_posterior_zpts, temporal_posterior_zpts)]

                elif vs == 1 and vt == 2:
                    print("Spatio = 1 | Temporal = 2")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_posterior_ypts, spatio_posterior_zpts = list(), list()
                    temporal_previous_ypts, temporal_previous_zpts = list(), list()
                    temporal_posterior_ypts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_posterior_ypts.append(spatio_posterior_points[i][1])

                        temporal_previous_ypts.append(temporal_previous_points[i][1])
                        temporal_posterior_ypts.append(temporal_posterior_points[i][1])

                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])
                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2 + y3) / 3 for y1, y2, y3 in zip(spatio_posterior_ypts, temporal_previous_ypts, temporal_posterior_ypts)]
                    mean_zpoints = [(z1 + z2 + z3) / 3 for z1, z2, z3 in zip(spatio_posterior_zpts, temporal_previous_zpts, temporal_posterior_zpts)]

                elif vs == 0 and vt == -1:
                    print("Spatio = 0 | Temporal = -1")
                    temporal_previous_points, temporal_posterior_points = list(), list()

                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    spatio_previous_ypts, spatio_previous_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_ypts.append(spatio_previous_points[i][1])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = spatio_previous_ypts
                    mean_zpoints = spatio_previous_zpts

                elif vs == 0 and vt == 0:
                    print("Spatio = 0 | Temporal = 0")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_ypts, spatio_previous_zpts = list(), list()
                    temporal_previous_ypts, temporal_previous_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_ypts.append(spatio_previous_points[i][1])

                        temporal_previous_ypts.append(temporal_previous_points[i][1])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2) / 2 for y1, y2 in zip(spatio_previous_ypts, temporal_previous_ypts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(spatio_previous_zpts, temporal_previous_zpts)]

                elif vs == 0 and vt == 1:
                    print("Spatio = 0 | Temporal = 1")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_ypts, spatio_previous_zpts = list(), list()
                    temporal_posterior_ypts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_ypts.append(spatio_previous_points[i][1])

                        temporal_posterior_ypts.append(temporal_posterior_points[i][1])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])

                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2) / 2 for y1, y2 in zip(spatio_previous_ypts, temporal_posterior_ypts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(spatio_previous_zpts, temporal_posterior_zpts)]

                elif vs == 0 and vt == 2:
                    print("Spatio = 0 | Temporal = 2")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=2,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_ypts, spatio_previous_zpts = list(), list()
                    temporal_previous_ypts, temporal_previous_zpts = list(), list()
                    temporal_posterior_ypts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_ypts.append(spatio_previous_points[i][1])

                        temporal_previous_ypts.append(temporal_previous_points[i][1])
                        temporal_posterior_ypts.append(temporal_posterior_points[i][1])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])
                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2 + y3) / 3 for y1, y2, y3 in zip(spatio_previous_ypts, temporal_previous_ypts, temporal_posterior_ypts)]
                    mean_zpoints = [(z1 + z2 + z3) / 3 for z1, z2, z3 in zip(spatio_previous_zpts, temporal_previous_zpts, temporal_posterior_zpts)]

                elif vs == -1 and vt == -1:
                    print("Spatio = -1 | Temporal = -1")
                    spatio_previous_points, spatio_posterior_points = list(), list()
                    temporal_previous_points, temporal_posterior_points = list(), list()

                elif vs == -1 and vt == 0:
                    print("Spatio = -1 | Temporal = 0")
                    spatio_previous_points, spatio_posterior_points = list(), list()

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    temporal_previous_ypts, temporal_previous_zpts = list(), list()

                    for i in range(60):
                        temporal_previous_ypts.append(temporal_previous_points[i][1])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = temporal_previous_ypts
                    mean_zpoints = temporal_previous_zpts

                elif vs == -1 and vt == 1:
                    print("Spatio = -1 | Temporal = 1")
                    spatio_previous_points, spatio_posterior_points = list(), list()

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    temporal_posterior_ypts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        temporal_posterior_ypts.append(temporal_posterior_points[i][1])

                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = temporal_posterior_ypts
                    mean_zpoints = temporal_posterior_zpts

                elif vs == -1 and vt == 2:
                    print("Spatio = -1 | Temporal = 2")

                    spatio_previous_points, spatio_posterior_points = list(), list()

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=2,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    # Mean
                    temporal_previous_ypts, temporal_posterior_ypts = list(), list()
                    temporal_previous_zpts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        temporal_previous_ypts.append(temporal_previous_points[i][1])
                        temporal_posterior_ypts.append(temporal_posterior_points[i][1])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])
                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of y and z
                    mean_ypoints = [(y1 + y2) / 2 for y1, y2 in zip(temporal_previous_ypts, temporal_posterior_ypts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(temporal_previous_zpts, temporal_posterior_zpts)]

                else:
                    print("Invalid option!")

                # print("Spatio ->\n Previous: {} \n Posterior: {}".format(len(spatio_previous_points), len(spatio_posterior_points)))
                # print("Temporal ->\n Previous: {} \n Posterior: {}".format(len(temporal_previous_points), len(temporal_posterior_points)))

                missingsagsequences = list(dsagspatioavailability.keys())  # or: list(dtemporalavailability.keys())

                # for i in range(len(missingsagsequences)):
                #     print("{}: {}".format(missingsagsequences[i], sagcoldiff[i]))

                missingcol = sagcoldiff[list(dsagspatioavailability.keys()).index(ks)]
                points = interpolates(
                    registerstep=2,
                    missingsequences=missingsagsequences,
                    missingcol=missingcol,
                    meany=mean_ypoints,
                    meanz=mean_zpoints)
                interpolated_points.append(points)

            if save == 1:
                # Write points interpolated in txt file
                for i in range(len(missingsagsequences)):
                    self.write_points(
                        side=side,
                        step=2,
                        rootsequence=rootsequence,
                        sequence=missingsagsequences[i],
                        analyzedimage=imgnumber,
                        points=interpolated_points[i])

        if len(corcoldiff) > 0:
            # interpolated_points = list()

            for (ks, vs), (kt, vt) in zip(dcorspatioavailability.items(), dcortemporalavailability.items()):
                print("\n*** Coronal ***")
                print("Spatio -> {}: {}".format(ks, vs))
                print("Temporal -> {}: {}".format(kt, vt))

                if vs == 2 and vt == -1:
                    print("Spatio = 2 | Temporal = -1")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_xpts, spatio_previous_zpts = list(), list()
                    spatio_posterior_xpts, spatio_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_xpts.append(spatio_previous_points[i][0])
                        spatio_posterior_xpts.append(spatio_posterior_points[i][0])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])
                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = [(x1 + x2) / 2 for x1, x2 in zip(spatio_previous_xpts, spatio_posterior_xpts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(spatio_previous_zpts, spatio_posterior_zpts)]

                elif vs == 2 and vt == 0:
                    print("Spatio = 2 | Temporal = 0")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_xpts, spatio_previous_zpts = list(), list()
                    spatio_posterior_xpts, spatio_posterior_zpts = list(), list()
                    temporal_previous_xpts, temporal_previous_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_xpts.append(spatio_previous_points[i][0])
                        spatio_posterior_xpts.append(spatio_posterior_points[i][0])

                        temporal_previous_xpts.append(temporal_previous_points[i][0])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])
                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = [(x1 + x2 + x3) / 3 for x1, x2, x3 in zip(spatio_previous_xpts, spatio_posterior_xpts, temporal_previous_xpts)]
                    mean_zpoints = [(z1 + z2 + z3) / 3 for z1, z2, z3 in zip(spatio_previous_zpts, spatio_posterior_zpts, temporal_previous_zpts)]

                elif vs == 2 and vt == 1:
                    print("Spatio = 2 | Temporal = 1")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_xpts, spatio_previous_zpts = list(), list()
                    spatio_posterior_xpts, spatio_posterior_zpts = list(), list()
                    temporal_posterior_xpts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_xpts.append(spatio_previous_points[i][0])
                        spatio_posterior_xpts.append(spatio_posterior_points[i][0])

                        temporal_posterior_xpts.append(temporal_posterior_points[i][0])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])
                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = [(x1 + x2 + x3) / 3 for x1, x2, x3 in zip(spatio_previous_xpts, spatio_posterior_xpts, temporal_posterior_xpts)]
                    mean_zpoints = [(z1 + z2 + z3) / 3 for z1, z2, z3 in zip(spatio_previous_zpts, spatio_posterior_zpts, temporal_posterior_zpts)]

                elif vs == 2 and vt == 2:
                    print("Spatio = 2 | Temporal = 2")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_xpts, spatio_previous_zpts = list(), list()
                    spatio_posterior_xpts, spatio_posterior_zpts = list(), list()
                    temporal_previous_xpts, temporal_previous_zpts = list(), list()
                    temporal_posterior_xpts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_xpts.append(spatio_previous_points[i][0])
                        spatio_posterior_xpts.append(spatio_posterior_points[i][0])

                        temporal_previous_xpts.append(temporal_previous_points[i][0])
                        temporal_posterior_xpts.append(temporal_posterior_points[i][0])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])
                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])
                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = [(x1 + x2 + x3 + x4) / 4 for x1, x2, x3, x4 in zip(spatio_previous_xpts, spatio_posterior_xpts, temporal_previous_xpts, temporal_posterior_xpts)]
                    mean_zpoints = [(z1 + z2 + z3 + z4) / 4 for z1, z2, z3, z4 in zip(spatio_previous_zpts, spatio_posterior_zpts, temporal_previous_zpts, temporal_posterior_zpts)]

                elif vs == 1 and vt == -1:
                    print("Spatio = 1 | Temporal = -1")
                    temporal_previous_points, temporal_posterior_points = list(), list()

                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    spatio_posterior_xpts, spatio_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_posterior_xpts.append(spatio_posterior_points[i][0])

                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = spatio_posterior_xpts
                    mean_zpoints = spatio_posterior_zpts

                elif vs == 1 and vt == 0:
                    print("Spatio = 1 | Temporal = 0")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_posterior_xpts, spatio_posterior_zpts = list(), list()
                    temporal_previous_xpts, temporal_previous_zpts = list(), list()

                    for i in range(60):
                        spatio_posterior_xpts.append(spatio_posterior_points[i][0])

                        temporal_previous_xpts.append(temporal_previous_points[i][0])

                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])

                    # Calculate the mean of y and z
                    mean_xpoints = [(x1 + x2) / 2 for x1, x2 in zip(spatio_posterior_xpts, temporal_previous_xpts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(spatio_posterior_zpts, temporal_previous_zpts)]

                elif vs == 1 and vt == 1:
                    print("Spatio = 1 | Temporal = 1")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_posterior_xpts, spatio_posterior_zpts = list(), list()
                    temporal_posterior_xpts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_posterior_xpts.append(spatio_posterior_points[i][0])

                        temporal_posterior_xpts.append(temporal_posterior_points[i][0])

                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = [(x1 + x2) / 2 for x1, x2 in zip(spatio_posterior_xpts, temporal_posterior_xpts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(spatio_posterior_zpts, temporal_posterior_zpts)]

                elif vs == 1 and vt == 2:
                    print("Spatio = 1 | Temporal = 2")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_posterior_xpts, spatio_posterior_zpts = list(), list()
                    temporal_previous_xpts, temporal_previous_zpts = list(), list()
                    temporal_posterior_xpts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_posterior_xpts.append(spatio_posterior_points[i][0])

                        temporal_previous_xpts.append(temporal_previous_points[i][0])
                        temporal_posterior_xpts.append(temporal_posterior_points[i][0])

                        spatio_posterior_zpts.append(spatio_posterior_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])
                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = [(x1 + x2 + x3) / 3 for x1, x2, x3 in zip(spatio_posterior_xpts, temporal_previous_xpts, temporal_posterior_xpts)]
                    mean_zpoints = [(z1 + z2 + z3) / 3 for z1, z2, z3 in zip(spatio_posterior_zpts, temporal_previous_zpts, temporal_posterior_zpts)]

                elif vs == 0 and vt == -1:
                    print("Spatio = 0 | Temporal = -1")
                    temporal_previous_points, temporal_posterior_points = list(), list()

                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    spatio_previous_xpts, spatio_previous_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_xpts.append(spatio_previous_points[i][0])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = spatio_previous_ypts
                    mean_zpoints = spatio_previous_zpts

                elif vs == 0 and vt == 0:
                    print("Spatio = 0 | Temporal = 0")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_xpts, spatio_previous_zpts = list(), list()
                    temporal_previous_xpts, temporal_previous_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_xpts.append(spatio_previous_points[i][0])

                        temporal_previous_xpts.append(temporal_previous_points[i][0])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = [(x1 + x2) / 2 for x1, x2 in zip(spatio_previous_xpts, temporal_previous_xpts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(spatio_previous_zpts, temporal_previous_zpts)]

                elif vs == 0 and vt == 1:
                    print("Spatio = 0 | Temporal = 1")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_xpts, spatio_previous_zpts = list(), list()
                    temporal_posterior_xpts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_xpts.append(spatio_previous_points[i][0])

                        temporal_posterior_xpts.append(temporal_posterior_points[i][0])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])

                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = [(x1 + x2) / 2 for x1, x2 in zip(spatio_previous_xpts, temporal_posterior_xpts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(spatio_previous_zpts, temporal_posterior_zpts)]

                elif vs == 0 and vt == 2:
                    print("Spatio = 0 | Temporal = 2")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    spatio_previous_xpts, spatio_previous_zpts = list(), list()
                    temporal_previous_xpts, temporal_previous_zpts = list(), list()
                    temporal_posterior_xpts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        spatio_previous_xpts.append(spatio_previous_points[i][0])

                        temporal_previous_xpts.append(temporal_previous_points[i][0])
                        temporal_posterior_xpts.append(temporal_posterior_points[i][0])

                        spatio_previous_zpts.append(spatio_previous_points[i][2])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])
                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = [(x1 + x2 + x3) / 3 for x1, x2, x3 in zip(spatio_previous_xpts, temporal_previous_xpts, temporal_posterior_xpts)]
                    mean_zpoints = [(z1 + z2 + z3) / 3 for z1, z2, z3 in zip(spatio_previous_zpts, temporal_previous_zpts, temporal_posterior_zpts)]

                elif vs == -1 and vt == -1:
                    print("Spatio = -1 | Temporal = -1")
                    spatio_previous_points, spatio_posterior_points = list(), list()
                    temporal_previous_points, temporal_posterior_points = list(), list()

                elif vs == -1 and vt == 0:
                    print("Spatio = -1 | Temporal = 0")
                    spatio_previous_points, spatio_posterior_points = list(), list()

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    temporal_previous_xpts, temporal_previous_zpts = list(), list()

                    for i in range(60):
                        temporal_previous_xpts.append(temporal_previous_points[i][0])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = temporal_previous_xpts
                    mean_zpoints = temporal_previous_zpts

                elif vs == -1 and vt == 1:
                    print("Spatio = -1 | Temporal = 1")
                    spatio_previous_points, spatio_posterior_points = list(), list()

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    temporal_posterior_xpts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        temporal_posterior_xpts.append(temporal_posterior_points[i][0])

                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = temporal_posterior_xpts
                    mean_zpoints = temporal_posterior_zpts

                elif vs == -1 and vt == 2:
                    print("Spatio = -1 | Temporal = 2")
                    spatio_previous_points, spatio_posterior_points =\
                        self.get_spatio_points(
                            dataset=dataset,
                            registerstep=3,
                            side=side,
                            sequence=kt)

                    temporal_previous_points, temporal_posterior_points =\
                        self.get_temporal_points(
                            registerstep=3,
                            side=side,
                            rootsequence=rootsequence,
                            imgnumber=imgnumber,
                            sequence=kt)

                    temporal_previous_xpts, temporal_previous_zpts = list(), list()
                    temporal_posterior_xpts, temporal_posterior_zpts = list(), list()

                    for i in range(60):
                        temporal_previous_xpts.append(temporal_previous_points[i][0])
                        temporal_posterior_xpts.append(temporal_posterior_points[i][0])

                        temporal_previous_zpts.append(temporal_previous_points[i][2])
                        temporal_posterior_zpts.append(temporal_posterior_points[i][2])

                    # Calculate the mean of x and z
                    mean_xpoints = [(x1 + x2) / 2 for x1, x2 in zip(temporal_previous_xpts, temporal_posterior_xpts)]
                    mean_zpoints = [(z1 + z2) / 2 for z1, z2 in zip(temporal_previous_zpts, temporal_posterior_zpts)]

                else:
                    print("Invalid option!")

                # print("Spatio ->\n Previous: {} \n Posterior: {}".format(len(spatio_previous_points), len(spatio_posterior_points)))
                # print("Temporal ->\n Previous: {} \n Posterior: {}".format(len(temporal_previous_points), len(temporal_posterior_points)))

                missingcorsequences = list(dcorspatioavailability.keys())  # or: list(dcorspatioavailability.keys())
                # for i in range(len(missingcorsequences)):
                #     print("{}: {}".format(missingcorsequences[i], corcoldiff[i]))

                missingcol = corcoldiff[list(dcorspatioavailability.keys()).index(ks)]
                points = interpolates(
                    registerstep=3,
                    missingsequences=missingcorsequences,
                    missingcol=missingcol,
                    meanx=mean_xpoints,
                    meanz=mean_zpoints)
                interpolated_points.append(points)

            if save == 1:
                # Write points interpolated in txt file
                for i in range(len(missingcorsequences)):
                    self.write_points(
                        side=side,
                        step=3,
                        rootsequence=rootsequence,
                        sequence=missingcorsequences[i],
                        analyzedimage=imgnumber,
                        points=interpolated_points[i])

        return interpolated_points


def execute(patient, rootsequence, side, imgnumber, coronalsequences, lsagsequences, rsagsequences, save=0):
    flag_interpolate = False

    interpolate = Interpolate(
        patient=patient,
        side=side,
        rootsequence=rootsequence,
        corsequences=coronalsequences,
        lsagsequences=leftsagittalsequences,
        rsagsequences=rightsagittalsequences)

    current_dataset = interpolate.get_register_info(
        side=side,
        rootsequence=rootsequence,
        imgnumber=imgnumber)

    pts1, pts2, pts3, pts = interpolate.get_points(current_dataset)

    lsagX, rsagX, corY, sagavailableX, coravailableY, sagcoldiff, corcoldiff =\
        interpolate.get_columns(
            points1=pts1,
            points2=pts2,
            points3=pts3)
    print("================= Columns =================")
    print("Left sagittal columns: {}".format(lsagX))
    print("Right sagittal columns: {}".format(rsagX))
    print("Coronal columns: {}\n".format(corY))
    print("Missing sagittal columns: {}".format(sagcoldiff))
    print("Missing coronal columns: {}".format(corcoldiff))
    print("===========================================")

    dictsagspatioavailability, dictcorspatioavailability, dictsagtemporalavailability, dictcortemporalavailability =\
        interpolate.check_availability(
            dataset=current_dataset,
            side=side,
            rootsequence=rootsequence,
            imgnumber=imgnumber,
            sagcoldiff=sagcoldiff,
            corcoldiff=corcoldiff)
    # print("\nDictionary sagittal spatio availability: {} ({})".format(dictsagspatioavailability, len(dictsagspatioavailability)))
    # print("Dictionary coronal spatio availability: {} ({})\n".format(dictcorspatioavailability, len(dictcorspatioavailability)))
    # print("Dictionary sagittal temporal availability: {} ({})".format(dictsagtemporalavailability, len(dictsagtemporalavailability)))
    # print("Dictionary coronal temporal availability: {} ({})\n".format(dictcortemporalavailability, len(dictcortemporalavailability)))

    if len(dictsagspatioavailability) > 0 or len(dictcorspatioavailability) > 0 or\
       len(dictsagtemporalavailability) > 0 or len(dictcortemporalavailability) > 0:
        flag_interpolate = True

        interpolatepts = interpolate.interpolation(
            dataset=current_dataset,
            dsagspatioavailability=dictsagspatioavailability,
            dcorspatioavailability=dictcorspatioavailability,
            dsagtemporalavailability=dictsagtemporalavailability,
            dcortemporalavailability=dictcortemporalavailability,
            rootsequence=rootsequence,
            imgnumber=imgnumber,
            side=side,
            sagcoldiff=sagcoldiff,
            corcoldiff=corcoldiff,
            save=save)
    else:
        flag_interpolate = False
        print("Does not need interpolation!")

    if flag_interpolate:
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')
        axes.set_xlabel('X axis')
        axes.set_ylabel('Y axis')
        axes.set_zlabel('Z axis')
        interpolate.plot3D(
            pts=pts,
            interpolatepts=interpolatepts,
            fig=figure,
            ax=axes,
            plot=True)


if __name__ == '__main__':
    try:
        patient = 'Matsushita'  # 'Iwasawa'
        rootsequence = 14  # 9
        side = 0  # 0 - left | 1 - right | 2 - Both
        imgnumber = 1  # Instant
        # Iwasawa
        # coronalsequences = '4, 5, 6, 7, 8, 9, 10, 11, 12'
        # leftsagittalsequences = '1, 2, 3, 4, 5, 6, 7, 8'
        # rightsagittalsequences = '12, 13, 14, 15, 16, 17'
        # Matsushita
        coronalsequences = '4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21'
        leftsagittalsequences = '2, 3, 4, 5, 6, 7'
        rightsagittalsequences = '10, 11, 12, 13, 14, 15'
        save = 0  # 0 - Not save points in txt file | 1 - Save

        txtargv2 = '-patient={}'.format(patient)
        txtargv3 = '-rootsequence={}'.format(rootsequence)
        txtargv4 = '-side={}'.format(side)
        txtargv5 = '-imgnumber={}'.format(imgnumber)
        txtargv6 = '-coronalsequences={}'.format(coronalsequences)
        txtargv7 = '-leftsagittalsequences={}'.format(leftsagittalsequences)
        txtargv8 = '-rightsagittalsequences={}'.format(rightsagittalsequences)
        txtargv9 = '-save={}'.format(save)

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

        txtargv = '{}|{}|{}|{}|{}|{}|{}|{}'.format(
            txtargv2,
            txtargv3,
            txtargv4,
            txtargv5,
            txtargv6,
            txtargv7,
            txtargv8,
            txtargv9)

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

        if txtargv.find('-save') != -1:
            txttmp = txtargv.split('-save')[1]
            txttmp = txttmp.split('=')[1]
            save = int(txttmp.split('|')[0])

    except ValueError:
        print(
            """
            Example of use:\n

            $ python {} -patient=Iwasawa -rootsequence=9 -side=0 -imgnumber=1
            -coronalsequences=4,5,6,7,8,9,10,11,12
            -leftsagittalsequences=1,2,3,4,5,6,7,8
            -rightsagittalsequences=12,13,14,15,16,17
            -save=0
            """.format(sys.argv[0]))
        exit()

    execute(
        patient=patient,
        rootsequence=rootsequence,
        side=side,
        imgnumber=imgnumber,
        coronalsequences=coronalsequences,
        lsagsequences=leftsagittalsequences,
        rsagsequences=rightsagittalsequences,
        save=save)
