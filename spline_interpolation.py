#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pydicom
# import dicom

from geomdl import BSpline
from geomdl import utilities as utils
from geomdl import Multi

from util.constant import *


class BSplineInterpolate(object):
    """ Using B-splines curves to points interpolation. Obs. Must contain exterme silhouettes """

    def __init__(self, patient, side, rootsequence, corsequences, lsagsequences, rsagsequences):
        self.patient = patient
        self.side = side
        self.rootsequence = rootsequence
        self.corsequences = corsequences
        self.lsagsequences = lsagsequences
        self.rsagsequences = rsagsequences

    def get_register_info(self, side, rootsequence, imgnumber):
        if side == 0:
            dataset = open(
                '{}/{}-{}.txt'.format(
                    DIR_RESULT_INTERPOLATED_BSPLINE_LEFT, rootsequence, imgnumber),
                'r').read().split('\n')
        else:
            dataset = open(
                '{}/{}-{}.txt'.format(
                    DIR_RESULT_INTERPOLATED_BSPLINE_RIGHT, rootsequence, imgnumber),
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

    def get_points(self, dataset, option=0):
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

        if option == 0:
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

        else:
            for i in range(len(dataset)):
                if dataset[i][4] == 1:
                    pts1.append(dataset[i][5])
                elif dataset[i][4] == 2:
                    pts2.append(dataset[i][5])
                else:
                    pts3.append(dataset[i][5])

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
        # print("Sagittal columns - Left (X): {} ({})".format(leftsagX, len(leftsagX)))
        # print("Sagittal columns - Right (X): {} ({})".format(rightsagX, len(rightsagX)))
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

    def check_extremes(self, side, allleftcolsag, allrightcolsag, allcolcor, availablecolsag, availablecolcor):
        """ Verifies that the silhouettes of the extremities of the lungs are available """

        # print("All left sagittal columns: {} ({})".format(allleftcolsag, len(allleftcolsag)))
        # print("All right sagittal columns: {} ({})".format(allrightcolsag, len(allrightcolsag)))
        # print("Available sagittal columns: {} ({})".format(availablecolsag, len(availablecolsag)))
        # print("All coronal columns: {} ({})".format(allcolcor, len(allcolcor)))
        # print("Available coronal columns: {} ({})\n".format(availablecolcor, len(availablecolcor)))

        # Get the first and last coronal column in the current instant
        firstcorcol = allcolcor[0]
        lastcorcol = allcolcor[-1]

        # Check if the first coronal column is present
        if firstcorcol in availablecolcor:
            firstcor = True
        else:
            firstcor = False
        # Check if the last coronal column is present
        if lastcorcol in availablecolcor:
            lastcor = True
        else:
            lastcor = False

        # Get the first and last sagittal column in the current instant
        if side == 0:
            firstsagcol = allleftcolsag[0]
            lastsagcol = allleftcolsag[-1]
        else:
            firstsagcol = allrightcolsag[0]
            lastsagcol = allrightcolsag[-1]

        # Check if the first sagittal column is present
        if firstsagcol in availablecolsag:
            firstsag = True
        else:
            firstsag = False
        # Check if the lasst sagittal column is present
        if lastsagcol in availablecolsag:
            lastsag = True
        else:
            lastsag = False

        # print("First sagittal column: {} - {}".format(firstsagcol, firstsag))
        # print("Last sagittal column: {} - {}".format(lastsagcol, lastsag))
        # print("First coronal column: {} - {}".format(firstcorcol, firstcor))
        # print("Last coronal column: {} - {}\n".format(lastcorcol, lastcor))

        return firstsag, lastsag, firstcor, lastcor

    def point3D(self, plan, sequence, imgnumber, pts):
        xs, ys, zs = list(), list(), list()

        if imgnumber < 10:
            img = pydicom.dcmread("{}/{}/{}/{}/IM_0000{}.dcm".format(
                DIR_DICOM, self.patient, plan, sequence, imgnumber))
            # img = dicom.read_file("{}/{}/{}/{}/IM_0000{}.dcm".format(
            #     DIR_DICOM, self.patient, plan, sequence, imgnumber))
        else:
            img = pydicom.dcmread("{}/{}/{}/{}/IM_000{}.dcm".format(
                DIR_DICOM, self.patient, plan, sequence, imgnumber))
            # img = dicom.read_file("{}/{}/{}/{}/IM_000{}.dcm".format(
            #     DIR_DICOM, self.patient, plan, sequence, imgnumber))

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

    def generateBSpline(self, curve_pts, degree):
        # Create a B-Spline curve instance
        curve = BSpline.Curve()

        # Set evaluation delta
        curve.delta = 0.001

        # Set control points
        controlpts = curve_pts
        curve.ctrlpts = controlpts

        # Set curve degree
        curve.degree = degree

        # Auto-generate knot vector
        curve.knotvector =\
            utils.generate_knot_vector(curve.degree, len(curve.ctrlpts))

        return curve, curve.knotvector

    def draw_curve(self, curve):
        # Try to load the visualization module
        try:
            render_curve = True
            from geomdl.visualization import VisMPL
        except ImportError:
            render_curve = False

        # Draw the control point polygon and the evaluated curve
        if render_curve:
            vis_comp = VisMPL.VisCurve3D()
            curve.vis = vis_comp
            curve.render()

    def draw_curves(self, curvespts):
        # Try to load the visualization module
        try:
            render_curve = True
            from geomdl.visualization import VisMPL
        except ImportError:
            render_curve = False

        # Plot the curves using the curve container
        curves = Multi.MultiCurve()
        curves.delta = 0.01

        for curve in curvespts:
            curves.add(curve)

        if render_curve:
            vis_comp = VisMPL.VisCurve3D()
            curves.vis = vis_comp
            curves.render()

    def interpolation(self, side, rootsequence, imgnumber, curves, missingsagcol, missingcorcol, missingcoronalsequences, missingsagittalsequences, save=0):
        # print("Missing sagittal columns: {} ({})".format(missingsagcol, len(missingsagcol)))
        # print("Missing coronal columns: {} ({})".format(missingcorcol, len(missingcorcol)))

        interpolate_points, interpolate = list(), list()

        if len(missingsagcol) > 0:
            # print(len(curves))  # = 60
            # print(len(curves[0].curvepts))  # = 1001
            # print(len(curves[0].curvepts[0]))  # = 3
            curve_xpts, curves_xpts = list(), list()
            for i in range(len(curves)):
                for j in range(len(curves[i].curvepts)):
                    curve_xpts.append(curves[i].curvepts[j][0])
                curves_xpts.append(curve_xpts)
                curve_xpts = []
            # print(len(curves_xpts))
            # print(len(curves_xpts[0]))

            # interpolate_points, interpolate = list(), list()

            for j in range(len(missingsagcol)):
                for i in range(len(curves_xpts)):
                    # nearest = min(curves_xpts[i], key=lambda x: abs(x - missingsagcol[j]))  # return only the value
                    # nearest = min(range(len(curves_xpts[i])), key=lambda x: abs(x[1] - missingsagcol[j]))  # return only the index
                    nearest = min(enumerate(curves_xpts[i]), key=lambda x: abs(x[1] - missingsagcol[j]))  # return index and value
                    point = curves[i].curvepts[nearest[0]]
                    interpolate_points.append(point)
                interpolate.append(interpolate_points)
                interpolate_points = []

            if save == 1:
                # Write points interpolated in txt file
                for i in range(len(missingsagittalsequences)):
                    self.write_points(
                        side=side,
                        step=2,
                        rootsequence=rootsequence,
                        sequence=missingsagittalsequences[i],
                        analyzedimage=imgnumber,
                        points=interpolate[i])

        if len(missingcorcol) > 0:
            curve_ypts, curves_ypts = list(), list()

            for i in range(len(curves)):
                for j in range(len(curves[i].curvepts)):
                    curve_ypts.append(curves[i].curvepts[j][1])
                curves_ypts.append(curve_ypts)
                curve_ypts = []

            # interpolate_points, interpolate = list(), list()

            for j in range(len(missingcorcol)):
                for i in range(len(curves_ypts)):
                    # nearest = min(curves_ypts[i], key=lambda x: abs(x - missingcorcol[j]))  # return only the value
                    # nearest = min(range(len(curves_ypts[i])), key=lambda x: abs(x[1] - missingcorcol[j]))  # return only the index
                    nearest = min(enumerate(curves_ypts[i]), key=lambda x: abs(x[1] - missingcorcol[j]))  # return index and value
                    point = curves[i].curvepts[nearest[0]]
                    interpolate_points.append(point)
                interpolate.append(interpolate_points)
                interpolate_points = []

            if save == 1:
                # Write points interpolated in txt file
                for i in range(len(missingcoronalsequences)):
                    self.write_points(
                        side=side,
                        step=3,
                        rootsequence=rootsequence,
                        sequence=missingcoronalsequences[i],
                        analyzedimage=imgnumber,
                        points=interpolate[i])

        return interpolate

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

        points = [tuple(l) for l in points]

        instant_information.append(points)

        if side == 0:
            file = open('{}/{}-{}.txt'.format(DIR_RESULT_INTERPOLATED_BSPLINE_LEFT, rootsequence, analyzedimage), 'a')
        else:
            file = open('{}/{}-{}.txt'.format(DIR_RESULT_INTERPOLATED_BSPLINE_RIGHT, rootsequence, analyzedimage), 'a')
        file.write("{}\n".format(instant_information))
        file.close()


def execute(patient, rootsequence, side, imgnumber, coronalsequences, lsagsequences, rsagsequences, show=0, save=0):
    # print("Coronal sequences: {} ({})".format(coronalsequences, len(coronalsequences)))
    # print("Sagittal sequences (Left): {} ({})".format(lsagsequences, len(lsagsequences)))
    # print("Sagittal sequences (Right): {} ({})\n".format(rsagsequences, len(rsagsequences)))

    interpolate = BSplineInterpolate(
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

    pts1, pts2, pts3, pts =\
        interpolate.get_points(
            dataset=current_dataset,
            option=1)

    lsagX, rsagX, corY, sagavailableX, coravailableY, sagcoldiff, corcoldiff =\
        interpolate.get_columns(
            points1=pts1,
            points2=pts2,
            points3=pts3)
    # print("Coronal columns: {} ({})".format(corY, len(corY)))
    # print("Sagittal columns (Left): {} ({})".format(lsagX, len(lsagX)))
    # print("Sagittal columns (Right): {} ({})\n".format(rsagX, len(rsagX)))
    # print("Missing sagittal columns: {} ({})".format(sagcoldiff, len(sagcoldiff)))
    # print("Missing coronal columns: {} ({})\n".format(corcoldiff, len(corcoldiff)))

    # print(lsagsequences[lsagX.index(sagcoldiff[0])])
    # print(coronalsequences[corY.index(corcoldiff[0])])
    # print(coronalsequences[corY.index(corcoldiff[1])])
    missingcorsequences, missingsagsequences = list(), list()
    for i in range(len(corcoldiff)):
        missingcorsequences.append(coronalsequences[corY.index(corcoldiff[i])])
    for i in range(len(sagcoldiff)):
        if side == 0:
            missingsagsequences.append(lsagsequences[lsagX.index(sagcoldiff[i])])
        else:
            missingsagsequences.append(rsagsequences[rsagX.index(sagcoldiff[i])])
    # print("Missing coronal sequence(s): {} ({})".format(missingcorsequences, len(missingcorsequences)))
    # print("Missing sagittal sequence(s): {} ({})".format(missingsagsequences, len(missingsagsequences)))

    firstcolsag, lastcolsag, firstcolcor, lastcolcor =\
        interpolate.check_extremes(
            side=side,
            allleftcolsag=lsagX,
            allrightcolsag=rsagX,
            allcolcor=corY,
            availablecolsag=sagavailableX,
            availablecolcor=coravailableY)

    interpolateptssag, interpolateptscor = list(), list()

    curvedegree = 3

    if len(sagcoldiff) > 0:
        if firstcolsag is True and lastcolsag is True:
            ctrlpts, curves = list(), list()
            for i in range(60):
                for j in range(len(pts2)):
                    ctrlpts.append(pts2[j][i])
                curves.append(ctrlpts)
                ctrlpts = []

            bsplinecurves, knotvectors = list(), list()
            for i in range(len(curves)):
                curve, knots = interpolate.generateBSpline(curves[i], curvedegree)
                bsplinecurves.append(curve)

            # Interpolation points
            interpolateptssag = interpolate.interpolation(
                side=side,
                rootsequence=rootsequence,
                imgnumber=imgnumber,
                curves=bsplinecurves,
                missingsagcol=sagcoldiff,
                missingcorcol=[],  # corcoldiff
                missingcoronalsequences=missingcorsequences,
                missingsagittalsequences=missingsagsequences,
                save=save)

            if show == 1:
                # Plot curves
                # lcurves = list()
                # for i in range(len(curves)):
                #     lcurves.append(bsplinecurves[i])
                # interpolate.draw_curves(lcurves)

                # Plot curves with same color
                lcurves = list()
                for i in range(len(curves)):
                    lcurves.append(bsplinecurves[i])

                bsctrlpts, bscurvepts = list(), list()
                for i in range(len(curves)):
                    bsctrlpts.append(np.array(bsplinecurves[i].ctrlpts))
                    bscurvepts.append(np.array(bsplinecurves[i].curvepts))
                # Draw the control points polygon, the 3D curve and the tangent vectors
                # fig = plt.figure(figsize=(10.67, 8), dpi=96)
                # ax = Axes3D(fig)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                # Plot 3D lines
                for i in range(len(curves)):
                    ax.plot(bsctrlpts[i][:, 0], bsctrlpts[i][:, 1], bsctrlpts[i][:, 2], color='black', linestyle='-.', marker='o', markersize=3)
                    ax.plot(bscurvepts[i][:, 0], bscurvepts[i][:, 1], bscurvepts[i][:, 2], color='green', linestyle='-')
                # Display the 3D plot
                # plt.show()

                # Method used to set the same scale for all axes
                # Stackoverflow:
                # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
                bsctrlpts = list(itertools.chain.from_iterable(bsctrlpts))
                xpts, ypts, zpts = list(), list(), list()
                for i in range(len(bsctrlpts)):
                    xpts.append(bsctrlpts[i][0])
                    ypts.append(bsctrlpts[i][1])
                    zpts.append(bsctrlpts[i][2])

                X = np.asarray(xpts)
                Y = np.asarray(ypts)
                Z = np.asarray(zpts)
                # # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
                Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
                Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
                Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

                # # Comment or uncomment following both lines to test the fake bounding box:
                for xb, yb, zb in zip(Xb, Yb, Zb):
                    ax.plot([xb], [yb], [zb], 'w')
                # Display the 3D plot
                plt.show()

                '''
                bsctrlpts = np.array(bsplinecurves[0].ctrlpts)
                bscurvepts = np.array(bsplinecurves[0].curvepts)
                # Draw the control points polygon, the 3D curve and the tangent vectors
                fig = plt.figure(figsize=(10.67, 8), dpi=96)
                ax = Axes3D(fig)
                # Plot 3D lines
                ax.plot(bsctrlpts[:, 0], bsctrlpts[:, 1], bsctrlpts[:, 2], color='black', linestyle='-.', marker='o')
                ax.plot(bscurvepts[:, 0], bscurvepts[:, 1], bscurvepts[:, 2], color='green', linestyle='-')
                # Display the 3D plot
                plt.show()
                '''

    if len(corcoldiff) > 0:
        if firstcolcor is True and lastcolcor is True:
            ctrlpts, curves = list(), list()
            # for i in range(60):
            #     ctrlpts.append(pts1[0][i])
            # curves.append(ctrlpts)
            # ctrlpts = []

            # for i in range(60):
            #     for j in range(len(pts3)):
            #         ctrlpts.append(pts3[j][i])
            #     curves.append(ctrlpts)
            #     ctrlpts = []

            pts1_pts3 = pts1 + pts3
            for i in range(60):
                for j in range(len(pts1_pts3)):
                    ctrlpts.append(pts1_pts3[j][i])
                curves.append(ctrlpts)
                ctrlpts = []

            bsplinecurves, knotvectors = list(), list()
            for i in range(len(curves)):
                curve, knots = interpolate.generateBSpline(curves[i], curvedegree)
                bsplinecurves.append(curve)

            # Interpolation points
            interpolateptscor = interpolate.interpolation(
                side=side,
                rootsequence=rootsequence,
                imgnumber=imgnumber,
                curves=bsplinecurves,
                missingsagcol=[],  # sagcoldiff
                missingcorcol=corcoldiff,
                missingcoronalsequences=missingcorsequences,
                missingsagittalsequences=missingsagsequences,
                save=save)

        # Plot curves
        # lcurves = list()
        # for i in range(len(curves)):
        #     lcurves.append(bsplinecurves[i])
        # interpolate.draw_curves(lcurves)

    interpolatepts = interpolateptssag + interpolateptscor

    if show == 1:
        pts1 = list(itertools.chain.from_iterable(pts1))
        pts2 = list(itertools.chain.from_iterable(pts2))
        pts3 = list(itertools.chain.from_iterable(pts3))

        pts.append(pts1)
        pts.append(pts2)
        pts.append(pts3)
        pts = list(itertools.chain.from_iterable(pts))

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

    '''
    # curve1 = [(5, 10, 5), (15, 25, 5), (30, 30, 5), (45, 5, 5), (55, 5, 5)]
    curve1 = [[5, 10, 5], [15, 25, 5], [30, 30, 5], [45, 5, 5], [55, 5, 5]]
    curve2 = [[70, 40, 50], [60, 60, 50], [50, 50, 50], [35, 60, 50], [20, 40, 50]]
    curvedegree = 3

    bsplinecurve1, knotscurve1 = interpolate.generateBSpline(curve1, curvedegree)
    bsplinecurve2, knotscurve2 = interpolate.generateBSpline(curve2, curvedegree)

    # interpolate.draw_curve(bsplinecurve1)
    # interpolate.draw_curve(bsplinecurve2)

    lcurves = list()
    lcurves.append(bsplinecurve1)
    lcurves.append(bsplinecurve2)
    interpolate.draw_curves(lcurves)
    '''


if __name__ == '__main__':
    try:
        patient = 'Iwasawa'
        rootsequence = 9
        side = 0  # 0 - left | 1 - right | 2 - Both
        imgnumber = 2  # Instant
        coronalsequences = '4, 5, 6, 7, 8, 9, 10, 11, 12'
        leftsagittalsequences = '1, 2, 3, 4, 5, 6, 7, 8'
        rightsagittalsequences = '12, 13, 14, 15, 16, 17'
        show = 1
        save = 0  # 0 - Not save points in txt file | 1 - Save

        txtargv2 = '-patient={}'.format(patient)
        txtargv3 = '-rootsequence={}'.format(rootsequence)
        txtargv4 = '-side={}'.format(side)
        txtargv5 = '-imgnumber={}'.format(imgnumber)
        txtargv6 = '-coronalsequences={}'.format(coronalsequences)
        txtargv7 = '-leftsagittalsequences={}'.format(leftsagittalsequences)
        txtargv8 = '-rightsagittalsequences={}'.format(rightsagittalsequences)
        txtargv9 = '-save={}'.format(save)
        txtargv10 = '-show={}'.format(show)

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
                                        if len(sys.argv) > 9:
                                            txtargv10 = sys.argv[9]

        txtargv = '{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(
            txtargv2,
            txtargv3,
            txtargv4,
            txtargv5,
            txtargv6,
            txtargv7,
            txtargv8,
            txtargv9,
            txtargv10)

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

        if txtargv.find('-show') != -1:
            txttmp = txtargv.split('-show')[1]
            txttmp = txttmp.split('=')[1]
            show = int(txttmp.split('|')[0])

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
        show=show,
        save=save)
