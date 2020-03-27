#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import itertools

from geomdl import NURBS
from geomdl import utilities as utils
from geomdl import Multi

from util.constant import *


class NURBSInterpolate(object):
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

    def get_information_by_step(self, dataset, step):
        sequence, imagenumber = list(), list()
        for i in range(len(dataset)):
            if dataset[i][4] == step:
                sequence.append(dataset[i][2])
                imagenumber.append(dataset[i][3])

        return sequence, imagenumber

    def generateNURBS(self, curve_pts, degree, weights=None):
        # Create a NURBS curve instance
        curve = NURBS.Curve()

        # Set evaluation delta
        curve.delta = 0.01

        # Set curve degree
        curve.degree = degree

        # Set weights
        # weights = [1.0, 10.0, 1.0, 1.0, 1.0, 1.0]
        if weights is None:
            weights = [1] * (len(curve_pts))

        ctrlptsw = self.combine_ctrlpts_weights(curve_pts, weights)

        # Set control points
        # curve.ctrlpts = [[5.0, 5.0, 1.0], [100.0, 100.0, 10.0], [10.0, 15.0, 1.0], [15.0, 15.0, 1.0], [15.0, 10.0, 1.0], [10.0, 5.0, 1.0]]
        curve.ctrlpts = ctrlptsw

        # Auto-generate knot vector
        curve.knotvector = utils.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        # Set knot vector
        # curve.knotvector = [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]

        # return curve, curve.knotvector
        return curve

    def combine_ctrlpts_weights(self, ctrlpts, weights=None):
        if weights is None:
            weights = [1.0 for _ in range(len(ctrlpts))]

        ctrlptsw = []
        for pt, w in zip(ctrlpts, weights):
            # print("pt: {} | w: {}\n".format(pt, w))
            temp = [float(c * w) for c in pt]
            # print("temp: {}\n".format(temp))
            temp.append(float(w))
            ctrlptsw.append(temp)

        return ctrlptsw

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

    def write_points(self, patient, side, step, rootsequence, sequence, instant, imgnumber, points):
        # Saves register information about an respiration instant in a .txt file
        instant_information = list()

        instant_information.append(patient)

        if step == 2:
            instant_information.append('Sagittal')
        else:
            instant_information.append('Coronal')

        instant_information.append(sequence)
        instant_information.append(imgnumber)
        instant_information.append(step)

        points = [tuple(l) for l in points]

        instant_information.append(points)

        if side == 0:
            file = open('{}/{}-{}.txt'.format(DIR_RESULT_INTERPOLATED_NURBS_LEFT, rootsequence, instant), 'a')
        else:
            file = open('{}/{}-{}.txt'.format(DIR_RESULT_INTERPOLATED_NURBS_RIGHT, rootsequence, instant), 'a')
        file.write("{}\n".format(instant_information))
        file.close()

    def midpoint(self, p1, p2):
        x1 = p1[0]
        y1 = p1[1]
        z1 = p1[2]

        x2 = p2[0]
        y2 = p2[1]
        z2 = p2[2]

        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        zm = (z1 + z2) / 2

        mpt = (xm, ym, zm)

        return mpt

    def distance(self, p1, p2):
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
        return distance

    def point_interpolation(self, points):
        # print(points, len(points), len(points[0]))
        curvedegree = 3
        weights1 = [1] * 25
        weights2 = [10] * 10
        weights3 = [1] * 25
        weights = weights1 + weights2 + weights3
        controlpoints = points
        nurbscurve = self.generateNURBS(controlpoints, curvedegree, weights)

        interpolated_points = list()
        for i in range(len(controlpoints) - 1):
            mpt = self.midpoint(controlpoints[i], controlpoints[i + 1])
            nearest = min(nurbscurve.curvepts, key=lambda x: self.distance(x, mpt))
            interpolated_points.append(nearest)
        # print(len(interpolated_points))  # = 59

        allpoints = points + interpolated_points
        # weights = [1] * len(allpoints)
        # curve = self.generateNURBS(allpoints, curvedegree, weights)
        # self.draw_curve(curve)

        return allpoints


def execute(patient, rootsequence, side, imgnumber, coronalsequences, lsagsequences, rsagsequences, show=0, save=0):
    interpolate = NURBSInterpolate(
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
    # print(current_dataset[0][0])  # Patient
    # print(current_dataset[0][1])  # Plan
    # print(current_dataset[0][2])  # Sequence
    # print(current_dataset[0][3])  # Image
    # print(current_dataset[0][4])  # Step
    # print(current_dataset[0][5])  # Points

    pts1, pts2, pts3, pts =\
        interpolate.get_points(
            dataset=current_dataset,
            option=1)

    # Create file with the original points and interpolated points (Step 1)
    sequences, imagesnumbers =\
        interpolate.get_information_by_step(
            dataset=current_dataset,
            step=1)
    for i in range(len(pts1)):
        interpolated_points_step1 = interpolate.point_interpolation(pts1[i])
        if save == 1:
            interpolate.write_points(
                patient=patient,
                side=side,
                step=1,
                rootsequence=rootsequence,
                sequence=sequences[i],
                instant=imgnumber,
                imgnumber=imagesnumbers[i],
                points=interpolated_points_step1)

    # Create file with the original points and interpolated points (Step 2)
    sequences, imagesnumbers =\
        interpolate.get_information_by_step(
            dataset=current_dataset,
            step=2)
    for i in range(len(pts2)):
        interpolated_points_step2 = interpolate.point_interpolation(pts2[i])
        if save == 1:
            interpolate.write_points(
                patient=patient,
                side=side,
                step=2,
                rootsequence=rootsequence,
                sequence=sequences[i],
                instant=imgnumber,
                imgnumber=imagesnumbers[i],
                points=interpolated_points_step2)

    # Create file with the original points and interpolated points (Step 3)
    sequences, imagesnumbers =\
        interpolate.get_information_by_step(
            dataset=current_dataset,
            step=3)
    for i in range(len(pts3)):
        interpolated_points_step3 = interpolate.point_interpolation(pts3[i])
        if save == 1:
            interpolate.write_points(
                patient=patient,
                side=side,
                step=3,
                rootsequence=rootsequence,
                sequence=sequences[i],
                instant=imgnumber,
                imgnumber=imagesnumbers[i],
                points=interpolated_points_step3)

    '''
    # Set weighted control points
    curve1 = [[5.0, 5.0], [5.0, 10.0], [10.0, 10.0], [15.0, 15.0], [15.0, 20.0], [10.0, 25.0]]
    curve2 = [[15.0, 25.0], [20.0, 25.0], [20.0, 20.0], [20.0, 15.0], [25.0, 10.0], [25.0, 5.0]]

    weights1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    weights2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    curvedegree = 3

    nurbscurve1 = interpolate.generateNURBS(curve1, curvedegree, weights1)
    nurbscurve2 = interpolate.generateNURBS(curve2, curvedegree, weights2)

    # interpolate.draw_curve(nurbscurve1)
    # interpolate.draw_curve(nurbscurve2)

    lcurves = list()
    lcurves.append(nurbscurve1)
    lcurves.append(nurbscurve2)
    interpolate.draw_curves(lcurves)
    '''


if __name__ == '__main__':
    try:
        patient = 'Iwasawa'
        rootsequence = 9
        side = 0  # 0 - left | 1 - right | 2 - Both
        imgnumber = 1  # Instant
        coronalsequences = '4, 5, 6, 7, 8, 9, 10, 11, 12'
        leftsagittalsequences = '1, 2, 3, 4, 5, 6, 7, 8'
        rightsagittalsequences = '12, 13, 14, 15, 16, 17'
        show = 0
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
