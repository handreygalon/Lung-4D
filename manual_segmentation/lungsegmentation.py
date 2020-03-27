#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
import os

from geomdl import NURBS
from geomdl import utilities as utils

""" This file create a NURBS curve around the lung """


def generateNURBS(curve_pts, degree, weights=None):
    # Create a NURBS curve instance
    curve = NURBS.Curve()

    # Set evaluation delta
    curve.delta = 0.01

    # Set curve degree
    curve.degree = degree

    if heart == 0:
        # Set weights - obs. the 10th and 20th points are setted to 100 because is the down corner of the lung
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        # Set weights - obs. the 5yh, 10th are setted to 100 because is in the heart region
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 100.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    ctrlptsw = combine_ctrlpts_weights(curve_pts, weights)

    # Set control points
    curve.ctrlpts = ctrlptsw

    # Auto-generate knot vector
    curve.knotvector = utils.generate_knot_vector(curve.degree, len(curve.ctrlpts))

    return curve, curve.knotvector


def combine_ctrlpts_weights(ctrlpts, weights=None):
    if weights is None:
        weights = [1.0 for _ in range(len(ctrlpts))]
        # weights = [1] * (len(curve_pts))

    ctrlptsw = []
    for pt, w in zip(ctrlpts, weights):
        temp = [float(c * w) for c in pt]
        temp.append(float(w))
        ctrlptsw.append(temp)

    return ctrlptsw


def draw_curve(curve):
    # Try to load the visualization module
    try:
        render_curve = True
        from geomdl.visualization import VisMPL
    except ImportError:
        render_curve = False

    # Draw the control point polygon and the evaluated curve
    if render_curve:
        vis_comp = VisMPL.VisCurve2D()
        curve.vis = vis_comp
        curve.render()


def update_curve(curveinstance, ctrlpts, weights):
    ctrlptsw = combine_ctrlpts_weights(ctrlpts, weights)
    curveinstance.ctrlpts = ctrlptsw

    return ctrlptsw


def C0(curveinstance, ctrlpts, degree, weights):
    ctrlpts[-1] = ctrlpts[0]
    update_curve(curveinstance, ctrlpts, weights)

    return ctrlpts


def create_curve(ctrlpts):
    # Set weighted control points
    curve = list()
    for points in ctrlpts:
        curve.append(list(points))

    # Invert y coordinate of the control points because origin in openCV is up left corner of the image
    controlpts = curve
    convert_controlpts = map(lambda x: 255 - x[1], controlpts)
    for i in range(len(controlpts)):
        controlpts[i][1] = convert_controlpts[i]

    # weights = [1.0] * len(ctrlpts)
    weights = None

    curvedegree = 3

    nurbscurve, knotscurve = generateNURBS(controlpts, curvedegree, weights)

    # draw_curve(nurbscurve)

    if optmode != 0:
        C0(nurbscurve, controlpts, curvedegree, weights)

    # Invert y coordinate of the curve points because origin in openCV is up left corner of the image
    convert_curvepts = map(lambda x: 255 - x[1], nurbscurve.curvepts)
    for i in range(len(nurbscurve.curvepts)):
        nurbscurve.curvepts[i][1] = convert_curvepts[i]

    curve = list()
    for point in nurbscurve.curvepts:
        point = map(int, point)
        curve.append(tuple(point))
    curvepts = curve

    create_contour(points=curvepts)


def get_points(image):
    # Set up data to send to mouse handler
    data = {}
    data['image'] = image.copy()
    data['lines'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    points = data['lines']

    return points, data['image']


def create_contour(points):
    mask = np.zeros((256, 256, 3), np.uint8)
    contour = cv2.imread('{}/IM ({}).png'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)

    if optmode == 0:
        for i in range(len(points) - 1):
            cv2.line(
                mask, points[i], points[i + 1], (0, 0, 255), 1)
            cv2.line(
                contour, points[i], points[i + 1], (0, 0, 255), 1)
    else:
        for i in range(len(points) - 1):
            if i == len(points) - 1:
                i = 0
                cv2.line(
                    mask, points[i], points[len(points) - 1], (0, 0, 255), 1)
                cv2.line(
                    contour, points[i], points[len(points) - 1], (0, 0, 255), 1)
            cv2.line(
                mask, points[i], points[i + 1], (0, 0, 255), 1)
            cv2.line(
                contour, points[i], points[i + 1], (0, 0, 255), 1)

    if show == 1:
        cv2.imshow('Segmentation', contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save == 1:
        cv2.imwrite('{}/maskIM ({}).png'.format(DIR_dest, imgnumber), mask)
        cv2.imwrite('{}/segIM ({}).png'.format(DIR_dest, imgnumber), contour)


def read_points():
    """ Converter list of points (from a txt file) to int """

    dataset = open('{}/points.txt'.format(DIR_dest), 'r').read().split('\n')
    dataset.pop(-1)  # last element is empty

    all_points = []
    ltuplas = []

    for i in range(len(dataset)):
        lpts = dataset[i].replace('), ', ');')[1:-1].split(';')

        for j in range(len(lpts)):
            pts = lpts[j].split(',')
            tupla = (int(pts[0][1:]), int(pts[1][1:-1]))
            ltuplas.append(tupla)

        all_points.append(ltuplas)
        ltuplas = []

    return all_points


def rebuild_contour():
    # pts = read_points()[-1]
    # pts = [(128, 87), (119, 93), (113, 102), (108, 113), (105, 127), (104, 139), (103, 151), (101, 163), (100, 177), (97, 200), (102, 195), (108, 191), (114, 186), (120, 183), (127, 181), (134, 180), (142, 181), (148, 183), (156, 187), (162, 190), (161, 181), (157, 167), (155, 158), (154, 147), (152, 135), (150, 121), (149, 110), (147, 100), (142, 92), (136, 88)]
    pts = [(126, 56), (115, 67), (108, 77), (100, 86), (90, 101), (82, 116), (75, 134), (76, 158), (83, 144), (93, 130), (104, 122), (119, 125), (130, 137), (134, 154), (131, 169), (124, 180), (135, 182), (145, 185), (154, 190), (163, 196), (164, 182), (164, 166), (166, 148), (165, 131), (163, 114), (161, 103), (158, 84), (151, 70), (143, 59), (134, 55)]

    create_curve(pts)

    if save == 1:
        file = open('{}/points.txt'.format(DIR_dest), 'a')
        file.write("{}\n".format(pts))
        file.close()


def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        data['lines'].append((x, y))  # prepend the point

        print('Point {}: {}'.format(len(data['lines']), data['lines'][-1]))

        cv2.circle(data['image'], (x, y), 2, (0, 0, 255), -1)
        cv2.imshow("Image", data['image'])

        if len(data['lines']) == npoints:
            print("Finish segmentation! Please press ESC")
            cv2.imshow("Image", data['image'])

            if save == 1:
                file = open('{}/points.txt'.format(DIR_dest), 'a')
                file.write("{}\n".format(data['lines']))
                file.close()

            cv2.waitKey(0)
            cv2.destroyAllWindows()

            create_curve(data['lines'])
    elif event == cv2.EVENT_MOUSEMOVE and len(data['lines']) >= 1:
        # print('Position: ({}, {})'.format(x, y))
        image = data['image'].copy()

        if len(data['lines']) < npoints:
            # this is just for a line visualization
            cv2.line(image, data['lines'][len(data['lines']) - 1], (x, y), (0, 255, 0), 1)

        cv2.imshow("Image", image)

    elif event == cv2.EVENT_RBUTTONDOWN:
        rebuild_contour()


if __name__ == '__main__':

    try:
        optmode = 0  # 0 - diaphragm | 1 - lung
        opthickness = 1
        patient = 'Iwasawa'
        plan = 'Sagittal'
        sequence = 1
        side = 0  # 0 - left | 1 - right
        imgnumber = 1
        show = 1  # 0 - Not show | 1 - Show
        save = 0  # 0 - Not save | 1 - Save
        heart = 0  # 0 - Heart not appears | 1 - Heart appears
        txtargv2 = '-mode={}'.format(optmode)
        txtargv3 = '-thickness={}'.format(opthickness)
        txtargv4 = '-patient={}'.format(patient)
        txtargv5 = '-plan={}'.format(plan)
        txtargv6 = '-sequence={}'.format(sequence)
        txtargv7 = '-side={}'.format(side)
        txtargv8 = '-imgnumber={}'.format(imgnumber)
        txtargv9 = '-show={}'.format(show)
        txtargv10 = '-save={}'.format(save)
        txtargv11 = '-heart={}'.format(heart)

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
                                            if len(sys.argv) > 10:
                                                txtargv11 = sys.argv[10]

        txtargv = '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}'\
            .format(txtargv2, txtargv3, txtargv4, txtargv5, txtargv6, txtargv7, txtargv8, txtargv9, txtargv10, txtargv11)

        if txtargv.find('-mode') != -1:
            txttmp = txtargv.split('-mode')[1]
            txttmp = txttmp.split('=')[1]
            optmode = int(txttmp.split('|')[0])

        if txtargv.find('-thickness') != -1:
            txttmp = txtargv.split('-thickness')[1]
            txttmp = txttmp.split('=')[1]
            opthickness = int(txttmp.split('|')[0])

        if txtargv.find('-patient') != -1:
            txttmp = txtargv.split('-patient')[1]
            txttmp = txttmp.split('=')[1]
            patient = txttmp.split('|')[0]

        if txtargv.find('-plan') != -1:
            txttmp = txtargv.split('-plan')[1]
            txttmp = txttmp.split('=')[1]
            plan = txttmp.split('|')[0]

        if txtargv.find('-sequence') != -1:
            txttmp = txtargv.split('-sequence')[1]
            txttmp = txttmp.split('=')[1]
            sequence = int(txttmp.split('|')[0])

        if txtargv.find('-side') != -1:
            txttmp = txtargv.split('-side')[1]
            txttmp = txttmp.split('=')[1]
            side = int(txttmp.split('|')[0])

        if txtargv.find('-imgnumber') != -1:
            txttmp = txtargv.split('-imgnumber')[1]
            txttmp = txttmp.split('=')[1]
            imgnumber = int(txttmp.split('|')[0])

        if txtargv.find('-show') != -1:
            txttmp = txtargv.split('-show')[1]
            txttmp = txttmp.split('=')[1]
            show = int(txttmp.split('|')[0])

        if txtargv.find('-save') != -1:
            txttmp = txtargv.split('-save')[1]
            txttmp = txttmp.split('=')[1]
            save = int(txttmp.split('|')[0])

        if txtargv.find('-heart') != -1:
            txttmp = txtargv.split('-heart')[1]
            txttmp = txttmp.split('=')[1]
            heart = int(txttmp.split('|')[0])

    except ValueError:
        print '''
        Examples of use:

        Sagittal:
        $ python {} -mode=0 -thickness=1 -patient=Iwasawa -plan=Sagittal -sequence=1 -side=0 -imgnumber=1

        Coronal:
        $ python {} -mode=0 -thickness=1 -patient=Iwasawa -plan=Coronal -sequence=1 -side=1 -imgnumber=1

        Parameters:

        patient = Iwasawa -> Patient's name
        plan = Sagittal
        sequence = 1
        side = 0 -> Use only to Coronal plan (0 to left lung and 1 to right lung)
        imgnumber = 1 -> image number
        mode = 0 -> Segments only the diaphragmatic region; 1 -> Segments entire lung
        thickness = 1 -> thickness of the contour'''.format(sys.argv[0], sys.argv[0])
        exit()

    if optmode == 0:
        npoints = 10
    else:
        npoints = 30

    DIR_root = os.path.expanduser("~/Documents/UDESC")
    DIR_orig = '{}/tests/segmentation/{}/pre_processed/{}/{}'\
        .format(DIR_root, patient, plan, sequence)
    # DIR_orig = '{}/tests/segmentation/{}/original/{}/{}'\
    #     .format(DIR_root, patient, plan, sequence)

    if plan == 'Sagittal':
        if optmode == 0:
            DIR_dest = '{}/segmented/diaphragm/{}/{}/{}'\
                .format(DIR_root, patient, plan, sequence)
        else:
            DIR_dest = '{}/segmented/lung/{}/{}/{}'\
                .format(DIR_root, patient, plan, sequence)
    else:
        if optmode == 0:
            if side == 0:
                DIR_dest = '{}/segmented/diaphragm/{}/{}/{}_L'\
                    .format(DIR_root, patient, plan, sequence)
            else:
                DIR_dest = '{}/segmented/diaphragm/{}/{}/{}_R'\
                    .format(DIR_root, patient, plan, sequence)
        else:
            if side == 0:
                DIR_dest = '{}/segmented/lung/{}/{}/{}_L'\
                    .format(DIR_root, patient, plan, sequence)
            else:
                DIR_dest = '{}/segmented/lung/{}/{}/{}_R'\
                    .format(DIR_root, patient, plan, sequence)

    img = cv2.imread('{}/IM ({}).png'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)
    # img = cv2.imread('{}/IM ({}).jpg'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)
    pts, final_image = get_points(img)
    # cv2.imshow('Image', final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
