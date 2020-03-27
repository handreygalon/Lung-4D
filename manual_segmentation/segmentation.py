#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import sys
# from operator import itemgetter

DIR_ROOT = os.path.expanduser("~/Documents/UDESC")
DIR_JPG = os.path.expanduser("~/Documents/UDESC/JPG")
DIR_MAN_DIAHPRAGM_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/diaphragm")
DIR_MAN_LUNG_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/lung/Manual")


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


def draw_contour(img_mask, img_contour, points, mode, thickness):
    if mode == 0:
        for i in range(len(points) - 1):
            cv2.line(
                img_mask, points[i], points[i + 1], (0, 0, 255), thickness)
            cv2.line(
                img_contour, points[i], points[i + 1], (0, 0, 255), thickness)
    else:
        for i in range(len(points)):  # for i in range(len(points) - 1):
            if i == len(points) - 1:
                i = 0
                cv2.line(
                    img_mask, points[i], points[len(points) - 1], (0, 0, 255), thickness)
                cv2.line(
                    img_contour, points[i], points[len(points) - 1], (0, 0, 255), thickness)

            cv2.line(
                img_mask, points[i], points[i + 1], (0, 0, 255), thickness)
            cv2.line(
                img_contour, points[i], points[i + 1], (0, 0, 255), thickness)

    cv2.imwrite('{}/maskIM ({}).png'.format(DIR_dest, imgnumber), img_mask)
    cv2.imwrite('{}/segIM ({}).png'.format(DIR_dest, imgnumber), img_contour)


def create_contour(points):
    mask = np.zeros((256, 256, 3), np.uint8)
    # contour = cv2.imread('{}/IM ({}).png'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)
    contour = cv2.imread('{}/IM ({}).jpg'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)

    if optmode == 0:
        # draw_contour(mask, contour, points, mode=0, thickness=1)
        for i in range(len(points) - 1):
            cv2.line(
                mask, points[i], points[i + 1], (0, 0, 255), 1)
            cv2.line(
                contour, points[i], points[i + 1], (0, 0, 255), 1)
    else:
        # draw_contour(mask, contour, points, mode=1, thickness=1)
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


def rebuild_contour(mode=0):
    # pts = read_points()[imgnumber - 1]
    # pts = [(144, 193), (147, 186), (152, 179), (157, 174), (167, 171), (173, 171), (180, 172), (186, 175), (190, 178), (194, 182)]
    pts = [(144, 199), (148, 194), (153, 189), (158, 185), (167, 183), (176, 184), (181, 186), (186, 189), (190, 191), (194, 194)]
    # newy = list(map(lambda y: y[1] + 1, pts))
    newy = list(map(lambda y: y[1] - 4, pts))
    for i in range(len(pts)):
        pts[i] = (pts[i][0], newy[i])
    print(pts)

    mask = np.zeros((256, 256, 3), np.uint8)
    # contour = cv2.imread('{}/IM ({}).png'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)
    contour = cv2.imread('{}/IM ({}).jpg'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)

    thickness = 1
    if optmode == 0:
        for i in range(len(pts) - 1):
            cv2.line(
                mask, pts[i], pts[i + 1], (0, 0, 255), thickness)
            cv2.line(
                contour, pts[i], pts[i + 1], (0, 0, 255), thickness)
    else:
        for i in range(len(pts)):
            if i == len(pts) - 1:
                i = 0
                cv2.line(
                    mask, pts[i], pts[len(pts) - 1], (0, 0, 255), thickness)
                cv2.line(
                    contour, pts[i], pts[len(pts) - 1], (0, 0, 255), thickness)
            cv2.line(
                mask, pts[i], pts[i + 1], (0, 0, 255), thickness)
            cv2.line(
                contour, pts[i], pts[i + 1], (0, 0, 255), thickness)

    if show == 1:
        cv2.imshow('Segmentation', contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save == 1:
        # cv2.imwrite('maskIM ({}).png'.format(imgnumber), mask)
        cv2.imwrite('{}/maskIM ({}).png'.format(DIR_dest, imgnumber), mask)
        # cv2.imwrite('segIM ({}).png'.format(imgnumber), contour)
        cv2.imwrite('{}/segIM ({}).png'.format(DIR_dest, imgnumber), contour)

        # file = open('points.txt', 'a')
        file = open('{}/points.txt'.format(DIR_dest), 'a')
        file.write("{}\n".format(pts))
        file.close()


def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        data['lines'].append((x, y))  # prepend the point
        # cv2.circle(data['image'], (x, y), 3, (0, 0, 255), 5)

        # print('Add point: ', data['lines'])
        print('Point {}: {}'.format(len(data['lines']), data['lines'][-1]))

        if len(data['lines']) >= 2:
            cv2.line(
                data['image'],
                data['lines'][len(data['lines']) - 2],
                data['lines'][len(data['lines']) - 1],
                (0, 0, 255),
                1)
            cv2.imshow("Image", data['image'])
        # else:
        #     cv2.circle(data['image'], (x, y), 3, (0, 0, 255), 5)
        #     cv2.imshow("Image", data['image'])

        if len(data['lines']) == npoints:
            print("Finish segmentation! Please press ESC")
            if optmode != 0:
                cv2.line(
                    data['image'],
                    data['lines'][0],
                    data['lines'][len(data['lines']) - 1],
                    (0, 0, 255),
                    1)
            cv2.imshow("Image", data['image'])

            create_contour(data['lines'])

            file = open('{}/points.txt'.format(DIR_dest), 'a')
            file.write("{}\n".format(data['lines']))
            file.close()

            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
        optmode = 1  # 0 - diaphragm | 1 - lung
        opthickness = 1
        patient = 'Matsushita'
        plan = 'Sagittal'
        sequence = 2
        side = 0  # 0 - left | 1 - right
        imgnumber = 1
        save = 0  # 0 - Do not save | 1 - Save
        show = 0  # 0 - Do not show | 1 - Show
        txtargv2 = '-mode={}'.format(optmode)
        txtargv3 = '-thickness={}'.format(opthickness)
        txtargv4 = '-patient={}'.format(patient)
        txtargv5 = '-plan={}'.format(plan)
        txtargv6 = '-sequence={}'.format(sequence)
        txtargv7 = '-side={}'.format(side)
        txtargv8 = '-imgnumber={}'.format(imgnumber)
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

        txtargv = '{}|{}|{}|{}|{}|{}|{}|{}|{}'\
            .format(txtargv2, txtargv3, txtargv4, txtargv5, txtargv6, txtargv7, txtargv8, txtargv9, txtargv10)
        print(txtargv)

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

        if txtargv.find('-save') != -1:
            txttmp = txtargv.split('-save')[1]
            txttmp = txttmp.split('=')[1]
            save = int(txttmp.split('|')[0])

        if txtargv.find('-show') != -1:
            txttmp = txtargv.split('-show')[1]
            txttmp = txttmp.split('=')[1]
            show = int(txttmp.split('|')[0])

    except ValueError:
        print("""
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
        thickness = 1 -> thickness of the contour""".format(sys.argv[0], sys.argv[0]))
        exit()

    if optmode == 0:
        npoints = 10
    else:
        npoints = 60

    # DIR_orig = '{}/tests/segmentation/{}/pre_processed/{}/{}'.format(DIR_ROOT, patient, plan, sequence)
    # DIR_orig = '{}/tests/segmentation/{}/original/{}/{}'.format(DIR_ROOT, patient, plan, sequence)
    DIR_orig = '{}/{}/{}/{}'.format(DIR_JPG, patient, plan, sequence)

    if plan == 'Sagittal':
        if optmode == 0:
            DIR_dest = '{}/{}/{}/{}'\
                .format(DIR_MAN_DIAHPRAGM_MASKS, patient, plan, sequence)
        else:
            DIR_dest = '{}/{}/{}/{}'\
                .format(DIR_MAN_LUNG_MASKS, patient, plan, sequence)
    else:
        if optmode == 0:
            if side == 0:
                DIR_dest = '{}/{}/{}/{}_L'\
                    .format(DIR_MAN_DIAHPRAGM_MASKS, patient, plan, sequence)
            else:
                DIR_dest = '{}/{}/{}/{}_R'\
                    .format(DIR_MAN_DIAHPRAGM_MASKS, patient, plan, sequence)
        else:
            if side == 0:
                DIR_dest = '{}/{}/{}/{}_L'\
                    .format(DIR_MAN_LUNG_MASKS, patient, plan, sequence)
            else:
                DIR_dest = '{}/{}/{}/{}_R'\
                    .format(DIR_MAN_LUNG_MASKS, patient, plan, sequence)

    # img = cv2.imread('{}/IM ({}).png'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)
    img = cv2.imread('{}/IM ({}).jpg'.format(DIR_orig, imgnumber), cv2.IMREAD_COLOR)
    pts, final_image = get_points(img)
    # cv2.imshow('Image', final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
