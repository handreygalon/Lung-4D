#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
# import matplotlib.pyplot as plt
import os

DIR_JPG = os.path.expanduser("~/Documents/UDESC/JPG")
DIR_MAN_DIAHPRAGM_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/diaphragm")
DIR_MAN_LUNG_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/lung/Manual")


class Create2DST(object):

    def __init__(self, patient, plan, sequence):
        self.patient = patient
        self.plan = plan
        self.sequence = sequence

    def create3DMatrix(self, plan, sequence, mask=0):
        img_num = 1
        l_npy = []

        if mask == 1:
            for arq in range(50):
                img =\
                    cv2.imread('{}/{}/{}/{}/maskIM ({}).png'
                               .format(
                                   DIR_MAN_LUNG_MASKS,
                                   self.patient,
                                   plan,
                                   sequence,
                                   img_num), 0)

                l_npy.append(img)

                img_num = img_num + 1
        else:
            for arq in range(50):
                img =\
                    cv2.imread('{}/{}/{}/{}/IM ({}).jpg'
                               .format(
                                   DIR_JPG,
                                   self.patient,
                                   plan,
                                   sequence,
                                   img_num), 0)

                l_npy.append(img)

                img_num = img_num + 1

        data = np.stack(l_npy, axis=0)

        return data

    def createOrthogonalPlane(self, data, column):
        slice2D = data[:, :, column]
        img2DST = np.swapaxes(slice2D, 0, 1)

        return img2DST


if __name__ == '__main__':
    try:
        patient = 'Iwasawa'
        plan = 'Sagittal'
        sequence = 5
        mask = 0  # 0 - Does not use masks | 1 - Use masks
        # Left Coronal columns: [70, 76, 82, 88, 94, 101, 107, 113] (8)
        # Right Coronal columns: [161, 168, 174, 180, 186, 192] (6)
        # Sagittal columns: [101, 107, 114, 121, 128, 134, 141, 148, 154] (9)
        # columns = '70, 76, 82, 88, 94, 101, 107, 113'  # Coronal Left
        # columns = '161, 168, 174, 180, 186, 192'  # Coronal Right
        # columns = '101, 107, 114, 121, 128, 134, 141, 148, 154'  # Sagittal
        columns = '101, 114, 128, 141, 154'  # Sagittal
        show = 1
        save = 1

        txtargv2 = '-patient={}'.format(patient)
        txtargv3 = '-plan={}'.format(plan)
        txtargv4 = '-sequence={}'.format(sequence)
        txtargv5 = '-mask={}'.format(mask)
        txtargv6 = '-columns={}'.format(columns)
        txtargv7 = '-show={}'.format(show)
        txtargv8 = '-save={}'.format(save)

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

        txtargv = '{}|{}|{}|{}|{}|{}|{}'.format(
            txtargv2,
            txtargv3,
            txtargv4,
            txtargv5,
            txtargv6,
            txtargv7,
            txtargv8)

        print(txtargv)
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

        if txtargv.find('-mask') != -1:
            txttmp = txtargv.split('-mask')[1]
            txttmp = txttmp.split('=')[1]
            mask = int(txttmp.split('|')[0])

        if txtargv.find('-columns') != -1:
            txttmp = txtargv.split('-columns')[1]
            txttmp = txttmp.split('=')[1]
            txttmp = txttmp.split('|')[0]
            txttmp = txttmp.split(',')
            if len(columns) > 0:
                columns = [int(x) for x in txttmp]
            else:
                columns = []

        if txtargv.find('-show') != -1:
            txttmp = txtargv.split('-show')[1]
            txttmp = txttmp.split('=')[1]
            show = int(txttmp.split('|')[0])

        if txtargv.find('-save') != -1:
            txttmp = txtargv.split('-save')[1]
            txttmp = txttmp.split('=')[1]
            save = int(txttmp.split('|')[0])

    except ValueError:
        print("""
        Examples of use:

        $ python {} -patient=Iwasawa -plan=Sagittal -sequence=1 -mask=0 -column=125 -show=1 -save=0

        Parameters:
        mask = 0 -> create file using original DICOM images
               1 -> create file using diaphragm masks
        """.format(sys.argv[0]))
        exit()

    if mask == 1:
        filename = '{}/{}/{}/{}/maskIM (1).png'.format(DIR_MAN_LUNG_MASKS, patient, plan, sequence)

        image = cv2.imread('{}'.format(filename), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        for i in range(len(columns)):
            for row in range(h):
                image[row, columns[i]] = (255, 255, 0)
                image[row, 255] = (255, 255, 0)
        cv2.imwrite('corte.jpg', image)
        image = cv2.imread('corte.jpg', 0)

        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        create_2DST = Create2DST(patient, plan, sequence)

        data = create_2DST.create3DMatrix(plan=plan, sequence=sequence, mask=1)

        for i in range(len(columns)):
            img2DST = create_2DST.createOrthogonalPlane(data, columns[i])
            # cv2.imwrite('RM_{}_{}_{}_{}_2DST.jpg'.format(patient, plan, sequence, column[i]), img2DST)

            res = np.hstack((image, img2DST))

            if show == 1:
                cv2.imshow('image', img2DST)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow('res', res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save == 1:
                cv2.imwrite('img2DST_{}.jpg'.format(columns[i]), img2DST)
                cv2.imwrite('{}.jpg'.format(columns[i]), res)

        os.remove('corte.jpg')

    else:
        filename = '{}/{}/{}/{}/IM (1).jpg'.format(DIR_JPG, patient, plan, sequence)

        image = cv2.imread('{}'.format(filename), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        for i in range(len(columns)):
            for row in range(h):
                image[row, columns[i]] = (255, 255, 0)
                image[row, 255] = (255, 255, 0)
        cv2.imwrite('corte.jpg', image)
        image = cv2.imread('corte.jpg', 0)

        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        create_2DST = Create2DST(patient, plan, sequence)

        data = create_2DST.create3DMatrix(plan=plan, sequence=sequence, mask=0)

        for i in range(len(columns)):
            img2DST = create_2DST.createOrthogonalPlane(data, columns[i])
            # cv2.imwrite('RM_{}_{}_{}_{}_2DST.jpg'.format(patient, plan, sequence, column[i]), img2DST)

            res = np.hstack((image, img2DST))

            if show == 1:
                cv2.imshow('image', img2DST)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow('res', res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save == 1:
                cv2.imwrite('img2DST_{}.jpg'.format(columns[i]), img2DST)
                cv2.imwrite('{}.jpg'.format(columns[i]), res)

        os.remove('corte.jpg')
