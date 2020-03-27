#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np

DIR_JPG = os.path.expanduser("~/Documents/UDESC/JPG")
DIR_MAN_DIAHPRAGM_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/diaphragm")
DIR_MAN_LUNG_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/lung/Manual")
DIR_ASM = os.path.expanduser("~/Documents/UDESC/ASM/Mascaras/JPG")


class Create2DST(object):
    def __init__(self, patient, plan, sequence, side=0):
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
        elif mask == 0:
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
        else:
            if plan == 'Coronal':
                for arq in range(50):
                    if side == 0:
                        img =\
                            cv2.imread('{}/{}/{}/{}_L/mascIM ({}).jpg'
                                       .format(
                                           DIR_ASM,
                                           self.patient,
                                           plan,
                                           sequence,
                                           img_num), 0)
                    else:
                        img =\
                            cv2.imread('{}/{}/{}/{}_R/mascIM ({}).jpg'
                                       .format(
                                           DIR_ASM,
                                           self.patient,
                                           plan,
                                           sequence,
                                           img_num), 0)

                    l_npy.append(img)

                    img_num = img_num + 1
            else:
                for arq in range(50):
                    img =\
                        cv2.imread('{}/{}/{}/{}/mascIM ({}).jpg'
                                   .format(
                                       DIR_ASM,
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
        plan = 'Coronal'
        sequence = 5
        mask = 0  # 0 - Does not use masks | 1 - Use masks
        # Coronal columns: [70, 76, 82, 88, 94, 101, 107, 113] (8)
        # Sagittal columns: [101, 107, 114, 121, 128, 134, 141, 148, 154] (9)
        column = 101
        show = 1
        save = 0

        txtargv2 = '-patient={}'.format(patient)
        txtargv3 = '-plan={}'.format(plan)
        txtargv4 = '-sequence={}'.format(sequence)
        txtargv5 = '-mask={}'.format(mask)
        txtargv6 = '-column={}'.format(column)
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

        if txtargv.find('-column') != -1:
            txttmp = txtargv.split('-column')[1]
            txttmp = txttmp.split('=')[1]
            column = int(txttmp.split('|')[0])

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

    create_2DST = Create2DST(patient, plan, sequence)

    if mask == 1:
        filename = '{}/{}/{}/{}/maskIM (1).png'.format(DIR_MAN_LUNG_MASKS, patient, plan, sequence)

        image = cv2.imread('{}'.format(filename), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        for row in range(h):
            image[row, column] = (255, 255, 0)
            image[row, 255] = (255, 255, 0)

        cv2.imwrite('corte.jpg', image)
        image = cv2.imread('corte.jpg', 0)

        data = create_2DST.create3DMatrix(plan=plan, sequence=sequence, mask=1)

        img2DST = create_2DST.createOrthogonalPlane(data, column)

        res = np.hstack((image, img2DST))

        if show == 1:
            cv2.imshow('image', img2DST)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('res', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save == 1:
            cv2.imwrite('img2DST.png', img2DST)

            cv2.imwrite('{}.png'.format(column), res)

        os.remove('corte.jpg')

    elif mask == 0:
        filename = '{}/{}/{}/{}/IM (1).jpg'.format(DIR_JPG, patient, plan, sequence)
        # print('DIR_JPG/{}/{}/{}/IM (1).jpg'.format(patient, plan, sequence))

        image = cv2.imread('{}'.format(filename), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        for row in range(h):
            image[row, column] = (255, 255, 0)
            image[row, 255] = (255, 255, 0)

        cv2.imwrite('corte.jpg', image)
        image = cv2.imread('corte.jpg', 0)

        data = create_2DST.create3DMatrix(plan=plan, sequence=sequence, mask=0)

        img2DST = create_2DST.createOrthogonalPlane(data, column)

        res = np.hstack((image, img2DST))

        if show == 1:
            cv2.imshow('image', img2DST)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('res', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save == 1:
            cv2.imwrite('img2DST.jpg', img2DST)

            cv2.imwrite('{}.jpg'.format(column), res)

        os.remove('corte.jpg')

    else:
        if plan == 'Coronal':
            side = 0
            if side == 0:
                filename = '{}/{}/{}/{}_L/mascIM (1).jpg'.format(DIR_ASM, patient, plan, sequence)
            else:
                filename = '{}/{}/{}/{}_R/mascIM (1).jpg'.format(DIR_ASM, patient, plan, sequence)
        else:
            filename = '{}/{}/{}/{}/mascIM (1).jpg'.format(DIR_ASM, patient, plan, sequence)

        image = cv2.imread('{}'.format(filename), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        for row in range(h):
            image[row, column] = (255, 255, 0)
            image[row, 255] = (255, 255, 0)

        cv2.imwrite('corte.jpg', image)
        image = cv2.imread('corte.jpg', 0)

        data = create_2DST.create3DMatrix(plan=plan, sequence=sequence, mask=2)

        img2DST = create_2DST.createOrthogonalPlane(data, column)

        res = np.hstack((image, img2DST))

        if show == 1:
            cv2.imshow('image', img2DST)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('res', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save == 1:
            cv2.imwrite('img2DST.png', img2DST)

            cv2.imwrite('{}.png'.format(column), res)

        os.remove('corte.jpg')
