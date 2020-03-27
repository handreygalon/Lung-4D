#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

DIR_JPG = os.path.expanduser("~/Documents/UDESC/JPG")
DIR_MAN_DIAHPRAGM_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/diaphragm")


class Create2DST(object):
    def __init__(self, patient, plan, sequence, side):
        self.patient = patient
        self.plan = plan
        self.sequence = sequence
        self.side = side

        if plan == 'Sagittal':
            self.DIR_DICOM = "{}/{}/{}/{}".format(DIR_JPG, patient, plan, sequence)
            self.DIR_Masks = "{}/{}/{}/{}".format(DIR_MAN_DIAHPRAGM_MASKS, patient, plan, sequence)
        else:
            if side == 0:
                self.DIR_DICOM = "{}/{}/{}/{}_L".format(DIR_JPG, patient, plan, sequence)
                self.DIR_Masks = "{}/{}/{}/{}_L".format(DIR_MAN_DIAHPRAGM_MASKS, patient, plan, sequence)
            elif side == 1:
                self.DIR_DICOM = "{}/{}/{}/{}_R".format(DIR_JPG, patient, plan, sequence)
                self.DIR_Masks = "{}/{}/{}/{}_R".format(DIR_MAN_DIAHPRAGM_MASKS, patient, plan, sequence)
            else:
                self.DIR_DICOM = "{}/{}/{}/{}".format(DIR_JPG, patient, plan, sequence)
                self.DIR_Masks = "{}/{}/{}/{}".format(DIR_MAN_DIAHPRAGM_MASKS, patient, plan, sequence)

    def create3DMatrix(self, mask=0):
        img_num = 1
        l_npy = []

        if mask == 1:
            for arq in range(50):
                img = cv2.imread('{}/maskIM ({}).png'.format(self.DIR_Masks, img_num), 0)

                l_npy.append(img)

                img_num = img_num + 1
        else:
            for arq in range(50):
                # img = cv2.imread('{}/IM ({}).png'.format(self.DIR_DICOM, img_num), 0)
                img = cv2.imread('{}/IM ({}).jpg'.format(self.DIR_DICOM, img_num), 0)

                l_npy.append(img)

                img_num = img_num + 1

        data = np.stack(l_npy, axis=0)

        return data

    def createOrthogonalPlane(self, data, column):
        slice2D = data[:, :, column]
        img2DST = np.swapaxes(slice2D, 0, 1)

        return img2DST


if __name__ == '__main__':
    patient = 'Matsushita'
    plan = 'Coronal'
    sequence = 21
    side = 2  # 0 - left | 1 - right | 2 - Both (obs. use side = 2 when mask = 0)
    mask = 0  # 0 - Use DICOM | 1 - Use Masks
    show = 1  # 0 - Not show | 1 - Show
    save = 0  # 0 - Not save | 1 - Save
    # Iwasawa
    # Cor [70, 76, 82, 88, 94, 101, 107, 113, 168, 174, 180, 186, 192]
    # Sag [134, 141, 148, 154, 161]
    # Matsushita
    # Cor [66, 72, 78, 84, 90, 96, 103, 109, 151, 157, 163, 170, 176, 182, 188]
    # Sag [98, 102, 105, 109, 112, 115, 119, 122, 125, 129, 132, 135, 139, 142, 146, 149, 152, 156]
    column = 180

    create_2DST = Create2DST(patient, plan, sequence, side)

    if mask == 1:
        stack = create_2DST.create3DMatrix(mask)
        img2DST = create_2DST.createOrthogonalPlane(stack, column)

        filename = 'IM (1).jpg'
        image = cv2.imread('{}/{}/{}/{}/{}'
                           .format(DIR_JPG, patient, plan, sequence, filename), cv2.IMREAD_COLOR)

        h, w = image.shape[:2]
        for row in range(h):
            image[row, column] = (255, 255, 0)
            image[row, 255] = (255, 255, 0)
        cv2.imwrite('corte.jpg', image)
        image = cv2.imread('corte.jpg', 0)

        res = np.hstack((image, img2DST))

        if show == 1:
            cv2.imshow('res', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save == 1:
            cv2.imwrite('Mask_{}_{}_{}_cut.png'.format(plan, sequence, column), res)
            cv2.imwrite('Mask_{}_{}_{}_2DST.png'.format(plan, sequence, column), img2DST)

        os.remove("corte.jpg")

    else:
        stack = create_2DST.create3DMatrix(mask)
        img2DST = create_2DST.createOrthogonalPlane(stack, column)
        # cv2.imshow('image', img2DST)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # filename = 'IM (1).png'
        filename = 'IM (1).jpg'
        image = cv2.imread('{}/{}/{}/{}/{}'
                           .format(DIR_JPG, patient, plan, sequence, filename), cv2.IMREAD_COLOR)

        h, w = image.shape[:2]
        for row in range(h):
            image[row, column] = (255, 255, 0)
            image[row, 255] = (255, 255, 0)
        cv2.imwrite('corte.jpg', image)
        image = cv2.imread('corte.jpg', 0)

        res = np.hstack((image, img2DST))
        if show == 1:
            cv2.imshow('res', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save == 1:
            cv2.imwrite('{}_{}_{}_cut.png'.format(plan, sequence, column), res)
            cv2.imwrite('{}_{}_{}_2DST.png'.format(plan, sequence, column), img2DST)

        os.remove("corte.jpg")
