#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from util.constant import *


class Create2DST(object):
    """
    Create 2DST images in intersection segments

    References
    ----------
    [1] Tavares, R. S., "Segmentacao do Pulmao em Sequencias de
    Imagens de Ressonancia Magnetica Utilizando a Transformada de
    Hough". Dissertacao (Mestrado) - Escola Politecnica da Universidade
    de Sao Paulo, Sao Paulo, 2011.
    """

    def __init__(self, patient):
        self.patient = patient

    def openFile(self):
        dataset = open('mapping.txt', 'r').read().split('\n')
        dataset.pop(-1)

        return dataset

    def create3DMatrix(self, plan, sequence, mask=False):
        """
        Define a Spatio Temporal Volume (STV) stacking the images
        from temporal sequences.

        Params
        ------

        plan: string
            Coronal or Sagittal

        sequence: int
            Sequence number of a patient

        mask: boolean, optional
            If 'True', create 2DST images using silhete (masks) os lungs,
            otherwise, create 2DST images using DICOM files

        Return
        ------
        data: numpy array
            Tridimensional array that represents the stack of the images
        """
        img_num = 1
        l_npy = []

        if mask:
            for arq in range(50):
                img =\
                    cv2.imread('{}/{}/{}/{}/maskIM ({}).png'
                               .format(
                                   DIR_MAN_DIAHPRAGM_MASKS,
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
        """
        Create a plan that is parallel to t axis. The intersection of STV with
        this plan defines a 2D spatio temporal (2DST) image.

        Params
        ------
        data: numpy array
            The 3D matrix (STV)

        column: int
            Location that intersect the STV

        Return
        ------
        img2DST: numpy array
            Image that represents the 2DST image
        """
        slice2D = data[:, :, column]
        img2DST = np.swapaxes(slice2D, 0, 1)

        return img2DST


if __name__ == '__main__':
    try:
        patient = 'Iwasawa'
        mask = 0  # 0 - Does not use masks | 1 - Use masks

        txtargv2 = '-patient={}'.format(patient)
        txtargv3 = '-mask={}'.format(mask)

        if len(sys.argv) > 1:
            txtargv2 = sys.argv[1]
            if len(sys.argv) > 2:
                txtargv3 = sys.argv[2]

        txtargv = '{}|{}'.format(txtargv2, txtargv3)

        if txtargv.find('-patient') != -1:
            txttmp = txtargv.split('-patient')[1]
            txttmp = txttmp.split('=')[1]
            patient = txttmp.split('|')[0]

        if txtargv.find('-mask') != -1:
            txttmp = txtargv.split('-mask')[1]
            txttmp = txttmp.split('=')[1]
            mask = int(txttmp.split('|')[0])
    except ValueError:
        print("""
        Examples of use:

        $ python {} -patient=Iwasawa -mask=0

        Parameters:

        patient = Iwasawa -> Patient's name
        mask = 0 -> create file using original DICOM images
               1 -> create file using diaphragm masks
        """.format(sys.argv[0]))
        exit()

    create_2DST = Create2DST(patient)

    dataset = create_2DST.openFile()

    if mask:
        for data in dataset:
            # Create 2DST image - Coronal
            coronal_sequence = data.split(';')[0].split('-')[1]
            coronal_column = int(data.split(';')[0].split('-')[2])

            stack =\
                create_2DST.create3DMatrix(
                    'Coronal', coronal_sequence, mask=True)

            img2DST = create_2DST.createOrthogonalPlane(stack, coronal_column)

            cv2.imwrite('{}/{}/Coronal/{}.png'
                        .format(DIR_2DST_Mask, patient, data), img2DST)

            # Create 2DST image - Sagittal
            sagittal_sequence = data.split(';')[1].split('-')[1]
            sagittal_column = int(data.split(';')[1].split('-')[2])

            stack =\
                create_2DST.create3DMatrix(
                    'Sagittal', sagittal_sequence, mask=True)

            img2DST = create_2DST.createOrthogonalPlane(stack, sagittal_column)

            cv2.imwrite('{}/{}/Sagittal/{}.png'
                        .format(DIR_2DST_Mask, patient, data), img2DST)
    else:
        for data in dataset:
            # Create 2DST image - Coronal
            coronal_sequence = data.split(';')[0].split('-')[1]
            coronal_column = int(data.split(';')[0].split('-')[2])

            stack = create_2DST.create3DMatrix('Coronal', coronal_sequence)

            img2DST = create_2DST.createOrthogonalPlane(stack, coronal_column)
            # cv2.imshow('image', img2DST)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite('{}/{}/Coronal/{}.png'
                        .format(DIR_2DST_DICOM, patient, data), img2DST)

            # Create 2DST image - Sagittal
            sagittal_sequence = data.split(';')[1].split('-')[1]
            sagittal_column = int(data.split(';')[1].split('-')[2])

            stack = create_2DST.create3DMatrix('Sagittal', sagittal_sequence)

            img2DST = create_2DST.createOrthogonalPlane(stack, sagittal_column)
            # cv2.imshow('image', img2DST)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite('{}/{}/Sagittal/{}.png'
                        .format(DIR_2DST_DICOM, patient, data), img2DST)

    # coronal_sequence = dataset[0].split(';')[0].split('-')[1]
    # coronal_column = int(dataset[0].split(';')[0].split('-')[2])
    # stack = create_2DST.create3DMatrix('Coronal', coronal_sequence)
    # img2DST = create_2DST.createOrthogonalPlane(stack, coronal_column)
    # cv2.imshow('image', img2DST)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # sagittal_sequence = dataset[0].split(';')[1].split('-')[1]
    # sagittal_column = int(dataset[0].split(';')[1].split('-')[2])
    # stack = create_2DST.create3DMatrix('Sagittal', sagittal_sequence)
    # img2DST = create_2DST.createOrthogonalPlane(stack, sagittal_column)
    # cv2.imshow('image', img2DST)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
