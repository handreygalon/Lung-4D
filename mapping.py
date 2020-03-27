#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# import dicom
import pydicom
import numpy as np
import os
import os.path
from util.constant import *


class Mapping(object):
    """
    A pixel in MR image can be mapped to the 3D by using the DICOM mapping
    matrix [1]. The sagittal and coronal images have a common line segment,
    and the pixels of such line segment that are in the coronal e sagittal
    images, occupy the same 3D positions.

    Parameters
    ----------
    patient: string
        Name of patient that have DICOM files

    References:
    -----------
    [1] Tsuzuki, M. S. G.; Takase, F. K.; Gotoh, T.; Kagei, S.; Asakura, A.;
    Iwasawa, T.; Inoue, T. "Animated solid model of the lung lonstructed
    from unsynchronized MR sequential images". In: Computer-Aided Design.
    Vol. 41, pp. 573 - 585, 2009.
    """

    def __init__(self, patient):
        self.patient = patient

    def mapCoronal(self, sequence):
        """
        A pixel in a MR image can be mapped to the 3D by using the DICOM
        mapping matrix.
        Create the mapping matrix to a coronal sequence

        Params
        ------
        sequence: int
            Sequence number of a patient

        Return:
        -------
        matrix_coronal: numpy array
            Mapping matrix of coronal plan
        """

        path_coronal = "{}/Coronal/{}".format(self.patient, sequence)

        # names = os.listdir("{}/{}".format(DIR_DICOM, path_coronal))

        # ds_c = dicom.read_file("{}/{}/IM_00001.dcm".format(DIR_DICOM, path_coronal))
        ds_c = pydicom.dcmread("{}/{}/IM_00001.dcm".format(DIR_DICOM, path_coronal))

        # print("Image Position......:", ds_c.ImagePositionPatient)
        # print("Image Orientation...:", ds_c.ImageOrientationPatient)
        # print("Pixel spacing.......:", ds_c.PixelSpacing)

        xx_c = ds_c.ImageOrientationPatient[0]
        # xy_c = ds_c.ImageOrientationPatient[1]
        # xz_c = ds_c.ImageOrientationPatient[2]

        # yx_c = ds_c.ImageOrientationPatient[3]
        # yy_c = ds_c.ImageOrientationPatient[4]
        yz_c = ds_c.ImageOrientationPatient[5]

        delta_i_c = ds_c.PixelSpacing[0]
        delta_j_c = ds_c.PixelSpacing[1]

        sx_c = ds_c.ImagePositionPatient[0]
        sy_c = ds_c.ImagePositionPatient[1]
        sz_c = ds_c.ImagePositionPatient[2]

        matrix_coronal = np.matrix([[xx_c * delta_i_c, 0.0, 0.0, sx_c],
                                    [0.0, 0.0, 1.0, sy_c],
                                    [0.0, yz_c * delta_j_c, 0.0, sz_c],
                                    [0.0, 0.0, 0.0, 1.0]])
        # Check inverse matrix
        # print(np.dot(matrix_coronal, np.linalg.inv(matrix_coronal)))
        # print(matrix_coronal)
        # print(np.linalg.inv(matrix_coronal))
        return matrix_coronal

    def mapSagittal(self, sequence):
        """
        A pixel in a MR image can be mapped to the 3D by using the DICOM
        mapping matrix.
        Create the mapping matrix to a sagittal sequence

        Params
        ------
        sequence: int
            Sequence number of a patient

        Return:
        -------
        matrix_coronal: numpy array
            Mapping matrix of sagittal plan
        """

        path_sagittal = "{}/Sagittal/{}".format(self.patient, sequence)

        # ds_s = dicom.read_file("{}/{}/IM_00001.dcm".format(DIR_DICOM, path_sagittal))
        ds_s = pydicom.dcmread("{}/{}/IM_00001.dcm".format(DIR_DICOM, path_sagittal))

        # print("Image Position......:", ds_s.ImagePositionPatient)
        # print("Image Orientation...:", ds_s.ImageOrientationPatient)
        # print("Pixel spacing.......:", ds_s.PixelSpacing)

        # xx_s = ds_s.ImageOrientationPatient[0]
        xy_s = ds_s.ImageOrientationPatient[1]
        # xz_s = ds_s.ImageOrientationPatient[2]

        # yx_s = ds_s.ImageOrientationPatient[3]
        # yy_s = ds_s.ImageOrientationPatient[4]
        yz_s = ds_s.ImageOrientationPatient[5]

        delta_i_s = ds_s.PixelSpacing[0]
        delta_j_s = ds_s.PixelSpacing[1]

        sx_s = ds_s.ImagePositionPatient[0]
        sy_s = ds_s.ImagePositionPatient[1]
        sz_s = ds_s.ImagePositionPatient[2]

        matrix_sagittal = np.matrix([[0.0, 0.0, 1.0, sx_s],
                                     [xy_s * delta_i_s, 0.0, 0.0, sy_s],
                                     [0.0, yz_s * delta_j_s, 0.0, sz_s],
                                     [0.0, 0.0, 0.0, 1.0]])
        # Check inverse matrix
        # print(np.dot(matrix_sagittal, np.linalg.inv(matrix_sagittal)))
        # print(matrix_sagittal)
        # print(np.linalg.inv(matrix_sagittal))
        return matrix_sagittal

    def mapSagittalCoronal(self, mSagittal, mCoronal):
        """
        Determines the coordinate of the common straight line between
        coronal and sagittal sequences in 3D space.

        Params
        ------
        mSagittal: numpy array
            Mapping matrix of the sagittal plane

        mCoronal: numpy array
            Mapping matrix of the coronal plane

        Return
        ------
        i_c, i_s: int, int
            x coordinate that coronal and sagittal image intersect in 3D space
        """

        res_mCInv_mS = np.dot(np.linalg.inv(mCoronal), mSagittal)
        res_mSInv_mC = np.dot(np.linalg.inv(mSagittal), mCoronal)

        i_c = int(res_mCInv_mS.item((0, 3)))
        i_s = int(res_mSInv_mC.item((0, 3)))
        return i_c, i_s

    def findingIntersections(self, mask=False):
        """
        Determines the coordinate of the common straight line between
        coronal and sagittal sequences in 3D space to all the sequences
        from a patient.

        Parameters
        ----------
        mask: boolean, optional
            If 'True', create 2DST images using silhete (masks) os lungs,
            otherwise, create 2DST images using DICOM files
        """

        if mask:
            coronal_path =\
                "{}/{}/Coronal/".format(DIR_MAN_DIAHPRAGM_MASKS, self.patient)
            sagittal_path =\
                "{}/{}/Sagittal/".format(DIR_MAN_DIAHPRAGM_MASKS, self.patient)
        else:
            coronal_path =\
                "{}/{}/Coronal/".format(DIR_DICOM, self.patient)
            sagittal_path =\
                "{}/{}/Sagittal/".format(DIR_DICOM, self.patient)

        coronal_folders = list()
        for s in os.listdir(coronal_path):
            if s.isdigit():
                coronal_folders.append(int(s))
        coronal_folders = sorted(coronal_folders)

        sagittal_folders = list()
        for s in os.listdir(sagittal_path):
            if s.isdigit():
                sagittal_folders.append(int(s))
        sagittal_folders = sorted(sagittal_folders)

        # coronal_folders =\
        #     sorted([int(name) for name in os.listdir(coronal_path)])
        # sagittal_folders =\
        #     sorted([int(name) for name in os.listdir(sagittal_path)])

        file = open('mapping.txt', 'w')
        l_mapping = []

        for c in range(len(coronal_folders)):
            cor = self.mapCoronal(coronal_folders[c])
            # print(coronal_folders[c])
            for s in range(len(sagittal_folders)):
                # print(sagittal_folders[s])
                sag = self.mapSagittal(sagittal_folders[s])
                i_c, i_s = self.mapSagittalCoronal(sag, cor)
                l_mapping.append("c-{}-{};s-{}-{}".format(
                    coronal_folders[c], i_c, sagittal_folders[s], i_s))

        for item in l_mapping:
            file.write("{}\n".format(item))

        file.close()


if __name__ == '__main__':
    try:
        patient = 'Iwasawa'
        mask = 1

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

    mapping = Mapping(patient)
    if mask == 0:
        mapping.findingIntersections(mask=False)
    else:
        mapping.findingIntersections(mask=True)

    '''
    name = 'Iwasawa'
    mapping = Mapping(name)

    # sag = mapping.mapSagittal(1)
    # cor = mapping.mapCoronal(3)

    # i_c, i_s = mapping.mapSagittalCoronal(sag, cor)
    # print("ic: {} | is: {}".format(i_c, i_s))

    mask = True  # Use masks
    # mask = False  # Use DICOM
    mapping.findingIntersections(mask)
    '''
