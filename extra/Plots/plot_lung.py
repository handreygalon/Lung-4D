#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import dicom
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PointCloud(object):
    def __init__(self, patient):
        self.patient = patient
        self.DIR_DICOM = '/home/handrey/Documents/UDESC/DICOM/{}'\
            .format(patient)
        self.DIR_ASM = '/home/handrey/Documents/UDESC/ASM/Mascaras/PNG/{}'\
            .format(patient)

    def get_folder_names(self, plan):
        return sorted(map(int, os.listdir('{}/{}'.format(self.DIR_ASM, plan))))

    def read_file(self, plan, sequence):
        if plan == 'Coronal':
            filename = 'points_all.txt'
        else:
            filename = 'points.txt'

        points =\
            open(
                '{}/{}/{}/{}'.format(self.DIR_ASM, plan, sequence, filename),
                'r')\
            .read().split('\n')
        points.pop(-1)

        s = points[0].replace('),', ');')
        s = s[:-1]
        s = s[1:]
        s = s.replace(" ", "")
        ssplit = s.split(';')

        list_pts = []
        for i in range(len(ssplit)):
            spts = ssplit[i].split(',')

            tupla = (int(spts[0][1:]), int(spts[1][:-1]))
            list_pts.append(tupla)

        return list_pts

    def point3D(self, plan, sequence, pts, xs, ys, zs):
        img =\
            dicom.read_file(
                "{}/{}/{}/IM_00001.dcm".format(
                    self.DIR_DICOM, plan, sequence))

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


if __name__ == '__main__':
    patient = 'Matsushita'
    pc = PointCloud(patient)

    folders_cor = pc.get_folder_names('Coronal')
    folders_sag = pc.get_folder_names('Sagittal')

    # Coronal
    lpts_cor = []
    for folder in folders_cor:
        lpts_cor.append(pc.read_file('Coronal', folder))

    # Sagittal
    lpts_sag = []
    for folder in folders_sag:
        lpts_sag.append(pc.read_file('Sagittal', folder))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_pts = []
    y_pts = []
    z_pts = []
    i = 0
    for folder in folders_cor:
        pc.point3D('Coronal', str(folder),
                   lpts_cor[i], x_pts, y_pts, z_pts)
        i += 1

    X = np.asarray(x_pts)
    Y = np.asarray(y_pts)
    Z = np.asarray(z_pts)

    ax.scatter(X, Y, Z, s=1, c='r', marker='o')

    x_pts = []
    y_pts = []
    z_pts = []
    i = 0
    for folder in folders_sag:
        pc.point3D('Sagittal', str(folder),
                   lpts_sag[i], x_pts, y_pts, z_pts)
        i += 1

    X = np.asarray(x_pts)
    Y = np.asarray(y_pts)
    Z = np.asarray(z_pts)

    ax.scatter(X, Y, Z, s=1, c='r', marker='o')

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()
