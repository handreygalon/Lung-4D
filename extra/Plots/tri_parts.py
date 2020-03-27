#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import dicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


DIR_DICOM = os.path.expanduser("~/Documents/UDESC/DICOM")
DIR_JPG = os.path.expanduser("~/Documents/UDESC/JPG")
DIR_ASM_LUNG_MASKS_JPG = os.path.expanduser("~/Documents/UDESC/ASM/Mascaras/JPG")
DIR_ASM_LUNG_MASKS_PNG = os.path.expanduser("~/Documents/UDESC/ASM/Mascaras/PNG")
DIR_MAN_DIAHPRAGM_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/diaphragm")
DIR_MAN_LUNG_MASKS = os.path.expanduser("~/Documents/UDESC/segmented/lung")
DIR_2DST_DICOM = os.path.expanduser("~/Documents/UDESC/DICOM/2DST_DICOM")
DIR_2DST_Mask = os.path.expanduser("~/Documents/UDESC/DICOM/2DST_Masks")
DIR_2DST_Diaphragm = os.path.expanduser("~/Documents/UDESC/DICOM/2DST_diaphragm")


class PointCloud(object):
    def __init__(self, patient):
        self.patient = patient

    def get_folder_names(self, patient, plan):
        if patient == 'Matsushita':
            if plan == 'Sagittal':
                folder_lung_left = [2, 3, 6, 7, 9]
                folder_lung_right = [10, 11, 12, 13, 14]
                folder_all = folder_lung_left + folder_lung_right

                return folder_lung_left, folder_lung_right, folder_all
            else:
                return sorted(map(int, os.listdir('{}/{}/{}'.format(DIR_ASM_LUNG_MASKS_PNG, patient, plan))))

        elif patient == 'Iwasawa':
            if plan == 'Sagittal':
                folder_lung_left = [1, 2, 3, 4, 5, 6, 7, 8]
                folder_lung_right = [13, 14, 15, 16, 17]
                folder_all = folder_lung_left + folder_lung_right

                return folder_lung_left, folder_lung_right, folder_all
            else:
                # return sorted(map(int, os.listdir('{}/{}/{}'.format(DIR_ASM_LUNG_MASKS_PNG, patient, plan))))
                return [1, 3, 8, 9, 10, 11, 12, 13]

    def openASMFile(self, plan, sequence, side):
        """ Open the right file from images of lungs segmented by ASM """
        if plan == 'Coronal':
            if side == 0:
                left_dataset =\
                    open('{}/{}/{}/{}_L/Pontos.txt'.format(
                        DIR_ASM_LUNG_MASKS_JPG,
                        self.patient,
                        plan,
                        sequence,
                        side), 'r').read().split('\n')

                del left_dataset[50:]

                return left_dataset
            else:
                right_dataset =\
                    open('{}/{}/{}/{}_R/Pontos.txt'.format(
                        DIR_ASM_LUNG_MASKS_JPG,
                        self.patient,
                        plan,
                        sequence,
                        side), 'r').read().split('\n')

                del right_dataset[50:]

                return right_dataset
        else:
            dataset =\
                open('{}/{}/{}/{}/Pontos.txt'.format(
                    DIR_ASM_LUNG_MASKS_JPG,
                    self.patient,
                    plan,
                    sequence), 'r').read().split('\n')

            return dataset

    def openManualFile(self, plan, sequence, side):
        """ Open the right file from images of lungs segmented manually """
        if plan == 'Coronal':
            if side == 0:
                left_dataset =\
                    open('{}/{}/{}/{}_L/points_left.txt'.format(
                        DIR_ASM_LUNG_MASKS_JPG,
                        self.patient,
                        plan,
                        sequence,
                        side), 'r').read().split('\n')

                del left_dataset[50:]

                return left_dataset
            else:
                right_dataset =\
                    open('{}/{}/{}/{}_R/points_right.txt'.format(
                        DIR_ASM_LUNG_MASKS_JPG,
                        self.patient,
                        plan,
                        sequence,
                        side), 'r').read().split('\n')

                del right_dataset[50:]

                return right_dataset
        else:
            dataset =\
                open('{}/{}/{}_R/{}/points.txt'.format(
                    DIR_ASM_LUNG_MASKS_JPG,
                    self.patient,
                    plan,
                    sequence), 'r').read().split('\n')

            return dataset

    def readASMFile(self, plan, sequence, side, option=0):
        if plan == 'Coronal':
            if side == 0:
                # Left lung
                filename = 'points_left.txt'
            elif side == 1:
                # Right lung
                filename = 'points_right.txt'
            else:
                # Both of the lungs
                filename = 'points_all.txt'
        else:
            filename = 'points.txt'

        points =\
            open(
                '{}/{}/{}/{}/{}'.format(DIR_ASM_LUNG_MASKS_PNG, self.patient, plan, sequence, filename),
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
            # tupla = [int(spts[0][1:]), int(spts[1][:-1])]
            list_pts.append(tupla)

        if option == 0:
            return list_pts
        elif option == 1:
            return list_pts[25:35]
        else:
            list_left = list_pts[:25]
            list_right = list_pts[35:]
            return list_left + list_right

    def readManualPoints(self, plan, sequence, side, option=0):
        """ Read points from diaphragmatic surface segmented manually """
        if plan == 'Coronal':
            if side == 0:
                dataset =\
                    open('{}/{}/{}/{}_L/points.txt'.format(
                        DIR_MAN_DIAHPRAGM_MASKS,
                        self.patient,
                        plan,
                        sequence,
                        side), 'r').read().split('\n')

                del dataset[50:]
            elif side == 1:
                dataset =\
                    open('{}/{}/{}/{}_R/points.txt'.format(
                        DIR_MAN_DIAHPRAGM_MASKS,
                        self.patient,
                        plan,
                        sequence,
                        side), 'r').read().split('\n')

                del dataset[50:]
        else:
            dataset =\
                open('{}/{}/{}/{}/points.txt'.format(
                    DIR_MAN_DIAHPRAGM_MASKS,
                    self.patient,
                    plan,
                    sequence), 'r').read().split('\n')

            del dataset[50:]

        all_points, ltuples = list(), list()

        for i in range(len(dataset)):
            lpts = dataset[i].replace('), ', ');')[1:-1].split(';')

            for j in range(len(lpts)):
                pts = lpts[j].split(',')
                tupla = (int(pts[0][1:]), int(pts[1][1:-1]))
                ltuples.append(tupla)

            all_points.append(ltuples)
            ltuples = []

        return all_points

    def point3D(self, plan, sequence, imgnum, pts, xs, ys, zs):
        if imgnum < 10:
            img = dicom.read_file("{}/{}/{}/{}/IM_0000{}.dcm".format(
                DIR_DICOM, self.patient, plan, sequence, imgnum))
        else:
            img = dicom.read_file("{}/{}/{}/{}/IM_000{}.dcm".format(
                DIR_DICOM, self.patient, plan, sequence, imgnum))

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

    def plot3D(self, pts, howplot=0, ax=None, dots=0):
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        xpts, ypts, zpts = list(), list(), list()
        for i in range(len(pts)):
            xpts.append(pts[i][0])
            ypts.append(pts[i][1])
            zpts.append(pts[i][2])

        X = np.asarray(xpts)
        Y = np.asarray(ypts)
        Z = np.asarray(zpts)

        if howplot == 'scatter':
            ax.scatter(X, Y, Z, s=6, c='r', marker='o')
        elif howplot == 'tri':
            ax.plot_trisurf(X, Y, Z, linewidth=0.2, color='b', edgecolor='Black')
        elif howplot == 'tri_convex_hull':
            # IT IS NOT WORKS
            points = np.array(pts)

            cvx_hull = ConvexHull(points)
            # print(points.shape)
            # print(cvx_hull.simplices.shape)
            # print(points.ndim)
            # print(cvx_hull.simplices.ndim)

            # X, Y, Z = points.T
            X, Y, Z = cvx_hull.simplices.T

            # tri = mtri.Triangulation(X, Y, triangles=cvx_hull.simplices)

            ax.plot_trisurf(X, Y, Z, linewidth=0.2, color='b', edgecolor='Black')
        elif howplot == 'convex_hull':
            points = np.array(pts)

            cvx_hull = ConvexHull(points)
            x, y, z = points.T

            tri = mtri.Triangulation(x, y, triangles=cvx_hull.simplices)

            # ax.plot_trisurf(tri, z)
            ax.plot_trisurf(tri, z, cmap=plt.cm.bone)

            if dots == 1:
                hull_indices = np.unique(cvx_hull.simplices.flat)
                hull_points = points[hull_indices, :]
                xhull = hull_points[:, 0]
                yhull = hull_points[:, 1]
                zhull = hull_points[:, 2]
                ax.scatter(xhull, yhull, zhull, c='r', s=2)
        elif howplot == 'tri_closed':
            points = np.array(pts)

            points -= points.mean(axis=0)
            rad = np.linalg.norm(points, axis=1)
            zen = np.arccos(points[:, -1] / rad)
            azi = np.arctan2(points[:, 1], points[:, 0])
            tris = mtri.Triangulation(zen, azi)
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tris.triangles, cmap=plt.cm.bone)
        elif howplot == 'tri_not_closed':
            points = np.array(pts)

            hull = ConvexHull(points)
            hull_indices = np.unique(hull.simplices.flat)
            hull_points = points[hull_indices, :]
            xhull = hull_points[:, 0].tolist()
            yhull = hull_points[:, 1].tolist()
            zhull = hull_points[:, 2].tolist()
            lpoints = list()
            for i in range(len(xhull)):
                lpts = [xhull[i], yhull[i], zhull[i]]
                lpoints.append(lpts)
            points = np.asarray(lpoints)

            points -= points.mean(axis=0)
            rad = np.linalg.norm(points, axis=1)
            zen = np.arccos(points[:, -1] / rad)
            azi = np.arctan2(points[:, 1], points[:, 0])
            tris = mtri.Triangulation(zen, azi)
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tris.triangles, cmap=plt.cm.bone)

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')


def test(listpoints):
    lungimg = np.zeros((256, 256, 3), np.uint8)
    for i in range(len(listpoints)):
        if i <= 24:
            cv2.circle(lungimg, listpoints[i], 1, (0, 255, 255), -1)
        elif i > 24 and i < 35:  # Diaphragm region
            cv2.circle(lungimg, listpoints[i], 1, (0, 255, 0), -1)
        else:
            cv2.circle(lungimg, listpoints[i], 1, (0, 0, 255), -1)

    diaphragmimg = np.zeros((256, 256, 3), np.uint8)
    option = 1
    listpoints = pc.readASMFile(option)
    print("Points: \n{} ({})\n".format(listpoints, len(listpoints)))
    for i in range(len(listpoints)):
        cv2.circle(diaphragmimg, listpoints[i], 1, (0, 255, 0), -1)

    option = 2
    listpoints = pc.readASMFile(option)
    print("Points: \n{} ({})\n".format(listpoints, len(listpoints)))
    wallsimg = np.zeros((256, 256, 3), np.uint8)
    for i in range(len(listpoints)):
        cv2.circle(wallsimg, listpoints[i], 1, (0, 255, 0), -1)

    res = np.hstack((lungimg, diaphragmimg, wallsimg))
    cv2.imshow('Image', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def left_lung(patient, pc, opt):
    folders_cor = pc.get_folder_names(patient, 'Coronal')
    folders_sag_left, folders_sag_right, folders_sag_all = pc.get_folder_names(patient, 'Sagittal')

    # Coronal
    lpts_cor = []
    for folder in folders_cor:
        lpts_cor.append(pc.readASMFile(plan='Coronal', sequence=folder, side=0, option=opt))
    # print("Coronal points: {} ({})\n".format(lpts_cor, len(lpts_cor)))

    # Sagittal
    lpts_sag = []
    for folder in folders_sag_left:
        lpts_sag.append(pc.readASMFile(plan='Sagittal', sequence=folder, side=0, option=opt))

    x_pts, y_pts, z_pts = list(), list(), list()
    i = 0
    for folder in folders_cor:
        pc.point3D(plan='Coronal', sequence=folder, imgnum=1, pts=lpts_cor[i], xs=x_pts, ys=y_pts, zs=z_pts)
        i += 1

    # Create a list with all 3D points
    lpts_cor_3D = list()
    for i in range(len(x_pts)):
        lpts_cor_3D.append((x_pts[i], y_pts[i], z_pts[i]))
    # print("{}\n".format(lpts_cor_3D))

    x_pts, y_pts, z_pts = [], [], []
    i = 0
    for folder in folders_sag_left:
        pc.point3D(plan='Sagittal', sequence=folder, imgnum=1, pts=lpts_sag[i], xs=x_pts, ys=y_pts, zs=z_pts)
        i += 1

    # Create a list with all 3D points
    lpts_sag_3D = list()
    for i in range(len(x_pts)):
        lpts_sag_3D.append((x_pts[i], y_pts[i], z_pts[i]))
    # print("{}\n".format(lpts_sag_3D))

    lpts = lpts_sag_3D + lpts_cor_3D
    return lpts


def right_lung(patient, pc, opt):
    folders_cor = pc.get_folder_names(patient, 'Coronal')
    folders_sag_left, folders_sag_right, folders_sag_all = pc.get_folder_names(patient, 'Sagittal')
    # Coronal
    lpts_cor = []
    for folder in folders_cor:
        lpts_cor.append(pc.readASMFile(plan='Coronal', sequence=folder, side=1, option=opt))
    # print("Coronal points: {} ({})\n".format(lpts_cor, len(lpts_cor)))

    # Sagittal
    lpts_sag = []
    for folder in folders_sag_right:
        lpts_sag.append(pc.readASMFile(plan='Sagittal', sequence=folder, side=1, option=opt))

    x_pts, y_pts, z_pts = list(), list(), list()
    i = 0
    for folder in folders_cor:
        pc.point3D(plan='Coronal', sequence=folder, imgnum=1, pts=lpts_cor[i], xs=x_pts, ys=y_pts, zs=z_pts)
        i += 1

    # Create a list with all 3D points
    lpts_cor_3D = list()
    for i in range(len(x_pts)):
        lpts_cor_3D.append((x_pts[i], y_pts[i], z_pts[i]))
    # print("{}\n".format(lpts_cor_3D))

    x_pts, y_pts, z_pts = [], [], []
    i = 0
    for folder in folders_sag_right:
        pc.point3D(plan='Sagittal', sequence=folder, imgmun=1, pts=lpts_sag[i], xs=x_pts, ys=y_pts, zs=z_pts)
        i += 1

    # Create a list with all 3D points
    lpts_sag_3D = list()
    for i in range(len(x_pts)):
        lpts_sag_3D.append((x_pts[i], y_pts[i], z_pts[i]))
    # print("{}\n".format(lpts_sag_3D))

    lpts = lpts_sag_3D + lpts_cor_3D
    return lpts


def both_lung(patient, pc, opt):
    leftpts = left_lung(patient=patient, pc=pc, opt=opt)
    rightpts = right_lung(patient=patient, pc=pc, opt=opt)

    return leftpts, rightpts


if __name__ == '__main__':
    try:
        patient = 'Iwasawa'
        side_lung = 2  # 0 - Left, 1 - Right, 2 - Both, 3 - Both separated
        plot_option = 'convex_hull'
        plot_dots = 0  # 0 - Does not plot dots | 1 - plot dots
        lungs_part = 0  # 0 - Entire lung | 1 - Just diaphragm | 2 - Lung without diaphragm

        txtargv2 = '-patient={}'.format(patient)
        txtargv3 = '-side_lung={}'.format(side_lung)
        txtargv4 = '-plot={}'.format(plot_option)
        txtargv5 = '-plot_dots={}'.format(plot_dots)
        txtargv6 = '-lungs_part={}'.format(lungs_part)

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

        txtargv = '{}|{}|{}|{}|{}'.format(txtargv2, txtargv3, txtargv4, txtargv5, txtargv6)
        if txtargv.find('-patient') != -1:
            txttmp = txtargv.split('-patient')[1]
            txttmp = txttmp.split('=')[1]
            patient = txttmp.split('|')[0]

        if txtargv.find('-side_lung') != -1:
            txttmp = txtargv.split('-side_lung')[1]
            txttmp = txttmp.split('=')[1]
            side_lung = int(txttmp.split('|')[0])

        if txtargv.find('-plot') != -1:
            txttmp = txtargv.split('-plot')[1]
            txttmp = txttmp.split('=')[1]
            plot_option = txttmp.split('|')[0]

        if txtargv.find('-plot_dots') != -1:
            txttmp = txtargv.split('-plot_dots')[1]
            txttmp = txttmp.split('=')[1]
            plot_dots = int(txttmp.split('|')[0])

        if txtargv.find('-lungs_part') != -1:
            txttmp = txtargv.split('-lungs_part')[1]
            txttmp = txttmp.split('=')[1]
            lungs_part = int(txttmp.split('|')[0])

    except ValueError:
        print(
            """
            Example of use:\n

            $ python {} -patient=Iwasawa -side_lung=2 -plot=convex_hull -plot_dots=0 -lungs_part=0
            """.format(sys.argv[0]))
        exit()

    pc = PointCloud(patient)

    # listpoints = pc.readASMFile(plan, sequence, side_lung, lungs_part)
    # print("Points: \n{} ({})\n".format(listpoints, len(listpoints)))

    if side_lung == 0:
        lpts = left_lung(patient, pc, lungs_part)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        pc.plot3D(pts=lpts, howplot=plot_option, ax=ax, dots=plot_dots)
        plt.show()
    elif side_lung == 1:
        lpts = right_lung(patient, pc, lungs_part)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        pc.plot3D(pts=lpts, howplot=plot_option, ax=ax, dots=plot_dots)
        plt.show()
    elif side_lung == 2:
        leftpts, rightpts = both_lung(patient, pc, lungs_part)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        pc.plot3D(pts=leftpts, howplot=plot_option, ax=ax, dots=plot_dots)
        pc.plot3D(pts=rightpts, howplot=plot_option, ax=ax, dots=plot_dots)
        plt.show()
    else:
        leftpts, rightpts = both_lung(patient, pc, lungs_part)

        fig = plt.figure(figsize=plt.figaspect(0.4))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        pc.plot3D(pts=leftpts, howplot=plot_option, ax=ax, dots=plot_dots)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        pc.plot3D(pts=rightpts, howplot=plot_option, ax=ax, dots=plot_dots)
        plt.show()
