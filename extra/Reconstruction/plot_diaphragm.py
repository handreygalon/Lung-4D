#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import dicom
import pydicom
import numpy as np
import itertools
# import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial import ConvexHull

from geomdl import BSpline
from geomdl import NURBS
from geomdl import Multi
from geomdl import utilities as utils
from geomdl import compatibility as compat
# from geomdl import exchange
from geomdl.visualization import VisMPL

from util.constant import *


class PointCloud(object):
    def __init__(self, patient):
        self.patient = patient

    def read_points(self, plan, sequence):
        """
        Converter list of points (from a txt file) to int

        Parameters
        ----------
        plan: string
            Must be 'Coronal' or 'Sagittal'

        sequence: int
            Sequence's number of a patient
        """

        dataset =\
            open('{}/{}/{}/points.txt'
                 .format(DIR_MAN_DIAHPRAGM_MASKS, self.patient, plan, sequence), 'r')\
            .read().split('\n')
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

    def point3D(self, plan, sequence, imgnum, pts):
        """
        Converts 2D point to a 3D point using mapping matrix

        Parameters
        ----------
        plan: string
            Must be 'Coronal' or 'Sagittal'

        sequence: int
            Sequence's number of a patient

        imgnum: int
            Image's number of the selected sequence

        pts: list
            List of tuples. The tuples represents the 2D points

        References:
        -----------
        [1] Tsuzuki, M. S. G.; Takase, F. K.; Gotoh, T.; Kagei, S.; Asakura, A.;
        Iwasawa, T.; Inoue, T. "Animated solid model of the lung lonstructed
        from unsynchronized MR sequential images". In: Computer-Aided Design.
        Vol. 41, pp. 573 - 585, 2009.
        """
        xs, ys, zs = list(), list(), list()

        if imgnum < 10:
            img = pydicom.dcmread("{}/{}/{}/{}/IM_0000{}.dcm".format(
                DIR_DICOM, self.patient, plan, sequence, imgnum))
            # img = dicom.read_file("{}/{}/{}/{}/IM_0000{}.dcm".format(
            #     DIR_DICOM, self.patient, plan, sequence, imgnum))
        else:
            img = pydicom.dcmread("{}/{}/{}/{}/IM_000{}.dcm".format(
                DIR_DICOM, self.patient, plan, sequence, imgnum))
            # img = dicom.read_file("{}/{}/{}/{}/IM_000{}.dcm".format(
            #     DIR_DICOM, self.patient, plan, sequence, imgnum))

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

    def plot3D(self, pts, fig=None, ax=None, color='r', size=2, howplot=0, dots=0):
        """
        Plot 3D points (only one lung)

        Parameters
        ----------
        pts: list
            List of tuples. The tuples represents the 2D points

        References
        ----------
        Method used to set the same scale for all axes
        Stackoverflow:
        https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        Extra: plt.gca().set_aspect("equal", "datalim")
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
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

        if howplot == 'wireframe':
            xpts, ypts, zpts = list(), list(), list()
            for i in range(len(pts)):
                xpts.append(pts[i][0])
                ypts.append(pts[i][1])
                zpts.append(pts[i][2])

            X = np.asarray([xpts])
            Y = np.asarray([ypts])
            Z = np.asarray([zpts])

            if dots == 1:
                ax.scatter(X, Y, Z, s=size, c='r', marker='o')

            ax.plot_wireframe(X, Y, Z)
        else:
            ax.scatter(X, Y, Z, s=size, c=color, marker='o')
            ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        plt.show()

    def plotDiaphragm3D(self, allpts, leftpts, rightpts, color='r', size=2, triangulation=False, plot=False, figure=None, axes=None):
        """ Plots diaphragm region of both lungs

        Parameters
        ----------
        allpts: list
            List of tuples with 3D points that represents the diaphragmatic surfaces (left and right)

        leftpts: list
            List of tuples with 3D points that represents the left diaphragmatic surface

        rightpts: list
            List of tuples with 3D points that represents the right diaphragmatic surface

        color: string
            Control points' color:
            Ex: b = blue | g = green | r = red | y =  yellow | k = black | m = magenta | c = cyan

        size: int
            The marker size in points**2

        triangulation: bool
            If True, the triangulation is made, otherwise it will not

        plot: bool
            If True, plot the surfaces, otherwise it will not

        figure: matplotlib.figure

        axes: matplotlib.axes
        """

        # LEFT
        lxpts, lypts, lzpts = list(), list(), list()
        for i in range(len(leftpts)):
            lxpts.append(leftpts[i][0])
            lypts.append(leftpts[i][1])
            lzpts.append(leftpts[i][2])

        lX = np.asarray(lxpts)
        lY = np.asarray(lypts)
        lZ = np.asarray(lzpts)

        axes.scatter(lX, lY, lZ, s=size, c='r', marker='o')

        if triangulation:
            axes.plot_trisurf(lX, lY, lZ, linewidth=0.2, antialiased=True)

        # RIGHT
        rxpts, rypts, rzpts = list(), list(), list()
        for i in range(len(rightpts)):
            rxpts.append(rightpts[i][0])
            rypts.append(rightpts[i][1])
            rzpts.append(rightpts[i][2])

        rX = np.asarray(rxpts)
        rY = np.asarray(rypts)
        rZ = np.asarray(rzpts)

        axes.scatter(rX, rY, rZ, s=size, c='r', marker='o')

        if triangulation:
            axes.plot_trisurf(rX, rY, rZ, linewidth=0.2, antialiased=True)

        # BOTH
        xpts, ypts, zpts = list(), list(), list()
        for i in range(len(allpts)):
            xpts.append(allpts[i][0])
            ypts.append(allpts[i][1])
            zpts.append(allpts[i][2])

        X = np.asarray(xpts)
        Y = np.asarray(ypts)
        Z = np.asarray(zpts)

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            axes.plot([xb], [yb], [zb], 'w')

        # plt.grid()
        if plot:
            plt.show()

    def createBsplineSurface(self, pts, corsequences, sagsequences, degree=3):
        # Create a BSpline surface instance
        surface = BSpline.Surface()

        # Set up the surface
        surface.degree_u = degree
        surface.degree_v = degree

        npoints_u = len(sagsequences)
        npoints_v = len(corsequences)

        # Set control points of the diaphragmatic surface
        surface.set_ctrlpts(pts, npoints_u, npoints_v)
        surface.knotvector_u = utils.generate_knot_vector(surface.degree_u, npoints_u)
        surface.knotvector_v = utils.generate_knot_vector(surface.degree_v, npoints_v)

        return surface

    def plotBsplineSurface(self, pts, corsequences, lsagsequences=[], rsagsequences=[], degree=3, visualization_type=-1, side=2):
        # side = 0 (Left) | side = 1 (Right)

        if side == 0:
            lsurface = self.createBsplineSurface(pts=pts, corsequences=corsequences, sagsequences=lsagsequences)

            # Set visualization component
            if visualization_type == 0:
                lsurface.delta = 0.01
                vis_comp = VisMPL.VisSurfScatter()
            elif visualization_type == 1:
                vis_comp = VisMPL.VisSurface()
            elif visualization_type == 2:
                vis_comp = VisMPL.VisSurfWireframe()
            else:
                vis_comp = VisMPL.VisSurfTriangle()
            lsurface.vis = vis_comp

            # Render the surface
            lsurface.render()

        else:
            rsurface = self.createBsplineSurface(pts=pts, corsequences=corsequences, sagsequences=rsagsequences)

            # Set visualization component
            if visualization_type == 0:
                rsurface.delta = 0.01
                vis_comp = VisMPL.VisSurfScatter()
            elif visualization_type == 1:
                vis_comp = VisMPL.VisSurface()
            elif visualization_type == 2:
                vis_comp = VisMPL.VisSurfWireframe()
            else:
                vis_comp = VisMPL.VisSurfTriangle()
            rsurface.vis = vis_comp

            # Render the surface
            rsurface.render()

    def plotBsplineSurfaces(self, left_pts, right_pts, corsequences, lsagsequences, rsagsequences, degree=3, visualization_type=-1):
        """ Create B-spline surface

        Parameters
        ----------
        left_pts: list
            List of tuples with 3D points that represents the left diaphragmatic surface

        right_pts: list
            List of tuples with 3D points that represents the right diaphragmatic surface

        corsequences: list
            List of int with the coronal sequences available

        lsagsequences: list
            List of int with the sagittal sequences available from the left lung

        rsagsequences: list
            List of int with the sagittal sequences available from the right lung

        degree: int
            B-spline surface's Degree (Default = 2)

        visualization_type: int
            -1: (Default) Wireframe plot for the control points and triangulated plot for the surface points
            0: Wireframe plot for the control points and scatter plot for the surface points
            1: Triangular mesh plot for the surface and wireframe plot for the control points grid
            2: Scatter plot for the control points and wireframe for the surface points
        """
        left_surface = self.createBsplineSurface(pts=left_pts, corsequences=corsequences, sagsequences=lsagsequences)
        right_surface = self.createBsplineSurface(pts=right_pts, corsequences=corsequences, sagsequences=rsagsequences)

        # Create a MultiSurface
        surfaces = Multi.MultiSurface()
        surfaces.add(left_surface)
        surfaces.add(right_surface)

        # Set visualization component
        if visualization_type == 0:
            surfaces.delta = 0.01
            vis_comp = VisMPL.VisSurfScatter()
        elif visualization_type == 1:
            vis_comp = VisMPL.VisSurface()
        elif visualization_type == 2:
            vis_comp = VisMPL.VisSurfWireframe()
        else:
            vis_comp = VisMPL.VisSurfTriangle()
        surfaces.vis = vis_comp

        # Render the surface
        surfaces.render()

    def createNURBSSurface(self, pts, corsequences, sagsequences, degree=3):
        # Dimensions of the control points grid
        npoints_u = len(corsequences)
        npoints_v = len(sagsequences)

        # Weights vector
        weights = [1] * (npoints_u * npoints_v)

        # Combine weights vector with the control points list
        t_ctrlptsw = compat.combine_ctrlpts_weights(pts, weights)

        # Since NURBS-Python uses v-row order, we need to convert the exported ones
        n_ctrlptsw = compat.change_ctrlpts_row_order(t_ctrlptsw, npoints_u, npoints_v)

        # Since we have no information on knot vectors, let's auto-generate them
        n_knotvector_u = utils.generate_knot_vector(degree, npoints_u)
        n_knotvector_v = utils.generate_knot_vector(degree, npoints_v)

        # Create a NURBS surface instance
        surf = NURBS.Surface()

        # Using __call__ method to fill the surface object
        surf(degree, degree, npoints_u, npoints_v, n_ctrlptsw, n_knotvector_u, n_knotvector_v)

        return surf

    def plotNURBSSurface(self, pts, corsequences, lsagsequences, rsagsequences, degree=3, visualization_type=-1, side=2):
        # side = 0 (Left) | side = 1 (Right)

        if side == 0:
            lsurface = self.createNURBSSurface(pts=pts, corsequences=corsequences, sagsequences=lsagsequences)
            # Set visualization component
            if visualization_type == 0:
                lsurface.delta = 0.01
                vis_comp = VisMPL.VisSurfScatter()
            elif visualization_type == 1:
                vis_comp = VisMPL.VisSurface()
            elif visualization_type == 2:
                vis_comp = VisMPL.VisSurfWireframe()
            else:
                vis_comp = VisMPL.VisSurfTriangle()
            lsurface.vis = vis_comp

            # Render the surface
            lsurface.render()

        else:
            rsurface = self.createNURBSSurface(pts=pts, corsequences=corsequences, sagsequences=rsagsequences)

            # Set visualization component
            if visualization_type == 0:
                rsurface.delta = 0.01
                vis_comp = VisMPL.VisSurfScatter()
            elif visualization_type == 1:
                vis_comp = VisMPL.VisSurface()
            elif visualization_type == 2:
                vis_comp = VisMPL.VisSurfWireframe()
            else:
                vis_comp = VisMPL.VisSurfTriangle()
            rsurface.vis = vis_comp

            # Render the surface
            rsurface.render()

    def plotNURBSSurfaces(self, left_pts, right_pts, corsequences, lsagsequences, rsagsequences, degree=3, visualization_type=-1):
        left_surface = self.createNURBSSurface(pts=left_pts, corsequences=corsequences, sagsequences=lsagsequences)
        right_surface = self.createNURBSSurface(pts=right_pts, corsequences=corsequences, sagsequences=rsagsequences)

        # Create a MultiSurface
        surfaces = Multi.MultiSurface()
        surfaces.add(left_surface)
        surfaces.add(right_surface)

        # Set visualization component
        if visualization_type == 0:
            surfaces.delta = 0.01
            vis_comp = VisMPL.VisSurfScatter()
        elif visualization_type == 1:
            vis_comp = VisMPL.VisSurface()
        elif visualization_type == 2:
            vis_comp = VisMPL.VisSurfWireframe()
        else:
            vis_comp = VisMPL.VisSurfTriangle()
        surfaces.vis = vis_comp

        # Render the surface
        surfaces.render()

    def test_plot3D(self):
        # Coronal
        lpts_cor = [(70, 174), (76, 172), (82, 170), (88, 169),
                    (94, 169), (101, 170), (107, 172), (113, 174)]

        # Convert 2D points to 3D
        for i in range(len(lpts_cor)):
            x_pts, y_pts, z_pts = self.point3D('Coronal', 9, 1, lpts_cor)
        print("({}, {}, {})\n".format(x_pts, y_pts, z_pts))

        # Create a list with all 3D points
        lpts_cor_3D = list()
        for i in range(len(x_pts)):
            lpts_cor_3D.append((x_pts[i], y_pts[i], z_pts[i]))
        print("{}\n".format(lpts_cor_3D))

        # Sagittal
        lpts_sag = [(134, 174), (141, 174), (148, 176)]

        # Convert 2D points to 3D
        for i in range(len(lpts_sag)):
            x_pts, y_pts, z_pts = self.point3D('Sagittal', 1, 11, lpts_sag)
        print("({}, {}, {})\n".format(x_pts, y_pts, z_pts))

        # Create a list with all 3D points
        lpts_sag_3D = list()
        for i in range(len(x_pts)):
            lpts_sag_3D.append((x_pts[i], y_pts[i], z_pts[i]))
        print("{}\n".format(lpts_sag_3D))

        # Joins the lists of 3D points of the coronal and sagittal planes
        lpts = lpts_cor_3D + lpts_sag_3D
        print(lpts)

        self.plot3D(lpts)


# if __name__ == '__main__':
#     patient = 'Iwasawa'
#     pc = PointCloud(patient)
#     pc.test_plot3D()
