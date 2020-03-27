#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pydicom
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline

from util.constant import *


class Reconstruction(object):
    def __init__(self, patient):
        self.patient = patient
        self.colorleftlung = '#A91818'
        self.colorpointsleftlung = '#FF3232'
        self.colorrightlung = '#FF3232'
        self.colorpointsrightlung = '#701700'

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
        else:
            img = pydicom.dcmread("{}/{}/{}/{}/IM_000{}.dcm".format(
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

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        plt.show()

    def plotLungs3D(self, all_points, left_pts=None, right_pts=None, fig=None, axes=None, color='r', size=2, howplot=0, dots=0):
        """
        Plot 3D points (both lungs)

        Parameters
        ----------
        left_pts: list
            List of tuples. The tuples represents of the left lung points
        right_pts: list
            List of tuples. The tuples represents of the right lung points
        """
        # LEFT
        lxpts, lypts, lzpts = list(), list(), list()
        for i in range(len(left_pts)):
            lxpts.append(left_pts[i][0])
            lypts.append(left_pts[i][1])
            lzpts.append(left_pts[i][2])

        lX = np.asarray(lxpts)
        lY = np.asarray(lypts)
        lZ = np.asarray(lzpts)

        axes.scatter(lX, lY, lZ, s=size, c='r', marker='o')

        # RIGHT
        rxpts, rypts, rzpts = list(), list(), list()
        for i in range(len(right_pts)):
            rxpts.append(right_pts[i][0])
            rypts.append(right_pts[i][1])
            rzpts.append(right_pts[i][2])

        rX = np.asarray(rxpts)
        rY = np.asarray(rypts)
        rZ = np.asarray(rzpts)

        axes.scatter(rX, rY, rZ, s=size, c='r', marker='o')

        # BOTH
        xpts, ypts, zpts = list(), list(), list()
        for i in range(len(all_points)):
            xpts.append(all_points[i][0])
            ypts.append(all_points[i][1])
            zpts.append(all_points[i][2])

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

        plt.show()

    def plotColor3D(self, pts_step1, pts_step2, pts_step3, color1='r', color2='b', color3='c', size=2):
        """
        The points are colored with different colors according to the registration step

        Parameters:
        pts_step1: list
            Points of the first step of the register
        pts_step2: list
            Points of the second step of the register
        pts_step3: list
            Points of the third step of the register
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # STEP 1
        xpts1, ypts1, zpts1 = list(), list(), list()
        for i in range(len(pts_step1)):
            xpts1.append(pts_step1[i][0])
            ypts1.append(pts_step1[i][1])
            zpts1.append(pts_step1[i][2])

        X1 = np.asarray(xpts1)
        Y1 = np.asarray(ypts1)
        Z1 = np.asarray(zpts1)
        ax.scatter(X1, Y1, Z1, s=size, c=color1, marker='o')

        # STEP 2
        xpts2, ypts2, zpts2 = list(), list(), list()
        for i in range(len(pts_step2)):
            xpts2.append(pts_step2[i][0])
            ypts2.append(pts_step2[i][1])
            zpts2.append(pts_step2[i][2])

        X2 = np.asarray(xpts2)
        Y2 = np.asarray(ypts2)
        Z2 = np.asarray(zpts2)
        ax.scatter(X2, Y2, Z2, s=size, c=color2, marker='o')

        # STEP 3
        xpts3, ypts3, zpts3 = list(), list(), list()
        for i in range(len(pts_step3)):
            xpts3.append(pts_step3[i][0])
            ypts3.append(pts_step3[i][1])
            zpts3.append(pts_step3[i][2])

        X3 = np.asarray(xpts3)
        Y3 = np.asarray(ypts3)
        Z3 = np.asarray(zpts3)
        ax.scatter(X3, Y3, Z3, s=size, c=color3, marker='o')

        pts_steps = list()
        pts_steps.append(pts_step1)
        pts_steps.append(pts_step2)
        pts_steps.append(pts_step3)
        pts = list(itertools.chain.from_iterable(pts_steps))

        xpts, ypts, zpts = list(), list(), list()
        for i in range(len(pts)):
            xpts.append(pts[i][0])
            ypts.append(pts[i][1])
            zpts.append(pts[i][2])

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
            ax.plot([xb], [yb], [zb], 'w')

        plt.show()

    def alphaShape(self, left_pts=None, right_pts=None, left_color='#00FFFF', right_color='#FF00FF', alpha=4, opacity=1.0, online=False):
        """
        Execute the alpha-shape algorithm using the plotly library

        Parameters:
        left_pts: list
            Points of the left lung
        right_pts: list
            Points of the right lung
        left_color: string
            Color of the left lung to be plotted
        right_color: string
            Color of the right lung to be plotted
        alpha: int
            Determines how the mesh surface triangles are derived from the set of vertices (points).
            If "-1", Delaunay triangulation is used
            If "0", the convex-hull algorithm is used
            If ">0", the alpha-shape algorithm is used. Its value acts as the parameter for the mesh fitting.
        opacity: int (value must be between or equal to 0 and 1)
            Sets the opacity of the surface
        online: bool
            If True, plot in online mode, otherwise, plot in offline mode using plotly library
        """
        if online:
            py.sign_in('PythonAPI', 'ubpiol2cve')
            # py.sign_in('hgalon', 'gK9wn7glYHGnkS6s4J8T')

        def executeAlphaShape(points, color):
            x, y, z = list(), list(), list()
            for i in range(len(points)):
                x.append(points[i][0])
                y.append(points[i][1])
                z.append(points[i][2])

            trace = go.Mesh3d(x=x, y=y, z=z,
                              alphahull=alpha,
                              color=color,
                              opacity=opacity)

            return trace

        if left_pts is not None and right_pts is None:
            trace_left = executeAlphaShape(left_pts, left_color)

            data = [trace_left]
            layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
            fig = dict(data=data, layout=layout)

        elif left_pts is None and right_pts is not None:
            trace_right = executeAlphaShape(right_pts, right_color)

            data = [trace_right]
            layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
            fig = dict(data=data, layout=layout)

        elif left_pts is not None and right_pts is not None:
            trace_left = executeAlphaShape(left_pts, left_color)
            trace_right = executeAlphaShape(right_pts, right_color)

            data = [trace_left, trace_right]
            layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
            fig = dict(data=data, layout=layout)

        else:
            print("Invalid Option!")

        if online:
            py.plot(fig)
        else:
            offline.plot(fig)

    def pointCloud(self, left_pts=None, right_pts=None, left_color='#00FFFF', right_color='#FF00FF', online=False):
        """
        Plot the point cloud of the lungs

        Parameters:
        left_pts: list
            Points of the left lung
        right_pts: list
            Points of the right lung
        left_color: string
            Color of the left lung to be plotted
        right_color: string
            Color of the right lung to be plotted
        online: bool
            If True, plot in online mode, otherwise, plot in offline mode using plotly library
        """

        if online:
            py.sign_in('PythonAPI', 'ubpiol2cve')
            # py.sign_in('hgalon', 'gK9wn7glYHGnkS6s4J8T')

        def defineTrace(points):
            pointsize = 1
            x, y, z = list(), list(), list()
            for i in range(len(points)):
                x.append(points[i][0])
                y.append(points[i][1])
                z.append(points[i][2])

            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=pointsize,
                    line=dict(
                        # color='rgba(217, 217, 217, 0.14)',
                        # color='#FFFF00',
                        width=0.5
                    ),
                    # opacity=0.8
                )
            )

            return trace

        if left_pts is not None and right_pts is None:
            trace_left = defineTrace(left_pts)

            data = [trace_left]
            layout = go.Layout(
                scene=dict(
                    xaxis=dict(
                        title='X'),
                    yaxis=dict(
                        title='Y'),
                    zaxis=dict(
                        title='Z'),),
                # width=700,
                margin=dict(l=0, r=0, b=0, t=0))
            fig = go.Figure(data=data, layout=layout)

        elif left_pts is None and right_pts is not None:
            trace_right = defineTrace(right_pts)

            data = [trace_right]
            layout = go.Layout(
                scene=dict(
                    xaxis=dict(
                        title='X'),
                    yaxis=dict(
                        title='Y'),
                    zaxis=dict(
                        title='Z'),),
                # width=700,
                margin=dict(l=0, r=0, b=0, t=0))
            fig = go.Figure(data=data, layout=layout)

        elif left_pts is not None and right_pts is not None:
            trace_left = defineTrace(left_pts)
            trace_right = defineTrace(right_pts)

            data = [trace_left, trace_right]
            layout = go.Layout(
                scene=dict(
                    xaxis=dict(
                        title='X'),
                    yaxis=dict(
                        title='Y'),
                    zaxis=dict(
                        title='Z'),),
                # width=700,
                margin=dict(l=0, r=0, b=0, t=0))
            fig = dict(data=data, layout=layout)

        else:
            print("Invalid Option!")

        if online:
            py.plot(fig)
        else:
            offline.plot(fig)

    def alphaShape_pointCloud(self, left_pts=None, right_pts=None, left_color='#00FFFF', right_color='#FF00FF', alpha=4, opacity=1.0, online=False):
        if online:
            py.sign_in('PythonAPI', 'ubpiol2cve')
            # py.sign_in('hgalon', 'gK9wn7glYHGnkS6s4J8T')

        def defineTrace(points):
            pointsize = 1
            x, y, z = list(), list(), list()
            for i in range(len(points)):
                x.append(points[i][0])
                y.append(points[i][1])
                z.append(points[i][2])

            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=pointsize,
                    line=dict(
                        # color='rgba(217, 217, 217, 0.14)',
                        # color='#FFFF00',
                        width=0.5
                    ),
                    # opacity=0.8
                )
            )

            return trace

        def executeAlphaShape(points, color):
            x, y, z = list(), list(), list()
            for i in range(len(points)):
                x.append(points[i][0])
                y.append(points[i][1])
                z.append(points[i][2])

            trace = go.Mesh3d(x=x, y=y, z=z,
                              alphahull=alpha,
                              color=color,
                              opacity=opacity)

            return trace

        if left_pts is not None and right_pts is None:
            trace_point_cloud = defineTrace(left_pts)
            trace_alpha_shape = executeAlphaShape(left_pts, left_color)

            data = [trace_point_cloud, trace_alpha_shape]
            layout = go.Layout(
                scene=dict(
                    xaxis=dict(
                        title='X'),
                    yaxis=dict(
                        title='Y'),
                    zaxis=dict(
                        title='Z'),),
                # width=700,
                margin=dict(l=0, r=0, b=0, t=0))
            fig = go.Figure(data=data, layout=layout)

        elif left_pts is None and right_pts is not None:
            trace_point_cloud = defineTrace(right_pts)
            trace_alpha_shape = executeAlphaShape(right_pts, right_color)

            data = [trace_point_cloud, trace_alpha_shape]
            layout = go.Layout(
                scene=dict(
                    xaxis=dict(
                        title='X'),
                    yaxis=dict(
                        title='Y'),
                    zaxis=dict(
                        title='Z'),),
                # width=700,
                margin=dict(l=0, r=0, b=0, t=0))
            fig = go.Figure(data=data, layout=layout)

        elif left_pts is not None and right_pts is not None:
            left_trace_point_cloud = defineTrace(left_pts)
            left_trace_alpha_shape = executeAlphaShape(left_pts, left_color)

            right_trace_point_cloud = defineTrace(right_pts)
            right_trace_alpha_shape = executeAlphaShape(right_pts, right_color)

            data = [left_trace_point_cloud, left_trace_alpha_shape, right_trace_point_cloud, right_trace_alpha_shape]
            layout = go.Layout(
                scene=dict(
                    xaxis=dict(
                        title='X'),
                    yaxis=dict(
                        title='Y'),
                    zaxis=dict(
                        title='Z'),),
                # width=700,
                margin=dict(l=0, r=0, b=0, t=0))
            fig = go.Figure(data=data, layout=layout)

        else:
            print("Invalid Option!")

        if online:
            py.plot(fig)
        else:
            offline.plot(fig)
