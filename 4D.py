#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import itertools
import numpy as np

# import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.io as pio

# from results.read_results import *
from util.constant import *


def read(side, rootsequence, imgnumber):
    """ Read text file that contain information about point cloud """
    if side == 0:
        dataset = open(
            '{}/{}-{}.txt'.format(
                DIR_RESULT_INTERPOLATED_ST_LEFT, rootsequence, imgnumber),
            'r').read().split('\n')
    else:
        dataset = open(
            '{}/{}-{}.txt'.format(
                DIR_RESULT_INTERPOLATED_ST_RIGHT, rootsequence, imgnumber),
            'r').read().split('\n')
    dataset.pop(-1)

    information, instant, points, tuples = list(), list(), list(), list()

    for i in range(len(dataset)):
        string = dataset[i].split('[')
        # Info
        basicinfo = string[1].replace(" ", "").split(',')
        basicinfo.pop(-1)
        instant.append(basicinfo[0][1:-1])
        instant.append(basicinfo[1][1:-1])
        instant.append(int(basicinfo[2]))
        instant.append(int(basicinfo[3]))
        instant.append(int(basicinfo[4]))
        # Points
        string2 = string[2].replace(']', '')
        string3 = string2.replace('), ', ');')
        string4 = string3.split(';')

        for j in range(len(string4)):
            pts = string4[j].split(',')
            tupla = (float(pts[0][1:]), float(pts[1]), float(pts[2][:-1]))
            tuples.append(tupla)

        points.append(tuples)

        instant.append(tuples)
        tuples = []
        information.append(instant)
        instant = []

    return information


def get_points(side, rootsequence, imgnumber):
    if side == 0:
        points = read(
            side=0,
            rootsequence=rootsequence,
            imgnumber=imgnumber)
    else:
        points = read(
            side=1,
            rootsequence=rootsequence,
            imgnumber=imgnumber)

    points1, points2, points3, all_points = list(), list(), list(), list()

    for i in range(len(points)):
        if points[i][4] == 1:
            points1.append(points[i][5])
        elif points[i][4] == 2:
            points2.append(points[i][5])
        else:
            points3.append(points[i][5])

    points1 = list(itertools.chain.from_iterable(points1))
    points2 = list(itertools.chain.from_iterable(points2))
    points3 = list(itertools.chain.from_iterable(points3))

    all_points.append(points1)
    all_points.append(points2)
    all_points.append(points3)
    all_points = list(itertools.chain.from_iterable(all_points))

    return points1, points2, points3, all_points


def separate_points_by_coordinate(pointsstep1, pointsstep2, pointsstep3):
    points = list()

    points.append(pointsstep1)
    points.append(pointsstep2)
    points.append(pointsstep3)
    points = list(itertools.chain.from_iterable(points))

    X, Y, Z = list(), list(), list()

    for i in range(len(points)):
        X.append(points[i][0])
        Y.append(points[i][1])
        Z.append(points[i][2])

    X = tuple(X)
    Y = tuple(Y)
    Z = tuple(Z)

    return X, Y, Z


def point_cloud(X, Y, Z, size=1, color='#FF3232', bordercolor='#FF3232', legend='', width=0.5, opacity=1.0):
    """
    Build the point clound

    Parameters
    ----------
    X, Y, Z: Lists
        Represents X, Y and Z values respectively
    size: int
        Point's size
    color: string
        Point's color
    bordercolor: string
        Border color of points
    legend: string
    width: float
    opacity: float
        Opacity of the trace. The value range from 0 to 1 (default value is 1) """
    point_cloud = go.Scatter3d(
        x=X,
        y=Y,
        z=Z,
        # showlegend=False,
        name=legend,
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            line=dict(
                color=bordercolor,
                width=width
            ),
            # opacity=opacity
        )
    )

    return point_cloud


def alpha_shapes(X, Y, Z, alpha=7, opacity=1.0, color='#FF3232'):
    """
    Execute alpha-shapes algorithm

    Parameters
    ----------
    X, Y, Z: Lists
        Represents X, Y and Z values respectively
    alpha: int
        Options:
            -1: Delaunay triangulation is used
            >0: the alpha-shape algorithm is used (The value acts as the parameter for the mesh fitting)
            0: the convex-hull algorithm is used
    opacity: int
        Opacity of the surface. The value range from 0 to 1 (default value is 1)
    color: string
        Color of the whole mesh

    References
    ----------
    https://plot.ly/python/3d-surface-lighting/ """
    lighting_effects = dict(ambient=0.5, roughness=0.2, diffuse=0.5, fresnel=0.2, specular=0.05)  # Combined effects
    shape = go.Mesh3d(
        x=X,
        y=Y,
        z=Z,
        alphahull=alpha,
        opacity=opacity,
        # lighting=dict(ambient=0.5),  # The value range from 0 to 1 (default value is 0.8)
        # lighting=dict(roughness=0.2),  # The value range from 0 to 1 (by default value is 0.5)
        # lighting=dict(diffuse=0.5),  # The value ranges from 0 to 1 (default value is 0.8).
        # lighting=dict(fresnel=0.2),  # The value can range from 0 to 5 (default value is 0.2).
        # lighting=dict(specular=0.14),  # The value range from 0 to 2 (default value is 0.05)
        lighting=lighting_effects,  # Combined effects
        color=color)

    return shape


def create_temp_images(imgnumber, viewoption):
    """
    Add imagem that represents the respiratory instant (MR image) to the video. For this it is necessary
    to resize the image to be with the same height of the image of the 3D lung.

    Parameters
    ----------
    imgnumber: int
        Image's number (1 - 50). Represent the respiratory instant
    viewoption: int
        Represents which view was chosen (See more at 'views' function)

    References
    ----------
    https://docs.opencv.org/3.4.0/dc/da3/tutorial_copyMakeBorder.html """
    patient = 'Matsushita'  # 'Iwasawa'
    plan = 'Coronal'
    sequence = 14  # 9

    # DIR = '{}/view-{}'.format(DIR_FRAMES, viewoption)

    imagerm = cv2.imread('{}/{}/{}/{}/IM ({}).jpg'.format(DIR_JPG, patient, plan, sequence, imgnumber), cv2.IMREAD_COLOR)
    imagelung = cv2.imread('{}/view-{}/{}.png'.format(DIR_FRAMES, viewoption, imgnumber), cv2.IMREAD_COLOR)
    borderType = cv2.BORDER_CONSTANT
    borderColor = (255, 255, 255)  # White

    height_rm, width_rm = imagerm.shape[:2]
    height_lung, width_lung = imagelung.shape[:2]

    # Build border. Thus, the MR image will have the same height as the 3D image of the lung
    temp = height_lung - height_rm
    top = int(temp / 2)
    bottom = top
    left = 0
    right = 0
    border = cv2.copyMakeBorder(imagerm, top, bottom, left, right, borderType, None, borderColor)
    cv2.imwrite('{}/view-{}/MR_({}).jpg'.format(DIR_FRAMES, viewoption, imgnumber), border)

    # Join images
    image_rm_border = cv2.imread('{}/view-{}/MR_({}).jpg'.format(DIR_FRAMES, viewoption, imgnumber), cv2.IMREAD_COLOR)
    res = np.hstack((image_rm_border, imagelung))
    cv2.imwrite('{}/view-{}/TEMP_({}).jpg'.format(DIR_FRAMES, viewoption, imgnumber), res)


def save_images(figure, imgnumber, viewoption):
    """
    Saves the images of the lung in 3D created by Plotly

    Parameters
    ----------
    figure: dict
        Dictionary with the necessary specifications to plot the figure
    imgnumber: int
        Image's number (1 - 50). Represent the respiratory instant
    viewoption: int
        Represents which view was chosen (See more at 'views' function) """
    if not os.path.exists(DIR_FRAMES):
        os.mkdir('{}'.format(DIR_FRAMES))
    if not os.path.exists('{}/view-{}'.format(DIR_FRAMES, viewoption)):
        os.mkdir('{}/view-{}'.format(DIR_FRAMES, viewoption))
    pio.write_image(figure, '{}/view-{}/{}.png'.format(DIR_FRAMES, viewoption, imgnumber))

    create_temp_images(imgnumber=imgnumber, viewoption=viewoption)


def create_video(velocity, viewoption):
    """
    Gather the images of each respiratory instant and build a video

    Parameters
    ----------
    velocity: int
        Represents the speed of breathing
    viewoption: int
        Represents which view was chosen (See more at 'views' function) """
    if not os.path.exists(DIR_LUNG_4D):
        os.mkdir('{}'.format(DIR_LUNG_4D))

    images = list()

    # Count how many respiratory instants exists
    DIR = "{}/view-{}".format(DIR_FRAMES, viewoption)
    countfiles = len([f for f in os.listdir(DIR) if f.endswith('.png') and os.path.isfile(os.path.join(DIR, f))])
    for i in range(countfiles):
        images.append(cv2.imread('{}/TEMP_({}).jpg'.format(DIR, i + 1)))

    height, width, layers = images[1].shape

    # Build the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('{}/lung4D_view{}.avi'.format(DIR_LUNG_4D, viewoption), fourcc, velocity, (width, height))

    for i in range(countfiles):
        video.write(images[i])

    cv2.destroyAllWindows()
    video.release()

    # Removes images that are no longer needed
    for i in range(countfiles):
        os.remove('{}/view-{}/MR_({}).jpg'.format(DIR_FRAMES, viewoption, i + 1))
        os.remove('{}/view-{}/TEMP_({}).jpg'.format(DIR_FRAMES, viewoption, i + 1))


def shape(rootsequence, side, image, option=0):
    """
    Builds lung shape in 3D

    Parameters
    ----------
    rootsequence: int
        Represents the coronal root sequence
    side: int
        0 (Left lung), 1 (Right lung)
    image: int
        Image's number (1 - 50). Represent the respiratory instant
    option: int
        0 (Just point cloud) | 1 (Alpha shapes) | 2 (Point cloud and alpha shapes) """
    # alpha Iwasawa = 4.5 | alpha Matsushita = 5.5
    lalpha = 5.5
    ralpha = 5.5

    left_points1, left_points2, left_points3, all_left_points =\
        get_points(
            side=0,
            rootsequence=rootsequence,
            imgnumber=image)

    leftX, leftY, leftZ =\
        separate_points_by_coordinate(left_points1, left_points2, left_points3)

    right_points1, right_points2, right_points3, all_right_points =\
        get_points(
            side=1,
            rootsequence=rootsequence,
            imgnumber=image)

    rightX, rightY, rightZ =\
        separate_points_by_coordinate(right_points1, right_points2, right_points3)

    if option == 1:
        lalpha_shapes = alpha_shapes(
            X=leftX,
            Y=leftY,
            Z=leftZ,
            alpha=lalpha,
            # color='#A91818',
            color='#808080',
            opacity=1.0)

        ralpha_shapes = alpha_shapes(
            X=rightX,
            Y=rightY,
            Z=rightZ,
            alpha=ralpha,
            # color='#FF3232',
            color='#808080',
            opacity=1.0)

        return [lalpha_shapes, ralpha_shapes]

    elif option == 2:
        lpoint_cloud = point_cloud(
            X=leftX,
            Y=leftY,
            Z=leftZ,
            legend='RIGHT LUNG',
            size=1,
            color='#000000',
            bordercolor='#000000',
            # color='#FF3232',
            # bordercolor='#FF3232',
            # color='#999999',
            # bordercolor='#999999',
            width=0.5,
            opacity=1.0)

        rpoint_cloud = point_cloud(
            X=rightX,
            Y=rightY,
            Z=rightZ,
            legend='LEFT LUNG',
            size=1,
            color='#000000',
            bordercolor='#000000',
            # color='#701700',
            # bordercolor='#701700',
            # color='#999999',
            # bordercolor='#999999',
            width=0.5,
            opacity=1.0)

        lalpha_shapes = alpha_shapes(
            X=leftX,
            Y=leftY,
            Z=leftZ,
            alpha=lalpha,
            # color='rgba(255, 0, 0, 0.1)',
            color='#808080',
            opacity=1.0)

        ralpha_shapes = alpha_shapes(
            X=rightX,
            Y=rightY,
            Z=rightZ,
            alpha=ralpha,
            # color='rgba(255, 0, 0, 0.1)',
            color='#808080',
            opacity=1.0)

        return [lpoint_cloud, rpoint_cloud, lalpha_shapes, ralpha_shapes]

    else:
        lpoint_cloud = point_cloud(
            X=leftX,
            Y=leftY,
            Z=leftZ,
            legend='RIGHT LUNG',
            size=1,
            # color='#FF3232',
            # bordercolor='#FF3232',
            color='#999999',
            bordercolor='#999999',
            width=0.5,
            opacity=1.0)

        rpoint_cloud = point_cloud(
            X=rightX,
            Y=rightY,
            Z=rightZ,
            legend='LEFT LUNG',
            size=1,
            color='#999999',
            bordercolor='#999999',
            # color='#701700',
            # bordercolor='#701700',
            width=0.5,
            opacity=1.0)

        return [lpoint_cloud, rpoint_cloud]


def views(image, viewoption, figure):
    """
    Parameters
    ----------
    image: int
        Image's number (1 - 50). Represent the respiratory instant
    viewoption: int
        Represents which view was chosen by the user
    figure: dict
        Dictionary with the necessary specifications to plot the figure

    References
    ----------
    https://plot.ly/python/3d-camera-controls/ """

    if viewoption == 1:
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2, y=2, z=0.1)
        )

        figure['layout'].update(
            scene=dict(camera=camera),
            title=name
        )
    elif viewoption == 2:
        # X-Z plane
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.1, y=2.5, z=0.1)
        )

        figure['layout'].update(
            scene=dict(camera=camera),
            title=name
        )
    elif viewoption == 3:
        # Y-Z plane
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2.5, y=0.1, z=0.1)
        )

        figure['layout'].update(
            scene=dict(camera=camera),
            title=name
        )
    elif viewoption == 4:
        # View from above
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.1, y=0.1, z=2.5)
        )

        figure['layout'].update(
            scene=dict(camera=camera),
            title=name
        )
    elif viewoption == 5:
        # Zooming In ... by reducing the norm the eye vector.
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.1, y=0.1, z=1)
        )

        figure['layout'].update(
            scene=dict(camera=camera),
            title=name
        )
    elif viewoption == 6:
        # X-Z plane (other side)
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.2, y=-3.0, z=0.1)
        )

        figure['layout'].update(
            scene=dict(camera=camera),
            title=name
        )
    elif viewoption == 7:
        # X-Z plane (other side)
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=-2.5, z=0.1)
        )

        figure['layout'].update(
            scene=dict(camera=camera),
            title=name
        )
    elif viewoption == 8:
        # X-Z plane (other side)
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.8, y=-3.0, z=-0.8)
        )

        figure['layout'].update(
            scene=dict(camera=camera),
            title=name
        )
    elif viewoption == 9:
        # X-Z plane (other side)
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            # eye=dict(x=1.8, y=-3.0, z=-1.5)
            eye=dict(x=0.8, y=-2.0, z=-0.5)
        )

        figure['layout'].update(
            scene=dict(camera=camera),
            title=name
        )
    elif viewoption == 10:
        # X-Z plane
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.1, y=-2.5, z=0.1)
        )

        figure['layout'].update(
            scene=dict(camera=camera),
            title=name
        )

    else:
        name = 'instant={} view={}.html'.format(imgnumber, viewoption)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )

    return figure


def build4D(rootsequence, side, imgnumber=1, meshoption=0, viewoption=0, show=0, save=0):
    """
    Main function

    Parameters
    ----------
    rootsequence: int
        Represents the coronal root sequence
    side: int
        0 (Left lung), 1 (Right lung)
    imgnumber: int
        Image's number (1 - 50). Represent the respiratory instant
    meshoption: int
        0 (Just point cloud) | 1 (Alpha shapes) | 2 (Point cloud and alpha shapes)
    viewoption: int
        Represents which view was chosen (See more at 'views' function)
    show: int
        Plot the figure
    save: int
        Save the figure """
    # First instant
    data = shape(rootsequence=rootsequence, side=side, image=imgnumber, option=meshoption)

    layout = go.Layout(
        # title='Lung',
        showlegend=False,
        # plot_bgcolor='#FFFF00',
        # paper_bgcolor='#FFFF00',
        # legend=dict(orientation="h"),
        # legend=dict(x=0.35, y=0.05),
        scene=dict(
            xaxis=dict(
                title='X',  # X
                # showgrid=False,
                # showbackground=True,
                # backgroundcolor='rgb(230, 230,230)',
                gridcolor="#adad85",
                gridwidth=3,
                color="#000000",
                # showticklabels=False,
                # tickwidth=10,
                # dtick=10,
                # range=[-150, 150],
                linecolor='#000000',
                linewidth=3,
                # zerolinecolor='rgb(255, 255, 255)',
                zeroline=False),
            yaxis=dict(
                title='Y',  # Y
                # showgrid=False,
                # showbackground=True,
                # backgroundcolor='rgb(230, 230,230)',
                gridcolor="#adad85",
                gridwidth=3,
                color="#000000",
                # showticklabels=False,
                # tickwidth=10,
                # dtick=10,
                # range=[-80, 80],
                linecolor='#000000',
                linewidth=3,
                # zerolinecolor='rgb(255, 255, 255)',
                zeroline=False),
            zaxis=dict(
                title='Z',  # Z
                # showgrid=False,
                # showbackground=True,
                # backgroundcolor='rgb(230, 230,230)',
                gridcolor="#adad85",
                gridwidth=3,
                color="#000000",
                # showticklabels=False,
                # tickwidth=10,
                # dtick=10,
                # range=[-150, 150],
                linecolor='#000000',
                linewidth=3,
                # zerolinecolor='rgb(255, 255, 255)',
                zeroline=False),
            # aspectmode='manual',
            # aspectratio=dict(x=1, y=0.53, z=1)
        ),
        # autosize=False,
        # width=600,
        # height=600,
        # paper_bgcolor="tomato",
        margin=dict(l=0, r=0, b=0, t=0))

    figure = dict(data=data, layout=layout)

    # Change angle of view
    fig = views(image=imgnumber, viewoption=viewoption, figure=figure)

    if save == 1:
        datas = list()

        for i in range(50):
            data = shape(rootsequence=rootsequence, side=side, image=i + 1, option=meshoption)
            datas.append(data)

        for i in range(50):
            figure = dict(data=datas[i], layout=layout)
            save_images(figure=figure, imgnumber=i + 1, viewoption=viewoption)

        create_video(velocity=2, viewoption=viewoption)

    if show == 1:
        offline.plot(fig)
        # offline.plot(fig, filename=name)


def frames4D(rootsequence, side, imgnumber=1, meshoption=0, viewoption=0):
    """ Build 4D model """

    # First instant
    data = shape(rootsequence=rootsequence, side=side, image=imgnumber, option=meshoption)

    # Mesh
    meshes = list()
    meshes.append(data)

    for i in range(2, 51):
        mesh = shape(rootsequence=rootsequence, side=side, image=i, option=meshoption)
        meshes.append(mesh)
    # print(meshes[0][0]['x'][0])

    frames = list()
    for i in range(len(meshes)):
        d = dict()
        d['data'] = meshes[i]
        d['layout'] = {'title': "{}".format(i + 1)}
        frames.append(d)
    '''
    frames = [{'data': meshes[0], 'layout': {'title': '1'}},
              {'data': meshes[1], 'layout': {'title': '2'}},
              ...
              {'data': meshes[48], 'layout': {'title': '49'}},
              {'data': meshes[49], 'layout': {'title': '50'}},
    '''

    layout = go.Layout(
        title='Lung',
        # showlegend=False,
        # legend=dict(orientation="h"),
        # legend=dict(x=0.35, y=0.05),
        scene=dict(
            xaxis=dict(
                title='X-AXIS'),
            yaxis=dict(
                title='Y-AXIS'),
            zaxis=dict(
                title='Z-AXIS'),),
        # width=700,
        margin=dict(l=0, r=0, b=0, t=0))

    # make figure
    # figure = {
    #     'data': [],
    #     'layout': {},
    #     'frames': []
    # }
    figure = dict(data=data, layout=layout, frames=frames)

    # Change angle of view
    fig = views(image=imgnumber, viewoption=viewoption, figure=figure)

    offline.plot(fig)
    # offline.plot(figure, filename=name)


if __name__ == '__main__':
    try:
        patient = 'Matsushita'  # 'Iwasawa'
        rootsequence = 14  # 9
        side = 0
        imgnumber = 1
        meshoption = 0  # 0 (Just point cloud) | 1 (Alpha shapes) | 2 (Point cloud and alpha shapes)
        viewoption = 9
        show = 0
        save = 0

        txtargv2 = '-patient={}'.format(patient)
        txtargv3 = '-rootsequence={}'.format(rootsequence)
        txtargv4 = '-side={}'.format(side)
        txtargv5 = '-imgnumber={}'.format(imgnumber)
        txtargv6 = '-meshoption={}'.format(meshoption)
        txtargv7 = '-viewoption={}'.format(viewoption)
        txtargv8 = '-show={}'.format(show)
        txtargv9 = '-save={}'.format(save)

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

        txtargv = '{}|{}|{}|{}|{}|{}|{}|{}'.format(
            txtargv2,
            txtargv3,
            txtargv4,
            txtargv5,
            txtargv6,
            txtargv7,
            txtargv8,
            txtargv9)

        if txtargv.find('-patient') != -1:
            txttmp = txtargv.split('-patient')[1]
            txttmp = txttmp.split('=')[1]
            patient = txttmp.split('|')[0]

        if txtargv.find('-rootsequence') != -1:
            txttmp = txtargv.split('-rootsequence')[1]
            txttmp = txttmp.split('=')[1]
            rootsequence = int(txttmp.split('|')[0])

        if txtargv.find('-side') != -1:
            txttmp = txtargv.split('-side')[1]
            txttmp = txttmp.split('=')[1]
            side = int(txttmp.split('|')[0])

        if txtargv.find('-imgnumber') != -1:
            txttmp = txtargv.split('-imgnumber')[1]
            txttmp = txttmp.split('=')[1]
            imgnumber = int(txttmp.split('|')[0])

        if txtargv.find('-meshoption') != -1:
            txttmp = txtargv.split('-meshoption')[1]
            txttmp = txttmp.split('=')[1]
            meshoption = int(txttmp.split('|')[0])

        if txtargv.find('-viewoption') != -1:
            txttmp = txtargv.split('-viewoption')[1]
            txttmp = txttmp.split('=')[1]
            viewoption = int(txttmp.split('|')[0])

        if txtargv.find('-show') != -1:
            txttmp = txtargv.split('-show')[1]
            txttmp = txttmp.split('=')[1]
            show = int(txttmp.split('|')[0])

        if txtargv.find('-save') != -1:
            txttmp = txtargv.split('-save')[1]
            txttmp = txttmp.split('=')[1]
            save = int(txttmp.split('|')[0])

    except ValueError:
        print(
            """
            Example of use:\n

            $ python {} -patient=Iwasawa -rootsequence=9 -side=0 -imgnumber=1
            -meshoption=1 -viewoption=9 -show=1 -save=0
            """.format(sys.argv[0]))
        exit()

    build4D(
        rootsequence=rootsequence,
        side=side,
        imgnumber=imgnumber,
        meshoption=meshoption,
        viewoption=viewoption,
        show=show,
        save=save)

    # frames4D(
    #     rootsequence=rootsequence,
    #     side=side,
    #     imgnumber=imgnumber,
    #     meshoption=meshoption,
    #     viewoption=viewoption)
