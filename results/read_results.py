#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_plot_info(filename):
    dataset = open('{}.txt'.format(filename), 'r').read().split('\n')
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


if __name__ == '__main__':
    ds = get_plot_info('9-4')
    print("{} ({})\n\n{} ({})\n".format(ds[0], len(ds[0]), ds[1], len(ds[1])))
