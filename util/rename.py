#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path

from constant import *


def add_extension(filenames):
    for filename in filenames:
        os.rename(
            os.path.join(DIR, filename),
            os.path.join(DIR, filename + ".dcm"))


def change_names(filenames, img_type):
    sorted_files = sorted(filenames)
    # lfiles = []
    # for filename in filenames:
    #     lfiles.append(int(filename.split('.')[0].split('_')[1]))
    # ordered_files = sorted(lfiles)

    if img_type == 'jpg':
        for i in range(len(sorted_files)):
            if i + 1 < 10:
                name = 'IM ({}).jpg'.format(i + 1)
            else:
                name = 'IM ({}).jpg'.format(i + 1)
            print('{} -> {}'.format(sorted_files[i], name))
            os.rename(
                os.path.join(DIR, sorted_files[i]),
                os.path.join(DIR, name))
    else:
        for i in range(len(sorted_files)):
            if i + 1 < 10:
                name = 'IM_0000{}.dcm'.format(i + 1)
            else:
                name = 'IM_000{}.dcm'.format(i + 1)
            print('{} -> {}'.format(sorted_files[i], name))
            os.rename(
                os.path.join(DIR, sorted_files[i]),
                os.path.join(DIR, name))


if __name__ == '__main__':
    patient = 'Matsushita'
    plan = 'Coronal'
    sequence = 22

    img_type = 'jpg'
    # img_type = 'dcm'
    if img_type == 'jpg':
        DIR =\
            '{}/{}/{}/{}'.format(DIR_JPG, patient, plan, sequence)
    else:
        DIR =\
            '{}/{}/{}/{}'.format(DIR_DICOM, patient, plan, sequence)

    filenames = os.listdir(DIR)
    # print("{} ({})\n".format(filenames, len(filenames)))
    # sorted_files = sorted(filenames)
    # print("{} ({})\n".format(sorted_files, len(sorted_files)))

    change_names(filenames, img_type)
