#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dicom
import pylab
import os
import os.path


class ConvertDICOM2JPG(object):

    """ Convert DICOM files to .jpg """

    def __init__(self, patient, plan, sequence):
        self.path = "/home/handrey/Documents/UDESC/DICOM"
        self.patient = patient
        self.plan = plan
        self.sequence = sequence

        self.folder = "{}/{}/{}/{}".format(
            self.path, self.patient, self.plan, self.sequence)

    def countFiles(self):
        folder = "{}/{}/{}/{}/".format(
            self.path, self.patient, self.plan, self.sequence)

        path = [os.path.join(folder, name) for name in os.listdir(folder)]
        files = [arq for arq in path if os.path.isfile(arq)]
        dcms = [arq for arq in files if arq.lower().endswith(".dcm")]

        return dcms

    def convertDICOM2JPG(self):
        img_num = 1

        for i in range(len(self.countFiles())):
            if img_num < 10:
                img = dicom.read_file(
                    '{}/IM_0000{}.dcm'.format(
                        self.folder,
                        img_num))

                pylab.imsave(
                    '{}/IM_0000{}.jpg'.format(
                        self.folder,
                        img_num),
                    img.pixel_array,
                    cmap=pylab.cm.bone)
            elif img_num >= 10 and img_num < 100:
                img = dicom.read_file(
                    '{}/IM_000{}.dcm'.format(
                        self.folder,
                        img_num))

                pylab.imsave(
                    '{}/IM_000{}.jpg'.format(
                        self.folder,
                        img_num),
                    img.pixel_array,
                    cmap=pylab.cm.bone)
            elif img_num >= 100 and img_num < 1000:
                img = dicom.read_file(
                    '{}/IM_00{}.dcm'.format(
                        self.folder,
                        img_num))

                pylab.imsave(
                    '{}/IM_00{}.jpg'.format(
                        self.folder,
                        img_num),
                    img.pixel_array,
                    cmap=pylab.cm.bone)
            else:
                img = dicom.read_file(
                    '{}/IM_0{}.dcm'.format(
                        self.folder,
                        img_num))

                pylab.imsave(
                    '{}/IM_0{}.jpg'.format(
                        self.folder,
                        img_num),
                    img.pixel_array,
                    cmap=pylab.cm.bone)

            img_num = img_num + 1

        img_num = 1


if __name__ == '__main__':
    patient = raw_input("Patient's name: ")
    plan = raw_input("Sagittal | Coronal: ")
    sequence = raw_input("Sequence number: ")

    convert = ConvertDICOM2JPG(patient, plan, sequence)
    convert.convertDICOM2JPG()
