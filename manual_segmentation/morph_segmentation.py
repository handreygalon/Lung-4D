#!/usr/bin/env python
# -*- coding: utf-8 -*-


import Image
import cv2
import pymorph as mm
import numpy as np
from adpil import *
from ia636 import *
import os
import sys


class MMLungSegmentation(object):
    def __init__(self):
        self.img_dir = '.'

    def alternative_solution(self, a, orientation='coronal', linethickness=10, outimg=False):
        '''
        Paramenters
        -----------
        a: original image in graylevel
        '''
        H, W = a.shape
        if orientation == 'coronal':
            # UL = mm.limits(a)[1]  # upper limit
            UL = 255

            b = 1 - iacircle(a.shape, H / 3, (1.4 * H / 3, W / 2))  # Circle
            b = b[0:70, W / 2 - 80:W / 2 + 80]  # Rectangle
            # if outimg:
            #     b_ = 0 * a; b_[0:70, W / 2 - 80:W / 2 + 80] = UL * b  # b_ only for presentation
            #     b_[:, W / 2 - linethickness / 2:W / 2 + linethickness / 2] = UL  # b_ only for presentation

            c = a + 0
            c[:, W / 2 - linethickness / 2:W / 2 + linethickness / 2] = UL
            c[0:70, W / 2 - 80:W / 2 + 80] = (1 - b) * c[0:70, W / 2 - 80:W / 2 + 80] + b * UL
            c[0:40, W / 2 - 70:W / 2 + 70] = UL

            d = mm.open(c, mm.img2se(mm.binary(np.ones((20, 10)))))

            e = mm.close(d, mm.seline(5))

            f = mm.close_holes(e)

            g = mm.subm(f, d)

            h = mm.close_holes(g)

            i = mm.areaopen(h, 1000)

            j1, j2 = iaotsu(i)
            # j = i > j1
            ret, j = cv2.threshold(
                cv2.GaussianBlur(i, (7, 7), 0),
                j1, 255,
                cv2.THRESH_BINARY)

            k = mm.open(j, mm.seline(20, 90))

            l = mm.areaopen(k, 1000)

            # m = mm.label(l)

            res = np.vstack([
                np.hstack([c, d, e, f, g]),
                np.hstack([h, i, j, k, l])
            ])
            cv2.imshow('Result', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            ################################
            # l_ = mm.blob(k,'AREA','IMAGE')
            # l = l_ == max(ravel(l_))

            # m = mm.open(l, mm.sedisk(3))  # VERIFICAR O MELHOR ELEMENTO ESTRUTURANTE AQUI

            # n = mm.label(m)

            if outimg:
                if not os.path.isdir('outimg'):
                    os.mkdir('outimg')

                def N(x):
                    # y = uint8(ianormalize(x, (0, 255)) + 0.5)
                    y = (ianormalize(x, (0, 255)) + 0.5).astype(np.uint8)
                    return y
                adwrite('outimg/a.png', N(a))
                adwrite('outimg/b.png', N(b_))
                adwrite('outimg/c.png', N(c))
                adwrite('outimg/d.png', N(d))
                adwrite('outimg/e.png', N(e))
                adwrite('outimg/f.png', N(f))
                adwrite('outimg/g.png', N(g))
                adwrite('outimg/h.png', N(h))
                adwrite('outimg/i.png', N(i))
                adwrite('outimg/j.png', N(j))
                adwrite('outimg/k.png', N(k))
                adwrite('outimg/l.png', N(l))
                adwrite('outimg/m.png', N(m))
                # adwrite('outimg/n.png', N(n))

            return m

        else:
            b = mm.areaopen(a, 500)

            c = mm.close(b, mm.sebox(3))

            d = mm.close_holes(c)

            e = mm.subm(d, c)

            f = mm.areaopen(e, 1000)

            # g = f > 5
            ret, g = cv2.threshold(
                cv2.GaussianBlur(f, (5, 5), 0),
                3, 255,
                cv2.THRESH_BINARY)
            # ret, g = cv2.threshold(
            #     cv2.GaussianBlur(f, (7, 7), 0),
            #     5, 255,
            #     cv2.THRESH_BINARY_INV)

            h = mm.asf(g, 'CO', mm.sedisk(5))

            i = mm.close_holes(h)

            res = np.vstack([
                np.hstack([a, b, c, d, e]),
                np.hstack([f, g, h, i, a])
            ])
            cv2.imshow('Result', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if outimg:
                if not os.path.isdir('outimg'):
                    os.mkdir('outimg')

                def N(x):
                    y = (ianormalize(x, (0, 255)) + 0.5).astype(np.uint8)
                    return y
                adwrite('outimg/a.png', N(a))
                adwrite('outimg/b.png', N(b))
                adwrite('outimg/c.png', N(c))
                adwrite('outimg/d.png', N(d))
                adwrite('outimg/e.png', N(e))
                adwrite('outimg/f.png', N(f))
                adwrite('outimg/g.png', N(g))
                adwrite('outimg/h.png', N(h))
                adwrite('outimg/i.png', N(i))

            return i

    def dicom2array(self, filename):
        import dicom
        # import Image
        # import pylab
        ds = dicom.read_file(filename)
        # pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
        # pylab.show()
        im = Image.fromarray(ds.pixel_array, 'I;16')  # I (32-bit signed integer pixels)
        # im.show()
        pix = np.array(im.getdata()).reshape(im.size[0], im.size[1], 1)
        pix = pix.reshape(pix.shape[0], pix.shape[1])
        # iashow(pix)
        return pix

    def colorize_segmentation(self, f_, g, k=0.15, t=3):  # 25jan2013
        '''
        Parameters
        ----------
        f: original image;
        g: labeled segmentation;
        k: transparency level;
        t: thickness
        '''

        # f = uint8(ianormalize(f_, (0, 255)) + 0.5)
        f = (ianormalize(f_, (0, 255)) + 0.5).astype(np.uint8)

        # g = g-1 #remove background if watershed was used

        z = mm.lblshow(g)
        # z = uint8(ianormalize(z,(0,255))+0.5)
        # z = (ianormalize(z,(0,255))+0.5).astype(np.uint8)

        zc = mm.gradm(z, mm.secross(0), mm.secross(t))

        # Regions:
        m = z[0] == 0
        z[0] = m * f + k * (1 - m) * z[0] + (1 - k) * (1 - m) * f
        m = z[1] == 0
        z[1] = m * f + k * (1 - m) * z[1] + (1 - k) * (1 - m) * f
        m = z[2] == 0
        z[2] = m * f + k * (1 - m) * z[2] + (1 - k) * (1 - m) * f
        # Contours:
        m = zc[0] == 0
        z[0] = m * z[0] + (1 - m) * zc[0]
        m = zc[1] == 0
        z[1] = m * z[1] + (1 - m) * zc[1]
        m = zc[2] == 0
        z[2] = m * z[2] + (1 - m) * zc[2]
        return z


if __name__ == '__main__':

    try:
        # filename_list = [sys.argv[1]]
        optype = 'sagittal'
        opthickness = 5
        opnmasks = 1

        patient = 'Iwasawa'
        plan = optype.capitalize()
        sequence = 5

        txtargv2 = '-type={}'.format(optype)
        txtargv3 = '-thickness={}'.format(opthickness)
        txtargv4 = '-nmasks={}'.format(opnmasks)
        txtargv5 = '-patient={}'.format(patient.capitalize())
        txtargv6 = '-sequence={}'.format(sequence)

        if len(sys.argv) > 2:
            txtargv2 = sys.argv[2]
            if len(sys.argv) > 3:
                txtargv3 = sys.argv[3]
                if len(sys.argv) > 4:
                    txtargv4 = sys.argv[4]
                    if len(sys.argv) > 5:
                        txtargv5 = sys.argv[5]
                        if len(sys.argv) > 6:
                            txtargv6 = sys.argv[6]
        txtargv = txtargv2 + '|' + txtargv3 + '|' + txtargv4 + '|' + txtargv5 + '|' + txtargv6

        if txtargv.find('-type') != -1:
            txttmp = txtargv.split('-type')[1]
            txttmp = txttmp.split('=')[1]
            optype = txttmp.split('|')[0]

        if txtargv.find('-thickness') != -1:
            txttmp = txtargv.split('-thickness')[1]
            txttmp = txttmp.split('=')[1]
            opthickness = int(txttmp.split('|')[0])

        if txtargv.find('-nmasks') != -1:
            txttmp = txtargv.split('-nmasks')[1]
            txttmp = txttmp.split('=')[1]
            opnmasks = int(txttmp.split('|')[0])

        if txtargv.find('-patient') != -1:
            txttmp = txtargv.split('-patient')[1]
            txttmp = txttmp.split('=')[1]
            patient = txttmp.split('|')[0]

        if txtargv.find('-sequence') != -1:
            txttmp = txtargv.split('-sequence')[1]
            txttmp = txttmp.split('=')[1]
            sequence = int(txttmp.split('|')[0])

        DIR_orig = '/home/handrey/Documents/UDESC/JPG/{}/{}/{}'\
            .format(patient, plan, sequence)
        DIR_dest = '/home/handrey/Documents/UDESC/segmented/{}/{}/{}'\
            .format(patient, plan, sequence)

        filename_list = os.listdir(DIR_orig)
        filename_list.sort()
    except:
        print '''
        Examples of use:
        $ python %s images/IM_00001.dcm -type=coronal -nmasks=2
        $ python %s images/MR0001.dcm -type=sagittal -thickness=7

        $ python %s -type=sagittal -thickness=7 -patient=iwasawa -sequence=5
        $ python %s -type=coronal -nmasks=2 -patient=iwasawa -sequence=5
        ''' % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])
        exit()

    mm_segmentation = MMLungSegmentation()

    for i in range(len(filename_list)):
        print('{}/IM ({}).jpg'.format(DIR_orig, i + 1))

        img = cv2.imread('{}/IM ({}).jpg'.format(DIR_orig, i + 1), 0)
        # im = im.astype(np.uint8)

        print('Filename: {} - Shape: {}'.format('IM ({}).jpg'.format(i + 1), img.shape))

        img_out = mm_segmentation.alternative_solution(img, optype, opthickness)
        cv2.imwrite('{}/lungIM ({}).png'.format(DIR_dest, i + 1), img_out)

        # tmp = mm.gshow(img, img_out)
        # tmp = colorize_segmentation(img, img_out)
        # num = filename[:-4]
        # adwrite(num + '_out.png', tmp)
        # adwrite(num + '_out.png', img_out)

        if optype == 'coronal' and opnmasks == 2:
            u, i, j = iaunique(ravel(img_out))
            tmp1 = mm.gradm(img_out == u[1], mm.sedisk(opthickness), mmsedisk(opthickness))
            tmp2 = mm.gradm(img_out == u[2], mm.sedisk(opthickness), mmsedisk(opthickness))
            s1 = sum(tmp1, 0)
            s2 = sum(tmp2, 0)
            s = mm.subm(s1, s2)
            meio = len(s) / 2
            if sum(s[0:meio]) > sum(s[meio::]):
                tmpR = tmp1
                tmpL = tmp2
            else:
                tmpR = tmp2
                tmpL = tmp1
            # adwrite(num + '_maskL.png', 255 * tmpL)
            # adwrite(num + '_maskR.png', 255 * tmpR)
            adwrite('{}/maskLIM ({}).png'.format(DIR_dest, i + 1), 255 * tmpL)
            adwrite('{}/maskRIM ({}).png'.format(DIR_dest, i + 1), 255 * tmpR)
        else:
            print("Write image")
            tmp = mm.gradm(img_out > 0, mm.sedisk(opthickness), mm.sedisk(opthickness))
            # adwrite(num + '_mask.png', 255 * tmp)
            adwrite('{}/maskIM ({}).png'.format(DIR_dest, i + 1), tmp)
