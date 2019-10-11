#!/usr/bin/env python3

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import astropy.io.fits as fits
from numpy.fft import fft, ifft, fft2, ifft2
from math import sin
import os

def fft_array(channel):
    fft = np.fft.fft2(channel)
    fft = np.fft.fftshift(fft)

    fft = np.absolute(fft)
    return fft
    #return np.real(fft)

def getfits(f):
    return fits.getdata(f)

def writefits(f, arr):
    if os.path.exists(f):
        os.remove(f)
    fits.writeto(f, arr)

def fix_psf(psf):
    """ Увеличивает чётную psf до такого размера,
    чтобы максимальный пиксель находился ровно по центру
    размер итоговой картинки получается НЕчётный
    пока работает не идеально
    """
    shape = psf.shape
    mask = np.zeros(shape)
    mask[shape[0]//4:shape[0]//4*3,shape[1]//4:shape[1]//4*3] =1.
    masked_psf = psf*mask
    maxpoint = np.unravel_index(masked_psf.argmax(), psf.shape)
    mpx, mpy = maxpoint[:2]
        
    new_shape = (mpx * 2 + 1 - psf.shape[0], mpy * 2 + 1 - psf.shape[1])
    return np.pad(psf, [(0, new_shape[0]), (0, new_shape[1])], mode="constant")

def expand_psf(model_shape, psf):
    """Увеличивает PSF до размера 2 * исходное изображение
    и располагает максимальный пиксель по центру, разумеется
    снаружи - нули"""
    
    psf_shape_y, psf_shape_x = psf.shape
    y, x = model_shape

    add_y = (2 * y - psf_shape_y) // 2 + 1
    add_x = (2 * x - psf_shape_x) // 2 + 1

    psfnew = np.pad(psf, [(add_y, add_y), (add_x, add_x)], mode="constant")
    return psfnew

def make_model_bigpsf(picture, psf):
    """ В расчёте на то, что psf нечётная, и
    самый яркий пиксель строго по центру """

    picture_shape = picture.shape
    
    step_y = (psf.shape[0] - 1) // 2
    step_x = (psf.shape[1] - 1) // 2
    psfshift_y = np.roll(psf, step_y, axis = 0)
    psf_rolled = np.roll(psfshift_y, step_x, axis = 1)

    reshape_y = (psf.shape[0] - picture.shape[0])
    reshape_x = (psf.shape[1] - picture.shape[1])
    picture = np.pad(picture, [(0, reshape_y), (0, reshape_x)], mode="constant")

    ftx = fft2(picture)
    fty = fft2(psf_rolled)
    
    convoluted = np.real(ifft2(ftx*fty))
    convoluted = convoluted[0:picture_shape[0], 0:picture_shape[1]]
    return convoluted

def make_psf_for(size):
    """size в формате y, x
    выдаёт картинку в 2 раза больше исходной
    с самым ярким пикселем в точке y, x, начиная с нуля
    """

    size = tuple(reversed(size))
    
    def psffunction(x, y):
        n = 50
        x -= size[0]
        y -= size[1]

        x /= size[0]
        y /= size[1]

        if sin(x) == 0:
            px = 1
        else:
            px = sin(n*x)/(n*sin(x))

        if sin(y) == 0:
            py = 1
        else:
            py = sin(n*y)/(n*sin(y))

        pixel = px * py
        return pixel

    psfshape = tuple((i * 2 + 1 for i in size))
    
    psf = np.zeros(psfshape)

    for y in range(0, psf.shape[0]):
        for x in range(0, psf.shape[1]):
            psf[y][x] = psffunction(x, y)

    psf /= np.sum(psf)
    return psf

def makeClean(dirtysource, psf, cleanblurradius, bottomlimit, maxit, gamma = 0.1, criticalbottom = None):
    psfmax = psf.max()
    
    psfk, psfm = [ int((x-1) / 2) for x in psf.shape]
    imgx, imgy = dirtysource.shape
    imgbx, imgby = (int(imgx + 2*psfk), int(imgy + 2*psfm))

    img2big = Image.new("F", (imgbx, imgby), 0)
    img2big.paste(Image.fromarray(dirtysource), (psfk, psfm))

    sourcebytes = np.array(img2big, dtype=np.float64)
    totalmax = sourcebytes.max()

    if criticalbottom is None:
        criticalbottom = bottomlimit * totalmax

    dirtysum = np.sum(sourcebytes)


    cleanPoints = np.ones((imgbx, imgby))

    psfsum = np.sum(psf)
    
    it = 0
    while it < maxit:
        maxpoint = np.unravel_index(sourcebytes.argmax(), sourcebytes.shape)
        y, x = maxpoint[:2]
        mpy, mpx = maxpoint[:2]

        maxvalue = sourcebytes[y][x]
        if maxvalue <= criticalbottom:
            break
        print(maxvalue)

        k = gamma * maxvalue / psfmax

        py = 0
        for y in range(mpy - psfm, mpy + psfm + 1):
            px = 0
            for x in range(mpx - psfk, mpx + psfk + 1):
                if y >= imgby or x >= imgbx or y < 0 or x < 0:
                    px += 1
                    continue

                #if (psf[py][px] == psfmax) and (sourcebytes[y][x] == maxvalue):
                #    print("success")
                sourcebytes[y][x] -= psf[py][px] * k
                px += 1
            py += 1

        cleanPoints[mpy][mpx] += psfsum * k

        if (it % 10) == 0:
            cleansum = np.sum(cleanPoints)
            dirtysum2 = np.sum(sourcebytes)
            print("full = {0}, sourcepic = {1}, it = {2}"
                  .format(cleansum + dirtysum2, dirtysum, it))

        it += 1
    print("iterations: {0}".format(it))
    
    cleanPoints = gaussian_filter(cleanPoints, sigma=cleanblurradius)
    cleanImage = Image.fromarray(cleanPoints)
    dirtyOutput = Image.fromarray(sourcebytes)

    cleanImage = cleanImage.crop((psfk, psfm, imgx + psfk, imgy + psfm))
    dirtyOutput = dirtyOutput.crop((psfk, psfm, imgx + psfk, imgy + psfm))

    return np.array(cleanImage), np.array(dirtyOutput)

def makeCleanBigPSF(dirtysource, psf, cleanblurradius, bottomlimit, maxit, gamma = 0.1, criticalbottom = None):
    psfmy, psfmx = np.unravel_index(psf.argmax(), psf.shape)[:2]

    imgy, imgx = dirtysource.shape
    totalmax = dirtysource.max()

    if criticalbottom is None:
        criticalbottom = bottomlimit * totalmax

    sourcebytes = dirtysource.copy()
    dirtysum = np.sum(sourcebytes)

    cleanImage = np.zeros((imgx,imgy))
    cleanPoints = np.array(cleanImage, dtype=np.float64)

    it = 0
    while it < maxit:
        maxpoint = np.unravel_index(sourcebytes.argmax(), sourcebytes.shape)
        y, x = maxpoint[:2]

        maxvalue = sourcebytes[y][x]
        if maxvalue <= criticalbottom:
            break
       # print(maxvalue)

        x0, x1 = (psfmx - x, psfmx - x + imgx)
        y0, y1 = (psfmy - y, psfmy - y + imgy)
        
        psf_slice = psf[y0:y1, x0:x1]

        psfsum = psf_slice.sum()
        psfmax = psf_slice.max()

        k = gamma * maxvalue / psfmax

        sourcebytes -= psf_slice * k
        cleanPoints[y][x] += psfsum * k

       # if (it % 10) == 0:
       #     cleansum = np.sum(cleanPoints)
        #    dirtysum2 = np.sum(sourcebytes)
       #     print("full = {0}, sourcepic = {1}, it = {2}"
       #           .format(cleansum + dirtysum2, dirtysum, it))

        it += 1
    #print("iterations: {0}".format(it))
    
    cleanPoints = gaussian_filter(cleanPoints, sigma=cleanblurradius)

    return cleanPoints, sourcebytes

def cleanFits(image_file, psf_file):
    dirtysource = getfits(image_file)
    criticalbottom = np.percentile(dirtysource,95)
    print(criticalbottom)
    psf = getfits(psf_file)
    psf = fix_psf(psf)
    bottomlimit = 0.00
    clean, dirtyoutput = makeCleanBigPSF(dirtysource, psf, 12.0, bottomlimit, criticalbottom=criticalbottom,\
                                     maxit=500, gamma=0.1)
    writefits("clean.fits", clean + dirtyoutput)
    print(np.max(dirtyoutput))