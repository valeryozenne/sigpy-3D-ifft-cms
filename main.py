#matplotlib notebook
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import cupy as cp
import time
import matplotlib.pyplot as plt
import numpy.ma as ma
import pathlib
import os
import sys
from sys import getsizeof

use_gpu=False

# load data after coil compression
#
ksp = np.load('/home/valery/DICOM/mp2rage_data.npy')

# remove 
ksp=np.squeeze(ksp)

# tranpose from [RO E1 E2 CHA] to [CHA E2 E1 RO]
# [CHA E2 E1 RO] = (4, 240, 320, 320)
ksp=np.transpose(ksp, (3, 2 , 1, 0))

# remove coils to satisfy sigpy and my laptop
ksp=ksp[0:4,:,:,:]*100

print(sys.getsizeof(ksp))

print(np.shape(ksp))

# check kspace sampling
mask=ma.masked_greater(np.abs(ksp), 0)

#display sampling
#plt.figure(1)
#lala=ksp[0,:,:,160]
#print(np.shape(lala))
#plt.imshow(np.abs(mask[0,:,:,160]))
#plt.show()

# get image dimension
img_shape = ksp.shape[1:]


# host to device
if (use_gpu==True):
   ksp_on_gpu0 = sp.to_device(ksp, 0)

# compute ifft on host
tstart = time.time()
F = sp.linop.FFT(ksp.shape, axes=(-1, -2, -3))
I=F.H * ksp
print("Ifft3D estimation duration numpy: {}s".format(time.time() - tstart))


if (use_gpu==True):
  try:
    # compute ifft on device
    tstart = time.time()
    F_gpu = sp.linop.FFT(ksp_on_gpu0.shape, axes=(-1, -2, -3))
    I_gpu=F_gpu.H * ksp_on_gpu0
    print("Ifft3D estimation duration cupy: {}s".format(time.time() - tstart))
  
    # display result
    #pl.ImagePlot(I_gpu, z=0, title=r'$F^H y$')
  except:
    print("Ifft3D could not be computed on device")

   
# compute csm on host
tstart = time.time()
#mps = mr.app.EspiritCalib(ksp).run()
mps=sp.mri.app.JsenseRecon(ksp).run()
print("EspiritCalib estimation duration numpy: {}s".format(time.time() - tstart))



# display result
#pl.ImagePlot(mps, z=0, title=r'$F^H y$')

if (use_gpu==True):
  try:
    # compute ifft on device
    tstart = time.time()
    mps_on_gpu = mr.app.EspiritCalib(ksp_on_gpu0).run()
    print("EspiritCalib estimation duration cupy: {}s".format(time.time() - tstart))
  
    # display result
    #pl.ImagePlot(mps_on_gpu, z=0, title=r'$F^H y$')
  except:
    print("EspiritCalib could not be computed on device")




###########################################################################



print("Define S")

S = sp.linop.Multiply(img_shape, mps)

#pl.ImagePlot(S.H * F.H * ksp, title=r'$S^H F^H y$')



mask = np.sum(abs(ksp), axis=0) > 0
#pl.ImagePlot(mask, title='Sampling Mask')

P = sp.linop.Multiply(ksp.shape, mask)


## W Linop
print("Define W Linop")

W = sp.linop.Wavelet(img_shape)
wav = W * S.H * F.H * ksp
#pl.ImagePlot(wav**0.1, title=r'$W S^H F^H y$')

print(np.amax(np.abs(wav)))
print(np.amin(np.abs(wav)))
print(np.shape(wav))

plt.figure(1)
lala=ksp[0,:,:,160]
print(np.shape(lala))
plt.imshow(np.abs(wav[:,:,160]))
plt.clim(0.0001,0.001)
plt.show()



pl.ImagePlot(wav, title=r'$W S^H F^H y$')

A = P * F * S * W.H

## Prox
print("Define Prox")

lamda = 0.005
proxg = sp.prox.L1Reg(wav.shape, lamda)
alpha = 1
wav_thresh = proxg(alpha, wav)

pl.ImagePlot(wav_thresh**0.1)


## Alg
print("Define Alg")

max_iter = 30
alpha = 1

def gradf(x):
    return A.H * (A * x - ksp)

wav_hat = np.zeros(wav.shape, np.complex)
alg = sp.alg.GradientMethod(gradf, wav_hat, alpha, proxg=proxg, max_iter=max_iter)

while not alg.done():
    alg.update()
    print('\rL1WaveletRecon, Iteration={}'.format(alg.iter), end='')

pl.ImagePlot(W.H(wav_hat))


## App

class L1WaveletRecon(sp.app.App):
    def __init__(self, ksp, mask, mps, lamda, max_iter):
        img_shape = mps.shape[1:]
        
        S = sp.linop.Multiply(img_shape, mps)
        F = sp.linop.FFT(ksp.shape, axes=(-1, -2))
        P = sp.linop.Multiply(ksp.shape, mask)
        self.W = sp.linop.Wavelet(img_shape)
        A = P * F * S * self.W.H
        
        proxg = sp.prox.L1Reg(A.ishape, lamda)
        
        self.wav = np.zeros(A.ishape, np.complex)
        alpha = 1
        def gradf(x):
            return A.H * (A * x - ksp)

        alg = sp.alg.GradientMethod(gradf, self.wav, alpha, proxg=proxg, 
                                    max_iter=max_iter)
        super().__init__(alg)
        
    def _output(self):
        return self.W.H(self.wav)


img = L1WaveletRecon(ksp, mask, mps, lamda, max_iter).run()
pl.ImagePlot(img)

