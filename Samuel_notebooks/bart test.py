import os

import sys

import h5py

import numpy as np

import torch

# import BART
os.environ['TOOLBOX_PATH'] = "/home/svangurp/scratch/samuel/bart-0.6.00/"

sys.path.append('/home/svangurp/scratch/samuel/bart-0.6.00/python/')

import bart

import fastmri.data.transforms as T

from fastmri.data.subsample import create_mask_for_mask_type

from fastmri import tensor_to_complex_np

import matplotlib.pyplot as plt
# open file
fname = '/scratch/svangurp/samuel/data/knee/train/file1000001.h5'
# read h5 file
data = h5py.File(fname, 'r')
# extract the multi coil k-space data
kspace = data["kspace"][()]
# selecting a single slice to work on
slice = 5

target = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace[slice], axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

# RSS combination
target = np.sqrt(np.sum(T.center_crop(target, (320, 320)) ** 2, 0))

crop_size = (320, 320)

# applying a random mask
mask_func = create_mask_for_mask_type(mask_type_str="random", center_fractions=[0.08], accelerations=[4])

# transforming one slice of the multi coil k-space data
_kspace = T.to_tensor(kspace)[slice]
# applying the mask to the one slice of the multi coil k-space data
masked_kspace, mask = T.apply_mask(_kspace, mask_func)

# splitting the values
linear_recon = masked_kspace[..., 0] + 1j * masked_kspace[..., 1]
# going to image space
linear_recon = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(linear_recon, axes=(-2, -1)), axes=(-2, -1)),

                               axes=(-2, -1))
# RSS coil combination
linear_recon = np.sqrt(np.sum(T.center_crop(linear_recon, (320, 320)) ** 2, 0))

# like fftshift
masked_kspace = masked_kspace.permute(1, 2, 0, 3).unsqueeze(0)


masked_kspace = tensor_to_complex_np(masked_kspace)
# estemating the sens maps
sens_maps = bart.bart(1, "ecalib -d0 -m1", masked_kspace)

reg_wt = 0.01
num_iters = 200

pred = bart.bart(1, f"pics -d0 -S -R T:7:0:{reg_wt} -i {num_iters}", masked_kspace, sens_maps)

pred = torch.from_numpy(np.abs(pred[0]))

# check for FLAIR 203

if pred.shape[1] < crop_size[1]:

    crop_size = (pred.shape[1], pred.shape[1])

pred = T.center_crop(pred, crop_size)


# plotting
plt.subplot(1, 3, 1)

plt.imshow(np.abs(target), cmap='gray')

plt.title('Fully-sampled')

plt.subplot(1, 3, 2)

plt.imshow(np.abs(linear_recon), cmap='gray')

plt.title('4x')

plt.subplot(1, 3, 3)

plt.imshow(np.abs(pred), cmap='gray')

plt.title('PICS')

plt.show()
print('hallo')