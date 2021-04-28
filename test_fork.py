

import os

import sys

​

import h5py

import numpy as np

import torch

​

os.environ['TOOLBOX_PATH'] = "/scratch/dkarkalousos/apps/bart-0.6.00/"

sys.path.append('/scratch/dkarkalousos/apps/bart-0.6.00/python/')

import bart

​

import fastmri.data.transforms as T

from fastmri.data.subsample import create_mask_for_mask_type

from fastmri import tensor_to_complex_np

from fastmri.fftc import ifft2c_new as ifft2c

from fastmri.coil_combine import rss_complex

​

​

import matplotlib.pyplot as plt

​

fname = '/data/projects/recon/data/public/fastmri/knee/multicoil/multicoil_train/file1000002.h5'

data = h5py.File(fname, 'r')

kspace = data["kspace"][()]

​

slice = 20

crop_size = (320, 320)

device = 'cuda'

​

target = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace[slice], axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

target = target / np.max(np.abs(target))

target = np.sqrt(np.sum(T.center_crop(target, crop_size) ** 2, 0))

​

crop_size = (320, 320)

mask_func = create_mask_for_mask_type(mask_type_str="random", center_fractions=[0.08], accelerations=[4])

​

_kspace = T.to_tensor(kspace)[slice]

masked_kspace, mask = T.apply_mask(_kspace, mask_func)

​

linear_recon = masked_kspace[..., 0] + 1j * masked_kspace[..., 1]

linear_recon = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(linear_recon, axes=(-2, -1)), axes=(-2, -1)),

                               axes=(-2, -1))

linear_recon = linear_recon / np.max(np.abs(linear_recon))

linear_recon = np.sqrt(np.sum(T.center_crop(linear_recon, (320, 320)) ** 2, 0))

​

masked_kspace = masked_kspace.permute(1, 2, 0, 3).unsqueeze(0)

masked_kspace = tensor_to_complex_np(masked_kspace)

​

sens_maps = bart.bart(1, "ecalib -d0 -m1", masked_kspace)

​

reg_wt = 0.01

num_iters = 200

pred = np.abs(bart.bart(1, f"pics -d0 -S -R T:7:0:{reg_wt} -i {num_iters}", masked_kspace, sens_maps)[0])

pred = torch.from_numpy(pred / np.max(np.abs(pred))).cpu().numpy()

​

# check for FLAIR 203

if pred.shape[1] < crop_size[1]:

    crop_size = (pred.shape[1], pred.shape[1])

​

pred = T.center_crop(pred, crop_size)

​

plt.subplot(1, 4, 1)

plt.imshow(np.abs(target), cmap='gray')

plt.title('Fully-sampled')

plt.colorbar()

plt.subplot(1, 4, 2)

plt.imshow(np.abs(linear_recon), cmap='gray')

plt.title('4x')

plt.colorbar()

plt.subplot(1, 4, 3)

plt.imshow(np.abs(pred), cmap='gray')

plt.title('PICS')

plt.colorbar()

plt.subplot(1, 4, 4)

plt.imshow(np.abs(target)-np.abs(pred), cmap='gray')

plt.title('PICS')

plt.colorbar()

plt.show()

