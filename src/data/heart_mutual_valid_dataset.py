import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms 
import numpy as np
import nibabel as nib
import glob
import os

from data.base_dataset import BaseDataset


class HeartMutualValidDataset(BaseDataset):
    def __init__(self, opt):
        """
        Creates dataset with augumentation from nifti files in a directory
        Args:
            images_directory_ct: path of directory containing images
            kind: one of ("ct", "mr") - type of images in dataset
            rotation_degrees: Range of degrees to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees).
            size: desired size; if None no resizing is done
            h_flip: if True horizontal flip is randomly applied 
            v_flip: if True vertical flip is randomly applied
        """
        images_directory_ct = "../data/affregcommon2mm_roi_ct_valid"
        size = (opt.load_size, opt.load_size)

        self.LABEL_ENCODING_ARRAY = np.array([  0, 205, 500, 600, 420, 550, 820, 850]).reshape((-1, 1, 1))
        self.SLICES_PER_SCAN = 96
        self.images_paths_ct = glob.glob(images_directory_ct + os.sep + "*image*")
        self.labels_paths_ct = [name.replace("_image.nii.gz", "_label.nii.gz") for name in self.images_paths_ct]
        self.normalize_scan = transforms.Normalize(0.5,0.5)
        self.resize = size
        if size is not None:
            self.label_resizer = transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
            self.image_resizer = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
        self.normalize_ct = transforms.Normalize(-221.20035285101997, 439.02997204491294)

    def __len__(self):
        return len(self.images_paths_ct) * self.SLICES_PER_SCAN

    def __getitem__(self, idx):
        scan_ct = nib.load(self.images_paths_ct[idx // self.SLICES_PER_SCAN])
        label_scan_ct = nib.load(self.labels_paths_ct[idx // self.SLICES_PER_SCAN])
        index_ct = idx % self.SLICES_PER_SCAN
        scan_slice_ct, label_slice_ct = self.transform(scan_ct, label_scan_ct, index_ct)

        return {"T_scan": scan_slice_ct, "T_labels": label_slice_ct.float()}

    def transform(self, scan, label_scan, index):
        #reading data
        scan_slice = scan.slicer[:,:, index:index+1].get_fdata()
        label_slice = label_scan.slicer[:,:, index:index+1].get_fdata()
        label_slice = torch.from_numpy(label_slice).reshape(1, 96, 80).float()
        scan_slice = torch.from_numpy(scan_slice).reshape(1, 96, 80).float()
        scan_slice = self.normalize_ct(scan_slice)
        # random transformations (images and labels are concatenated to ensure the same transformations are applied to them)
        label_slice_encoded = np.equal(label_slice, self.LABEL_ENCODING_ARRAY)      #onehot encoding label
        if self.resize is not None:
            scan_slice = self.image_resizer(scan_slice)
            label_slice_encoded = self.label_resizer(label_slice_encoded)
        scan_slice = (scan_slice-torch.min(scan_slice))/(torch.max(scan_slice) - torch.min(scan_slice))
        scan_slice = self.normalize_scan(scan_slice)
        return scan_slice, label_slice_encoded


