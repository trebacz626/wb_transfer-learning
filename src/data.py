import torch 
from torch.utils.data import Dataset
from torchvision import transforms 
import numpy as np
import nibabel as nib
import glob
import os

class HeartDataset(Dataset):
    def __init__(self, images_directory, kind = "ct" , rotation_degrees = 5, size = None, h_flip = False, v_flip = False):
        """
        Creates dataset with augumentation from nifti files in a directory
        Args:
            images_directory: path of directory containing images
            kind: one of ("ct", "mr") - type of images in dataset
            rotation_degrees: Range of degrees to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees).
            size: desired size; if None no resizing is done
            h_flip: if True horizontal flip is randomly applied 
            v_flip: if True vertical flip is randomly applied
        """
        self.LABEL_ENCODING_ARRAY = np.array([  0, 205, 500, 600, 420, 550, 820, 850]).reshape((-1, 1, 1))
        self.SLICES_PER_SCAN = 96
        self.images_paths = glob.glob(images_directory + os.sep + "*image*")
        self.labels_paths = [name.replace("_image.nii.gz", "_label.nii.gz") for name in self.images_paths]
        random_transforms = []
        random_transforms.append(transforms.RandomRotation(rotation_degrees))
        if h_flip:
            random_transforms.append(transforms.RandomHorizontalFlip())
        if v_flip:
            random_transforms.append(transforms.RandomVerticalFlip())
        self.random_transform = transforms.Compose(random_transforms) 
        self.resize = size
        if size is not None:
            self.label_resizer = transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
            self.image_resizer = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        if kind == "ct":
            self.normalize = transforms.Normalize(-221.20035285101997, 439.02997204491294)
        elif kind == "mr":
            self.normalize = transforms.Normalize(261.89050726996527, 321.0735487258476)
        else:
            raise ValueError("kind should be one of (\"ct\", \"mr\") ")

    def __len__(self):
        return len(self.images_paths * self.SLICES_PER_SCAN)

    def __getitem__(self, idx):
        scan = nib.load(self.images_paths[idx//self.SLICES_PER_SCAN])
        label_scan = nib.load(self.labels_paths[idx//self.SLICES_PER_SCAN])
        index = idx % self.SLICES_PER_SCAN
        #reading data
        scan_slice = scan.slicer[:,:, index:index+1].get_fdata()
        label_slice = label_scan.slicer[:,:, index:index+1].get_fdata()
        label_slice = torch.from_numpy(label_slice).reshape(1, 96, 80)
        scan_slice = torch.from_numpy(scan_slice).reshape(1, 96, 80)
        scan_slice = self.normalize(scan_slice)
        # random transformations (images and labels are concatenated to ensure the same transformations are applied to them)
        cated_tensor = torch.cat((scan_slice, label_slice), 0)
        cated_tensor = self.random_transform(cated_tensor)
        scan_slice, label_slice= cated_tensor[[0],:,:], cated_tensor[[1],:,:]
        label_slice_encoded = np.equal(label_slice, self.LABEL_ENCODING_ARRAY)      #onehot encoding label
        if self.resize is not None:
            scan_slice = self.image_resizer(scan_slice)
            label_slice_encoded = self.label_resizer(label_slice_encoded)
        return scan_slice, label_slice_encoded

