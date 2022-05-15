import wget
from pathlib import Path
import zipfile
import random

def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()

DATA_FOLDER = Path("./data")


CT_FOLDER_TRAIN: Path = DATA_FOLDER / "affregcommon2mm_roi_ct_train"
CT_FOLDER_VALID: Path = DATA_FOLDER / "affregcommon2mm_roi_ct_valid"
MR_FOLDER_TRAIN: Path = DATA_FOLDER / "affregcommon2mm_roi_mr_train"

CT_ZIP = CT_FOLDER_TRAIN.with_suffix(".zip")
MR_ZIP = MR_FOLDER_TRAIN.with_suffix(".zip")

if CT_FOLDER_VALID.exists() or CT_ZIP.exists() or MR_ZIP.exists() or CT_FOLDER_TRAIN.exists() or MR_FOLDER_TRAIN.exists():
    raise Exception("There are remeining folders and files in directory remove them to continue")

wget.download("http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/affregcommon2mm_roi_ct_train.zip", out=str(CT_ZIP))
wget.download("http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/affregcommon2mm_roi_mr_train.zip", out=str(MR_ZIP))

with zipfile.ZipFile(CT_ZIP, 'r') as zip_ref:
    zip_ref.extractall(DATA_FOLDER)

with zipfile.ZipFile(MR_ZIP, 'r') as zip_ref:
    zip_ref.extractall(DATA_FOLDER)

scan_numbers = [i for i in range(1001, 1021)]

random.seed(2134)
valid_numbers = random.sample(scan_numbers, k=len(scan_numbers)//2)
CT_FOLDER_VALID.mkdir(exist_ok=True)

for v_number in valid_numbers:
    current_image_file = CT_FOLDER_TRAIN / f"roi_ct_train_{v_number}_image.nii.gz"
    current_label_file = CT_FOLDER_TRAIN / f"roi_ct_train_{v_number}_label.nii.gz"

    assert current_image_file.exists()
    assert current_label_file.exists()

    new_image_file = CT_FOLDER_VALID / f"roi_ct_train_{v_number}_image.nii.gz"
    new_label_file = CT_FOLDER_VALID / f"roi_ct_train_{v_number}_label.nii.gz"

    current_image_file.replace(new_image_file)
    current_label_file.replace(new_label_file)


