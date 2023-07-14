import torch
from PIL import Image

from Utils.OSHelper import OSHelper
from torch.utils.data import Dataset
import numpy as np
from Utils.ImageHelper import ImageHelper
from ImageTransformer.ImageTransformer import ImageTransformer, IdentityTransformer
from MultiProcessingHelper import MultiProcessingHelper
from Utils.MetaImageHelper2 import MetaImageHelper
import json
from augmentations.simsiam_aug import NewSimSiamTransform

class TrainingDataset(Dataset):

    def __init__(self,
                 split_fold: int | str,
                 # image_size: tuple[int, int],
                 # load_size: tuple[int, int],
                 aug_conf: str,
                 mode: str,
                 n_worker,
                 preload=False,
                 verbose=True):
        self.split_fold = split_fold
        self.image_size = tuple((256, 128))
        self.load_size = tuple((300, 150))
        self.n_worker = n_worker
        self.preload = preload
        self.verbose = verbose
        self.mode = mode

        self.transformer = IdentityTransformer()
        if aug_conf is not None:
            if aug_conf.lower() != "none":
                self.transformer = ImageTransformer(**self.transformer_param_dict[aug_conf])

        self.data_root = OSHelper.format_path(r"/win/salmon\user\koku\data\BMDEst2")
        with open(OSHelper.path_join(self.data_root, r"xp_2_bone_noval.json"), 'r') as f:
            training_case_names = json.load(f)[str(split_fold)][f"{mode}"]

        self.xp_root = OSHelper.path_join(self.data_root, "Xp_LR_561")
        # self.drr_root = OSHelper.path_join(self.data_root, "Bone_DRR_LR_561")

        self.xp_pool = []
        # self.drr_pool = []
        for case_name in training_case_names:
            case_xp_dir = OSHelper.path_join(self.xp_root, case_name)
            if not OSHelper.path_exists(case_xp_dir):
                continue
            for slice_entry in OSHelper.scan_dirs_for_file(case_xp_dir, name_re_pattern=".+\\.mhd$"):
                self.xp_pool.append(slice_entry.path)
        assert len(self.xp_pool) > 0

        if self.verbose:
            print("Trainig Datasets")
            print(f"Xp: {len(self.xp_pool)}")

        if self.preload:
            args = []
            for xp_path in self.xp_pool:
                args.append((xp_path, self.load_size))
            self.xp_pool = MultiProcessingHelper().run(args=args, func=self._load_image, n_workers=self.n_worker,
                                                       desc="Loading Xp" if self.verbose else None,
                                                       mininterval=60, maxinterval=180)

    def __len__(self):
        return len(self.xp_pool)

    def __getitem__(self, idx):
        xp_path = self.xp_pool[idx]

        if self.preload:
            xp = xp_path.copy()
        else:
            xp = self._load_image(xp_path, self.load_size)

        xp = xp.astype(np.float32)
        # if self.mode == 'train':
            # transform_parameters = self.transformer.get_random_transform(img_shape=self.load_size)
            # xp = self.transformer.apply_transform(x=xp, transform_parameters=transform_parameters)
        data_transform = NewSimSiamTransform(self.image_size)
        img_list = data_transform(xp)

        for img in img_list:
            img = img / 255.
            img = ImageHelper.standardize(img, 0.5, 0.5)
            img = torch.clip(img, -1., 1.)
            # img = img.astype(torch.float32)
            img = torch.permute(img, (2, 0, 1))

        return img_list

    @staticmethod
    def _load_image(load_path, load_size):
        img, _ = MetaImageHelper.read(load_path)
        if img.ndim < 3:
            img = img[..., np.newaxis]
        else:
            img = np.transpose(img, (1, 2, 0))  # (H, W, 1)
        img = img.astype(np.float64)
        img = ImageHelper.denormal(img)
        img = ImageHelper.resize(img, output_shape=load_size)
        return img.astype(np.float32)

    transformer_param_dict = {"paired_synthesis": dict(
        brightness_range=(0.5, 1.5),
        contrast_range=(0.5, 1.5),
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=25,
        shear_range=8,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        lock_zoom_ratio=False
    )}
