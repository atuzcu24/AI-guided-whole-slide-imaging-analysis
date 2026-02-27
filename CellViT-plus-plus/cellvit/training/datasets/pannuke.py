# -*- coding: utf-8 -*-
# PanNuke Dataset
#
# Dataset information: https://arxiv.org/abs/2003.10778
# Please Prepare Dataset as described here: docs/readmes/pannuke.md
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import logging
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from numba import njit
from PIL import Image
from scipy.ndimage import center_of_mass, distance_transform_edt

try:
    from scipy.ndimage import sobel as _scipy_sobel
except ImportError:
    _scipy_sobel = None
try:
    from scipy.ndimage import convolve as _scipy_convolve
except ImportError:
    _scipy_convolve = None

from cellvit.training.datasets.base_cell_dataset import CellDataset
from cellvit.training.utils.tools import fix_duplicates, get_bounding_box

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())

from natsort import natsorted
from albumentations.pytorch import ToTensorV2


class PanNukeDataset(CellDataset):
    """PanNuke dataset

    Args:
        dataset_path (Union[Path, str]): Path to PanNuke dataset. Structure is described under ./docs/readmes/cell_segmentation.md
        folds (Union[int, list[int]]): Folds to use for this dataset
        transforms (Callable, optional): PyTorch transformations. Defaults to None.
        stardist (bool, optional): Return StarDist labels. Defaults to False
        regression (bool, optional): Return Regression of cells in x and y direction. Defaults to False
        cache_dataset: If the dataset should be loaded to host memory in first epoch.
            Be careful, workers in DataLoader needs to be persistent to have speedup.
            Recommended to false, just use if you have enough RAM and your I/O operations might be limited.
            Defaults to False.
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        folds: Union[int, list[int]],
        transforms: Callable = None,
        stardist: bool = False,
        regression: bool = False,
        cache_dataset: bool = False,
        proxy_channels: dict | None = None,
    ) -> None:
        if isinstance(folds, int):
            folds = [folds]

        self.dataset = Path(dataset_path).resolve()
        self.transforms = transforms
        self.images = []
        self.masks = []
        self.types = {}
        self.img_names = []
        self.folds = folds
        self.cache_dataset = cache_dataset
        self.stardist = stardist
        self.regression = regression
        self.proxy_channels = proxy_channels or {}
        self.proxy_type = self.proxy_channels.get("type", "sobel_mag")
        self.proxy_enabled = (
            self.proxy_channels.get("enabled", False)
            and self.proxy_type in ("sobel_mag", "stain_deconv_h")
        )
        self.proxy_normalize = self.proxy_channels.get("normalize", True)
        for fold in folds:
            image_path = self.dataset / f"fold{fold}" / "images"
            fold_images = [
                f for f in natsorted(image_path.glob("*.png")) if f.is_file()
            ]

            # sanity_check: mask must exist for image
            for fold_image in fold_images:
                mask_path = (
                    self.dataset / f"fold{fold}" / "labels" / f"{fold_image.stem}.npy"
                )
                if mask_path.is_file():
                    self.images.append(fold_image)
                    self.masks.append(mask_path)
                    self.img_names.append(fold_image.name)

                else:
                    logger.debug(
                        "Found image {fold_image}, but no corresponding annotation file!"
                    )

            fold_types = pd.read_csv(self.dataset / f"fold{fold}" / "types.csv")
            fold_type_dict = fold_types.set_index("img")["type"].to_dict()
            
            self.types = {
                **self.types,
                **fold_type_dict,
            }  # careful - should all be named differently

        logger.info(f"Created Pannuke Dataset by using fold(s) {self.folds}")
        logger.info(f"Resulting dataset length: {self.__len__()}")

        if self.cache_dataset:
            self.cached_idx = []  # list of idx that should be cached
            self.cached_imgs = {}  # keys: idx, values: numpy array of imgs
            self.cached_masks = {}  # keys: idx, values: numpy array of masks
            logger.info("Using cached dataset. Cache is built up during first epoch.")
        self.totensor_transform = ToTensorV2()

    @staticmethod
    def _sobel_numpy(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sobel gradients (fallback when scipy.ndimage.sobel unavailable)."""
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        if _scipy_convolve is not None:
            gx = _scipy_convolve(gray, kx, mode="nearest")
            gy = _scipy_convolve(gray, ky, mode="nearest")
        else:
            gx = np.zeros_like(gray, dtype=np.float64)
            gy = np.zeros_like(gray, dtype=np.float64)
            h, w = gray.shape
            padded = np.pad(gray.astype(np.float64), 1, mode="edge")
            for i in range(h):
                for j in range(w):
                    patch = padded[i : i + 3, j : j + 3]
                    gx[i, j] = np.sum(patch * kx)
                    gy[i, j] = np.sum(patch * ky)
        return gx.astype(np.float32), gy.astype(np.float32)

    @staticmethod
    def _stain_deconv_hematoxylin(img: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Extract Hematoxylin channel via Ruifrok & Johnston color deconvolution.
        img: (H,W,3) RGB uint8 or float. Returns (H,W) float H channel."""
        if img.dtype == np.uint8:
            rgb = img.astype(np.float64) / 255.0
        else:
            rgb = np.asarray(img, dtype=np.float64)
        eps = 1e-8
        od = -np.log10(np.clip(rgb + eps, eps, 1.0))
        # Ruifrok H&E stain matrix (columns: H, E; rows: R,G,B in OD space)
        W_he = np.array([
            [0.650, 0.072],
            [0.704, 0.990],
            [0.286, 0.105],
        ], dtype=np.float64)
        # Complement third column for full rank (cross product)
        v1, v2 = W_he[:, 0], W_he[:, 1]
        v3 = np.cross(v1, v2)
        v3 = v3 / (np.linalg.norm(v3) + eps)
        W = np.column_stack([v1, v2, v3])
        Q = np.linalg.pinv(W)
        h, w, _ = od.shape
        od_flat = od.reshape(-1, 3).T
        conc_flat = (Q @ od_flat).T
        hema = conc_flat[:, 0].reshape(h, w).astype(np.float32)
        if normalize:
            m_min, m_max = hema.min(), hema.max()
            if m_max > m_min:
                hema = (hema - m_min) / (m_max - m_min)
            else:
                hema = np.zeros_like(hema)
        hema = np.clip(hema, 0.0, 1.0)
        return hema

    def _add_proxy_channel(self, img: np.ndarray) -> np.ndarray:
        """Add proxy channel (Sobel mag or Hematoxylin). img: (H,W,3) -> (H,W,4)."""
        if self.proxy_type == "stain_deconv_h":
            proxy = self._stain_deconv_hematoxylin(img, self.proxy_normalize)
        else:
            proxy = self._add_sobel_proxy_channel(img)
        proxy = proxy[:, :, np.newaxis]
        if img.dtype == np.uint8:
            base = img
            proxy_uint = (np.clip(proxy, 0, 1) * 255).astype(np.uint8)
            return np.concatenate([base, proxy_uint], axis=-1)
        base = np.asarray(img, dtype=np.float32)
        return np.concatenate([base, proxy.astype(np.float32)], axis=-1)

    def _add_sobel_proxy_channel(self, img: np.ndarray) -> np.ndarray:
        """Add Sobel magnitude proxy. img: (H,W,3) -> (H,W) float."""
        if img.dtype == np.uint8:
            img_f = img.astype(np.float32) / 255.0
        else:
            img_f = np.asarray(img, dtype=np.float32)
        gray = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
        if _scipy_sobel is not None:
            gx = _scipy_sobel(gray, axis=1)
            gy = _scipy_sobel(gray, axis=0)
        else:
            gx, gy = self._sobel_numpy(gray)
        mag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
        if self.proxy_normalize:
            m_min, m_max = mag.min(), mag.max()
            if m_max > m_min:
                mag = (mag - m_min) / (m_max - m_min)
            else:
                mag = np.zeros_like(mag)
        return np.clip(mag, 0.0, 1.0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, str, str]:
        """Get one dataset item consisting of transformed image,
        masks (instance_map, nuclei_type_map, nuclei_binary_map, hv_map) and tissue type as string

        Args:
            index (int): Index of element to retrieve

        Returns:
            Tuple[torch.Tensor, dict, str, str]:
                torch.Tensor: Image, with shape (3, H, W), in this case (3, 256, 256)
                dict:
                    "instance_map": Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (256, 256)
                    "nuclei_type_map": Nuclei-Type-Map, for each nucleus (instance) the class is indicated by an integer. Shape (256, 256)
                    "nuclei_binary_map": Binary Nuclei-Mask, Shape (256, 256)
                    "hv_map": Horizontal and vertical instance map.
                        Shape: (2 , H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                        last is vertical (vertical gradient (-1 to 1)) Shape (2, 256, 256)
                    [Optional if stardist]
                    "dist_map": Probability distance map. Shape (256, 256)
                    "stardist_map": Stardist vector map. Shape (n_rays, 256, 256)
                    [Optional if regression]
                    "regression_map": Regression map. Shape (2, 256, 256). First is vertical, second horizontal.
                str: Tissue type
                str: Image Name
        """
        img_path = self.images[index]

        if self.cache_dataset:
            if index in self.cached_idx:
                img = self.cached_imgs[index]
                mask = self.cached_masks[index]
            else:
                # cache file
                img = self.load_imgfile(index)
                mask = self.load_maskfile(index)
                self.cached_imgs[index] = img
                self.cached_masks[index] = mask
                self.cached_idx.append(index)

        else:
            img = self.load_imgfile(index)
            mask = self.load_maskfile(index)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        tissue_type = self.types[img_path.name]
        inst_map = mask[:, :, 0].copy()
        type_map = mask[:, :, 1].copy()
        np_map = mask[:, :, 0].copy()
        np_map[np_map > 0] = 1
        hv_map = PanNukeDataset.gen_instance_hv_map(inst_map)

        # Proxy channel: Sobel magnitude or Hematoxylin (stain_deconv_h)
        if self.proxy_enabled:
            img = self._add_proxy_channel(img)

        # torch convert
        img = self.totensor_transform(image=img)["image"]

        if self.proxy_enabled:
            assert img.shape[0] == 4, (
                f"[PanNuke proxy] Expected 4 channels (RGB+proxy), got {img.shape[0]}. "
                "Check proxy_channels.enabled and transforms."
            )

        masks = {
            "instance_map": torch.Tensor(inst_map).type(torch.int64),
            "nuclei_type_map": torch.Tensor(type_map).type(torch.int64),
            "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
            "hv_map": torch.Tensor(hv_map).type(torch.float32),
        }

        # load stardist transforms if neccessary
        if self.stardist:
            dist_map = PanNukeDataset.gen_distance_prob_maps(inst_map)
            stardist_map = PanNukeDataset.gen_stardist_maps(inst_map)
            masks["dist_map"] = torch.Tensor(dist_map).type(torch.float32)
            masks["stardist_map"] = torch.Tensor(stardist_map).type(torch.float32)
        if self.regression:
            masks["regression_map"] = PanNukeDataset.gen_regression_map(inst_map)

        return img, masks, tissue_type, Path(img_path).name

    def __len__(self) -> int:
        """Length of Dataset

        Returns:
            int: Length of Dataset
        """
        return len(self.images)

    def set_transforms(self, transforms: Callable) -> None:
        """Set the transformations, can be used tp exchange transformations

        Args:
            transforms (Callable): PyTorch transformations
        """
        self.transforms = transforms

    def load_imgfile(self, index: int) -> np.ndarray:
        """Load image from file (disk)

        Args:
            index (int): Index of file

        Returns:
            np.ndarray: Image as array with shape (H, W, 3)
        """
        img_path = self.images[index]
        return np.array(Image.open(img_path)).astype(np.uint8)

    def load_maskfile(self, index: int) -> np.ndarray:
        """Load mask from file (disk)

        Args:
            index (int): Index of file

        Returns:
            np.ndarray: Mask as array with shape (H, W, 2)
        """
        mask_path = self.masks[index]
        mask = np.load(mask_path, allow_pickle=True)
        inst_map = mask[()]["inst_map"].astype(np.int32)
        type_map = mask[()]["type_map"].astype(np.int32)
        mask = np.stack([inst_map, type_map], axis=-1)
        return mask

    def load_cell_count(self):
        """Load Cell count from cell_count.csv file. File must be located inside the fold folder
        and named "cell_count.csv"

        Example file beginning:
            Image,Neoplastic,Inflammatory,Connective,Dead,Epithelial
            0_0.png,4,2,2,0,0
            0_1.png,8,1,1,0,0
            0_10.png,17,0,1,0,0
            0_100.png,10,0,11,0,0
            ...
        """
        df_placeholder = []
        for fold in self.folds:
            csv_path = self.dataset / f"fold{fold}" / "cell_count.csv"
            cell_count = pd.read_csv(csv_path, index_col=0)
            df_placeholder.append(cell_count)
        self.cell_count = pd.concat(df_placeholder)
        self.cell_count = self.cell_count.reindex(self.img_names)

    def get_sampling_weights_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by tissue type statistics

        Uses weight_config.yaml in the dataset root if present; otherwise derives
        tissue counts from self.types (from types.csv in folds).

        weight_config.yaml format:
            tissue:
                tissue_1: xxx
                tissue_2: xxx (name of tissue: count)
                ...

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        weight_config_path = (self.dataset / "weight_config.yaml").resolve()
        if weight_config_path.is_file():
            with open(weight_config_path, "r") as run_config_file:
                yaml_config = yaml.safe_load(run_config_file)
                tissue_counts = dict(yaml_config)["tissue"]
        else:
            from collections import Counter
            tissue_counts = dict(Counter(self.types.values()))

        # calculate weight for each tissue
        weights_dict = {}
        k = np.sum(list(tissue_counts.values()))
        for tissue, count in tissue_counts.items():
            w = k / (gamma * count + (1 - gamma) * k)
            weights_dict[tissue] = w

        weights = []
        for idx in range(self.__len__()):
            img_idx = self.img_names[idx]
            type_str = self.types[img_idx]
            weights.append(weights_dict[type_str])

        return torch.Tensor(weights)

    def get_sampling_weights_cell(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by cell type statistics

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        assert hasattr(self, "cell_count"), "Please run .load_cell_count() in advance!"
        binary_weight_factors = np.array([4191, 4132, 6140, 232, 1528])
        k = np.sum(binary_weight_factors)
        cell_counts_imgs = np.clip(self.cell_count.to_numpy(), 0, 1)
        weight_vector = k / (gamma * binary_weight_factors + (1 - gamma) * k)
        img_weight = (1 - gamma) * np.max(cell_counts_imgs, axis=-1) + gamma * np.sum(
            cell_counts_imgs * weight_vector, axis=-1
        )
        img_weight[np.where(img_weight == 0)] = np.min(
            img_weight[np.nonzero(img_weight)]
        )

        return torch.Tensor(img_weight)

    def get_sampling_weights_cell_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Get combined sampling weights by calculating tissue and cell sampling weights,
        normalizing them and adding them up to yield one score.

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        tw = self.get_sampling_weights_tissue(gamma)
        cw = self.get_sampling_weights_cell(gamma)
        weights = tw / torch.max(tw) + cw / torch.max(cw)

        return weights

    @staticmethod
    def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
        """Obtain the horizontal and vertical distance maps for each
        nuclear instance.

        Args:
            inst_map (np.ndarray): Instance map with each instance labelled as a unique integer
                Shape: (H, W)
        Returns:
            np.ndarray: Horizontal and vertical instance map.
                Shape: (2, H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                last is vertical (vertical gradient (-1 to 1))
        """
        orig_inst_map = inst_map.copy()  # instance ID map

        x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(orig_inst_map))

        if 0 in inst_list:
            inst_list.remove(0)  # remove background safely

        for inst_id in inst_list:
            inst_map = np.array(orig_inst_map == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            if inst_box[0] >= 2:
                inst_box[0] -= 2
            if inst_box[2] >= 2:
                inst_box[2] -= 2
            if inst_box[1] <= orig_inst_map.shape[0] - 2:
                inst_box[1] += 2
            if inst_box[3] <= orig_inst_map.shape[0] - 2:
                inst_box[3] += 2

            # improvement
            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.stack([x_map, y_map])
        return hv_map

    @staticmethod
    def gen_distance_prob_maps(inst_map: np.ndarray) -> np.ndarray:
        """Generate distance probability maps

        Args:
            inst_map (np.ndarray): Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (H, W)

        Returns:
            np.ndarray: Distance probability map, shape (H, W)
        """
        inst_map = fix_duplicates(inst_map)
        dist = np.zeros_like(inst_map, dtype=np.float64)
        inst_list = list(np.unique(inst_map))
        if 0 in inst_list:
            inst_list.remove(0)

        for inst_id in inst_list:
            inst = np.array(inst_map == inst_id, np.uint8)

            y1, y2, x1, x2 = get_bounding_box(inst)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

            inst = inst[y1:y2, x1:x2]

            if inst.shape[0] < 2 or inst.shape[1] < 2:
                continue

            # chessboard distance map generation
            # normalize distance to 0-1
            inst_dist = distance_transform_edt(inst)
            inst_dist = inst_dist.astype("float64")

            max_value = np.amax(inst_dist)
            if max_value <= 0:
                continue
            inst_dist = inst_dist / (np.max(inst_dist) + 1e-10)

            dist_map_box = dist[y1:y2, x1:x2]
            dist_map_box[inst > 0] = inst_dist[inst > 0]

        return dist

    @staticmethod
    @njit
    def gen_stardist_maps(inst_map: np.ndarray) -> np.ndarray:
        """Generate StarDist map with 32 nrays

        Args:
            inst_map (np.ndarray): Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (H, W)

        Returns:
            np.ndarray: Stardist vector map, shape (n_rays, H, W)
        """
        n_rays = 32
        # inst_map = fix_duplicates(inst_map)
        dist = np.empty(inst_map.shape + (n_rays,), np.float32)

        st_rays = np.float32((2 * np.pi) / n_rays)
        for i in range(inst_map.shape[0]):
            for j in range(inst_map.shape[1]):
                value = inst_map[i, j]
                if value == 0:
                    dist[i, j] = 0
                else:
                    for k in range(n_rays):
                        phi = np.float32(k * st_rays)
                        dy = np.cos(phi)
                        dx = np.sin(phi)
                        x, y = np.float32(0), np.float32(0)
                        while True:
                            x += dx
                            y += dy
                            ii = int(round(i + x))
                            jj = int(round(j + y))
                            if (
                                ii < 0
                                or ii >= inst_map.shape[0]
                                or jj < 0
                                or jj >= inst_map.shape[1]
                                or value != inst_map[ii, jj]
                            ):
                                # small correction as we overshoot the boundary
                                t_corr = 1 - 0.5 / max(np.abs(dx), np.abs(dy))
                                x -= t_corr * dx
                                y -= t_corr * dy
                                dst = np.sqrt(x**2 + y**2)
                                dist[i, j, k] = dst
                                break

        return dist.transpose(2, 0, 1)

    @staticmethod
    def gen_regression_map(inst_map: np.ndarray):
        n_directions = 2
        dist = np.zeros(inst_map.shape + (n_directions,), np.float32).transpose(2, 0, 1)
        inst_map = fix_duplicates(inst_map)
        inst_list = list(np.unique(inst_map))
        if 0 in inst_list:
            inst_list.remove(0)
        for inst_id in inst_list:
            inst = np.array(inst_map == inst_id, np.uint8)
            y1, y2, x1, x2 = get_bounding_box(inst)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

            inst = inst[y1:y2, x1:x2]
            y_mass, x_mass = center_of_mass(inst)
            x_map = np.repeat(np.arange(1, x2 - x1 + 1)[None, :], y2 - y1, axis=0)
            y_map = np.repeat(np.arange(1, y2 - y1 + 1)[:, None], x2 - x1, axis=1)
            # we use a transposed coordinate system to align to HV-map, correct would be -1*x_dist_map and -1*y_dist_map
            x_dist_map = (x_map - x_mass) * np.clip(inst, 0, 1)
            y_dist_map = (y_map - y_mass) * np.clip(inst, 0, 1)
            dist[0, y1:y2, x1:x2] = x_dist_map
            dist[1, y1:y2, x1:x2] = y_dist_map

        return dist
