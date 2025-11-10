#!/usr/bin/env python
"""Classes to load and manipulate images and masks.
"""
from typing import List, Union

import anndata
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")


class DataLoader:
    """
    DataLoader is an abstract class that provides methods to read, write
    and manipulate image and mask data. It is not meant for direct use,
    but is extended by Image and Mask classes.

    Attributes
    ----------
    `mask` : np.array
        A 2D or 3D matrix of integers, where each integer represents a
        different region.

    Constructor
    ----------
    file : str or list of str
        Path to file(s) to load.
    **kwargs : dict
        Keyword arguments to pass to the loader.

    Methods
    ----------
    `load` : Load data from file(s).
    `store_mask` : Store mask in the `mask` attribute, appending if necessary.
    """

    def __init__(self):
        self.mask = None

    def load(self, file: Union[str, List[str]], **kwargs) -> np.array:
        """
        Load data from file(s). Handles .npz, .h5ad and image files.

        Parameters
        ----------
        file : str or list of str
            Path to file(s) to load.
        **kwargs : dict
            Keyword arguments to pass to the loader.

        Returns
        ----------
        np.array
            The loaded data.
        """

        def data_loader(file):
            if file.endswith(".npz"):
                return self._load_dense_or_sparse
            return self._load_adata if file.endswith(".h5ad") else self._load_images

        if isinstance(file, list):
            if len(file) == 0:
                raise ValueError("List of files is empty.")
            file = self._flatten(file)
            # assert that all files have the same file ending
            assert (
                len({f.split(".")[-1] for f in file}) == 1
            ), "All files must have the same file ending."
            return data_loader(file[0])(file, **kwargs)
        return data_loader(file)(file, **kwargs)

    def _load_dense_or_sparse(
        self, mask_path: Union[str, List[str]], return_if_sparse: bool = False
    ):
        """
        Load a dense or sparse matrix from a .npz file.

        Parameters
        ----------
        mask_path : str or list of str
            Path to .npz file(s).
        return_if_sparse : bool
            If True, return `was_sparse` as well.
            Only honored when `mask_path` is a single file.

        Returns
        -------
        matrix : np.array
            The loaded matrix/matrices.
        was_sparse : bool
            Whether the matrix was originally a sparse matrix.
        """
        from sparse import load_npz

        def arr_loader(mask_path):
            was_sparse = False
            if not mask_path.endswith(".npz"):
                raise ValueError("Mask path must be a .npz file.")
            mask = np.load(mask_path)
            if "arr_0" in mask:
                mask = mask["arr_0"]
            else:
                mask = load_npz(mask_path).todense()
                was_sparse = True
            return mask, was_sparse

        if isinstance(mask_path, list):
            mask = [arr_loader(file)[0] for file in mask_path]
            mask = (
                self._merge_channels(*mask, pad_to=len(mask))
                if len(mask) > 1
                else mask[0]
            )
            return mask
        mask, was_sparse = arr_loader(mask_path)
        return (mask, was_sparse) if return_if_sparse else mask

    def _write_dense_or_sparse(self, mask, file, sparse=True):
        from sparse import COO, save_npz

        if sparse:
            save_npz(file, COO(mask))
        else:
            np.savez_compressed(file, mask)

    def _load_images(self, img):
        from tifffile import imread

        if isinstance(img, str):
            return imread(img)
        img = self._flatten(img)
        samples = [imread(sample) for sample in img]

        # only stack if needed
        if len(samples) == 1:
            return samples[0]

        return self._merge_channels(*samples, pad_to=len(samples))  # no padding

    def _flatten(self, l):
        if isinstance(l, list) and len(l) > 0 and isinstance(l[0], list):
            return np.concatenate(l).ravel().tolist()
        return l

    def _load_adata(self, file: str, **kwargs) -> anndata.AnnData:
        return anndata.read_h5ad(file, **kwargs)

    def _load_adata_as_df(self, file: str) -> pd.DataFrame:
        adata = self._load_adata(file)
        return self._adata_to_df(adata)

    def _merge_channels(
        self, *img: np.array, axis: int = 2, pad_to: int = 3
    ) -> np.array:
        stack = np.stack(img, axis=axis)
        if stack.shape[axis] < pad_to:
            stack = self._pad_channels(stack, axis, pad_to)
        return stack

    def _pad_channels(
        self, img: np.array, axis: int = 2, pad_to: int = 3
    ) -> np.array:  # pad to 3 (or n) channels
        if img.shape[axis] == pad_to:
            return img
        pad = np.zeros((img.shape[0], img.shape[1], pad_to - img.shape[axis]))
        return np.append(img, pad, axis=2)

    def _adata_to_df(self, adata: anndata.AnnData) -> pd.DataFrame:
        """
        Convert an AnnData object to a pandas DataFrame, keeping .obs (metadata)
        """
        return pd.concat([adata.obs, adata.to_df()], axis=1)

    def store_mask(self, mask):
        assert mask is not None, "Mask must not be None."
        if mask.ndim < 3:
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        if self.mask is None:
            self.mask = mask
        else:
            self.mask = np.concatenate((self.mask, mask), axis=2)


class Mask(DataLoader):
    """
    The Mask class provides methods to load, store, and manipulate masks.

    Attributes
    ----------
    mask : Stores a the numpy array of the mask
    shape : Stores the shape of the mask
    mask_names : Stores the mask names.

    Constructor
    ----------
    mask_path : str or list of str
        Path to mask file(s), optional (default `None`)
        One of `mask_arr` or `mask_path` must be provided.
    mask_arr : np.array
        Numpy array of the mask, optional (default `None`).
        One of `mask_arr` or `mask_path` must be provided.
    mask_names : list of str
        Names of the masks, optional (default: `None`).

    Methods
    ----------
    The Mask class provides the following methods:
    `load()`: Loads the image from the img_path argument
    `add_secondary_mask()`: Adds a secondary mask to the mask
    `create_tertiary()` : Creates a tertiary mask from the primary and secondary masks
    `get_mask()`: Returns the mask
    `shuffle_mask_ids()` : Shuffles the mask ids for plotting
    `plot()` : Plots the mask(s)
    `write()`: Writes the image to the `file` argument
    """

    def __init__(self, mask_path=None, mask_arr=None, mask_names=None):
        super().__init__()
        if mask_path is None and mask_arr is None:
            raise ValueError("Either mask_path or mask_arr must be provided.")
        self.mask = mask_arr if mask_arr is not None else self.load(mask_path)
        self.shape = self.mask.shape
        self.mask_names = mask_names
        self._create_mask_names()

    def __repr__(self):
        return self.get_mask().__repr__()

    def _create_mask_names(self):
        # Logic: create mask names only if missing, otherwise append new names in format Mask{i}
        # If creating new ones, check if mask has 3 channels and assign default names, otherwise
        # default back to Mask{i}.
        mask_names = self.mask_names
        ndim = self.get_mask().ndim
        if mask_names is None:
            if ndim != 2 and ndim == 3 and self.shape[2] == 3:
                self.mask_names = ["Cell", "Nucleus", "Cytoplasm"]
            elif ndim != 2 and ndim == 3 or ndim != 2:
                self.mask_names = [
                    f"Mask{i}" for i in range(1, self.get_mask().shape[2] + 1)
                ]
            else:
                self.mask_names = ["Mask"]
        elif ndim == 3 and len(self.mask_names) < self.get_mask().shape[2]:
            for i in range(self.shape[2] - len(self.mask_names)):
                self.mask_names.append(f"Mask{i + len(self.mask_names) + 1}")

    def filter_union(self):
        """Filter mask IDs that are not present in all masks"""
        mask = self.get_mask()
        if mask.ndim == 3:
            ids = [np.unique(mask[:, :, i]) for i in range(mask.shape[2])]
            ids = np.hstack(ids)
            ids, counts = np.unique(ids, return_counts=True)
            id_remove = np.unique(np.where(counts < mask.shape[2], ids, 0))
            id_remove = id_remove[id_remove != 0]
            np.where(np.isin(mask, id_remove), 0, mask)
            filtered_mask = np.where(np.isin(mask, id_remove), 0, mask)
            self.mask = filtered_mask

    def store_mask(self, mask):
        super().store_mask(mask)
        self.shape = self.mask.shape
        self._create_mask_names()
        return self.get_mask()

    def add_secondary_mask(self, mask: np.ndarray):
        """
        Store a secondary mask.

        Parameters:
        ----------
        mask : np.ndarray
            Numpy array of the mask
        """
        self.store_mask(mask)

    def create_tertiary(self, index_larger=0, index_smaller=1):
        """
        Create tertiary mask from two masks.

        Parameters
        ----------
        index_larger : int
            Index of the mask that is larger than the other mask
        index_smaller : int
            Index of the mask that is smaller than the other mask

        Returns
        -------
        np.ndarray
            Tertiary mask, usually representing cytoplasm
        """
        if self.get_mask().ndim < 3:
            raise ValueError(
                "Mask must have at least 3 dimensions. Did you forget to add the secondary mask?"
            )
        if self.shape[2] < 2:
            raise ValueError(
                "Mask must have at least 2 masks. Did you forget to add the secondary mask?"
            )

        new_mask = (
            self.get_mask()[:, :, index_larger] - self.get_mask()[:, :, index_smaller]
        )
        self.store_mask(new_mask)
        return new_mask

    def get_mask(self, mask_index=None):
        """Return the mask as 2D or 3D depending on if an index was selected"""
        if self.mask.ndim < 3 or mask_index is None:
            return self.mask
        return self.mask[:, :, mask_index]

    def shuffle_mask_ids(self, mask_index=0):
        """Shuffle mask ids before plotting."""

        cur_mask = self.get_mask(mask_index=mask_index)
        ids = np.unique(cur_mask)
        ids = ids[ids != 0]
        new_order = np.copy(ids)
        np.random.shuffle(new_order)
        tab = dict(zip(ids, new_order))
        tab[0] = 0
        new_ids = np.copy(cur_mask)
        for i in ids:
            new_ids[cur_mask == i] = tab[i]
        return new_ids

    def plot(self, mask_index=None, figsize=(10, 10)):
        """Plot the mask"""
        import matplotlib.pyplot as plt
        from skimage.color import label2rgb

        NCOL = 2
        mask_shuffled = self.shuffle_mask_ids(mask_index=mask_index)
        if mask_shuffled.ndim == 2:
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)
            ax.imshow(label2rgb(mask_shuffled))
        else:
            n_subplots = mask_shuffled.shape[2]
            n_rows = np.ceil(n_subplots / NCOL).astype(int)
            fig, ax = plt.subplots(
                n_rows, NCOL, sharex=True, sharey=True, figsize=figsize
            )
            ax_rav = ax.ravel()
            for i in range(n_subplots):
                ax_rav[i].imshow(label2rgb(mask_shuffled[:, :, i]))
            fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        return fig, ax

    def write(self, file, sparse=True):
        """
        Write all stored masks to a file.

        Parameters
        ----------
        file : str or list of str
            Path of file(s)
        sparse : bool, optional
            Whether to write the mask as a sparse matrix (default: `True`)
        """
        self._write_dense_or_sparse(self.get_mask(), file, sparse)


class MultiMask(Mask):
    """
    The MultiMask class provides methods to load, store, and manipulate masks containing
    one-to-multi mappings, e.g. multi-nucleated cells.


    Constructor
    ----------
    mask_path : str or list of str
        Path to mask file(s), optional (default `None`)
        One of `mask_arr` or `mask_path` must be provided.
    mask_arr : np.array
        Numpy array of the mask, optional (default `None`).
        If using `mask_arr`, mask must be 3D and the first layer must be the supermask.
        One of `mask_arr` or `mask_path` must be provided.
    mask_names : list of str
        Names of the masks, optional (default: `None`).

    """

    def store_mask(self, mask, idx=None):
        assert mask is not None, "Mask must not be None."
        if self.mask is None:
            self.mask = mask
        if self.mask.ndim < 3:
            self.mask = self.mask[:, :, np.newaxis]
        if idx is None:
            idx = self.mask.shape[2]
        self.mask = np.insert(self.mask, idx, mask, axis=2)
        self.shape = self.mask.shape

    def filter_union(self, index_larger=0, index_smaller=1):
        supmask = self.get_mask(index_larger)
        submask = self.get_mask(index_smaller)
        empty = []
        for supobject in np.unique(supmask):
            supss = supmask[supmask == supobject]
            subss = submask[supmask == supobject]
            if not np.any(np.logical_and(supss, subss)):
                empty.append(supobject)
        idx = np.isin(supmask, empty)
        np.putmask(supmask, idx, 0)
        self.mask[:, :, index_larger] = supmask

    def create_tertiary(self, index_larger=0, index_smaller=1):
        supmask = self.get_mask(index_larger).copy()
        submask = self.get_mask(index_smaller)
        idx = np.logical_and(supmask, submask)
        np.putmask(supmask, idx, 0)
        self.store_mask(supmask)


class Image(DataLoader):
    """
    The Image class provides methods to load, store, manipulate and segment images.

    Attributes
    ----------
    img : Stores the image loaded from the img_path argument
    shape : Stores the shape of the image
    mask : Stores a Mask object if the mask_path argument is not `None`
    channel_names : Stores the names of the channels in the image.
    The default names are `W1` to `Wn` for n-channel images.

    Constructor
    ----------
    img_path : str or list of str
        Path to image file(s).
    channel_names : list of str
        Names of the image channels (default: `W1` to `Wn`).
    mask_path : str or list of str
        Path to mask file(s), optional (default: `None`)
    mask_names : list of str
        Names of the masks, optional (default: `None`).

    Methods
    ----------
    The Image class provides the following methods:
    `load()`: Loads the image from the img_path argument
    `rescale_color()`: Rescales the image to the grey_levels argument
    """

    def __init__(
        self,
        img_path,
        channel_names=None,
        mask_path=None,
        mask_names=None,
    ):
        super().__init__()
        if isinstance(img_path, str):
            img_path = [img_path]
        self.img = self.load(img_path)
        self.normalise()
        self.shape = self.img.shape
        self.mask = (
            Mask(mask_path, mask_names=mask_names) if mask_path is not None else None
        )

        # store image and mask names
        if channel_names is None:
            self.channel_names = [f"W{i}" for i in range(1, len(img_path) + 1)]

    def __repr__(self):
        return self.img.__repr__()

    def get_image(self, channel=None):
        if channel is None:
            return self.img
        if self.img.ndim == 2:
            return self.img
        return self.img[:, :, 0] if self.img.shape[2] == 1 else self.img[:, :, channel]

    def store_img(self, img):
        """
        Store images in the `img` attribute. If other channels/images are already
        present, concatenate them.

        Parameters
        ----------
        img : np.array
            Image to store.
        """
        assert img is not None, "img must not be None."
        if img.ndim < 3:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        self.img = img if self.img is None else np.concatenate((self.img, img), axis=2)
        self.shape = self.img.shape

    def plot(self, channels=None, masks=None, figsize=(10, 10), ncols=3):
        """
        Plot image channels alongside masks

        Parameters
        ----------
        channels : int or list of int, optional
            Which channels to plot, by default None (i.e. all channels)
        masks : int or list of int, optional
            Which masks to plot, by default None (i.e. no masks)
        figsize : tuple, optional
            Size of the figure, by default (10, 10)
        ncols : int, optional
            Number of columns in the figure, by default 3

        Returns
        -------
        plt.Figure
            The figure object
        plt.Axes
            The axes object
        """
        import matplotlib.pyplot as plt
        from skimage.color import label2rgb

        cur_img = self.get_image(channel=channels)

        n_masks = 0

        if masks is not None:
            cur_mask = self.mask.get_mask(mask_index=masks)
            if cur_mask.ndim == 2:
                cur_mask = cur_mask.reshape(*cur_mask.shape, 1)
            n_masks = cur_mask.shape[2]

        n_channels = 1 if cur_img.ndim == 2 else cur_img.shape[2]

        n_total = n_channels + n_masks

        n_rows = np.ceil(n_total / ncols).astype(int) if n_total >= ncols else 1

        fig, ax = plt.subplots(
            n_rows, min(ncols, n_total), sharex=True, sharey=True, figsize=figsize
        )

        # Flatten axes array to more easily assign axes
        ax_rav = np.ravel(ax)

        # Loop over images and add to figure
        for i in range(n_total):
            if i < n_channels:
                ax_rav[i].imshow(cur_img[:, :, i], cmap="gray")
            else:
                ax_rav[i].imshow(label2rgb(cur_mask[:, :, i - n_channels]))

        # Remove empty axes
        if n_total < ncols * n_rows:
            for i in range(n_total, n_rows * ncols):
                fig.delaxes(ax_rav[i])

        # Remove whitespace between subplots
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()

        # Keep axes numbers only for left column and bottom row
        for i in range(n_total):
            if i % ncols != 0:
                ax_rav[i].yaxis.set_visible(False)
            if i < n_total - ncols:
                ax_rav[i].xaxis.set_visible(False)

        return fig, ax

    def normalise(self):
        """Normalise the image in-place to take values 0-1"""
        ceil = np.max(self.img)
        if ceil == 0:
            raise ValueError("Image is empty")
        elif ceil < 1 + 1e-6:
            pass
        elif ceil < 256:
            self.img = self.img / 255
        elif ceil < 65536:
            self.img = self.img / 65535
        else:
            raise ValueError("Image has unrecognised bit-depth, please use 8 or 16 bit")

    def rescale_color(self, grey_levels=256):
        """Quantize the image to a given number of grey levels in-place

        Args:
            img (matrix): Image to be quantized
            grey_levels (int): Number of grey levels to quantize the image to

        """
        from skimage.exposure import rescale_intensity

        if grey_levels is None:
            return self.img
        if self.img.ndim == 2:
            self.img = self.img.reshape(self.img.shape[0], self.img.shape[1], 1)
        for i in range(self.img.shape[2]):
            self.img[:, :, i] = rescale_intensity(
                self.img[:, :, i],
                out_range=(0, grey_levels - 1),
            ).astype(np.uint8)
        return self.img

    def segment(self, model, **kwargs):
        """
        Segment an image using a chosen model.

        Parameters
        ----------
        model : Model
            A segmentation model implementing the `segment` method.

        Returns
        -------
        np.ndarray
            Newly created mask. The mask is also stored in the `mask` attribute.
        """
        new_mask = model.segment(self.img, **kwargs)
        self.store_mask(new_mask)
        return new_mask

    def store_mask(self, mask):
        """Store a mask in the `mask` attribute"""
        if self.mask is None:
            self.mask = mask if isinstance(mask, Mask) else Mask(mask_arr=mask)
        else:
            self.mask.store_mask(mask)

    def subsegment(
        self,
        model,
        channel,
        mask_index=0,
        mapping="one-to-one",
        diameter=100,
        pad_edges=20,
        **kwargs,
    ):
        """
        Segment an image using a chosen model.

        Parameters
        ----------
        model : Model
            A segmentation model implementing the `segment` method.
        channel : int
            Index of channel to segment.
        mask_index : int, optional
            Index of mask to subsegment (default: 0)
        mapping : str, optional
            The mapping between the mask and the image.
            Must be one of "one-to-one", "one-to-many", "return" and "discard".
        diameter : int, optional
            Some models require a diameter to segment (default: 100)
        pad_edges : int, optional
            Some models require padding the edges of the image (default: 20)
        return_hit_summary : bool, optional
            Whether to return how many objects had zero, one or multiple hits.
            If True, will additionally return a dictionary with keys "zero", "one" and "many"
        kwargs : dict
            Other arguments to pass to the model.

        Returns
        -------
        Secondary mask. The mask is also stored in the `mask` attribute.
        """
        new_mask = model.subsegment(
            self,
            mask=self.mask,
            mask_index=mask_index,
            channel=channel,
            mapping=mapping,
            diameter=diameter,
            pad_edges=pad_edges,
            **kwargs,
        )
        multi_cells = None
        if isinstance(new_mask, tuple):
            multi_cells = new_mask[1]
            new_mask = new_mask[0]
        self.mask.store_mask(new_mask)
        return new_mask if multi_cells is None else (new_mask, multi_cells)

    def create_tertiary_mask(self):
        self.mask.create_tertiary()
