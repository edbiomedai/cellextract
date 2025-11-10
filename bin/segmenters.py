#!/usr/bin/env python
"""Segmentation Classes for CellPose and other methods."""
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from skimage.measure import regionprops

from images import Image, Mask


class SegmentationModel(ABC):
    """Abstract class for segmentation methods.

    Attributes
    ----------
    `model_name` : Name of the model used for segmentation
    `model` : Model used for segmentation

    Constructor
    ----------
    model : str
        Name of the model used for segmentation

    Methods
    -------
    The SegmentationModel class provides the following methods:
    `segment` : Segment image
    `subsegment` : Segment all channels in an image
    """

    def __init__(self, model):
        self.model_name = model
        # Initialise model with chosen method
        self.model = None

    @abstractmethod
    def _predictfun(self, img, channels, diameter):
        ...

    def _loop_over_masks(self, img, mask, layer=None):
        """Loop over objects in mask, returning the cropped image and coordinates"""
        mask_selected = self._select_layer(mask, layer)
        for prop in regionprops(mask_selected, img):
            yield prop["label"], prop["coords"]

    def _select_layer(self, img, layer=None):
        if layer is None or img.ndim == 2:
            return img
        if isinstance(layer, list):
            layer = [l for l in layer if l is not None]
        else:
            layer = [layer]
        return img[:, :, layer]

    def _remove_border(self, mask, px=0):
        """Remove objects touching the border of the image"""
        from skimage.segmentation import clear_border

        clear_border(mask, out=mask, buffer_size=px)

    def segment(self, img, channel, channel_aux=None, diameter=100, **kwargs):
        """
        Create a primary segmentation mask from one or two channels.

        Parameters
        ----------
        img : np.ndarray or List[np.ndarray]
            The image to be segmented
        channel_base : int
            The channel to be used for segmentation
        channel_aux : int
            The channel to aid in segmentation, can be `None`
        diameter : int, optional
            Expected diameter of objects, by default 100

        Returns
        -------
        np.ndarray
            Secondary mask
        """
        if isinstance(img, np.ndarray):
            cur_img = self._select_layer(img, layer=[channel, channel_aux])
        else:
            cur_img = [self._select_layer(i, layer=[channel, channel_aux]) for i in img]
        channels = [0, 0] if channel_aux is None else [1, 2]
        mask = self._predictfun(cur_img, channels=channels, diameter=diameter, **kwargs)
        if isinstance(mask, list):
            mask = [Mask(mask_arr=m) for m in mask]
        else:
            mask = Mask(mask_arr=mask)
        return mask

    def _crop_to_coords(
        self,
        img,
        coords,
    ):
        """Crop image using coordinates returned by `regionprops`"""
        rows = coords[:, 0]
        cols = coords[:, 1]
        if img.ndim == 3:
            return img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1, :]
        return img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    def _pad_edges(self, img, pad_edges=20):
        """Pad edges of an image. Improves segmentation with models that require background.
        Works under the assumption that background is 0."""
        if pad_edges == 0:
            return img
        if img.ndim == 3:
            npad = ((pad_edges, pad_edges), (pad_edges, pad_edges), (0, 0))
        else:
            npad = ((pad_edges, pad_edges), (pad_edges, pad_edges))
        return np.pad(img, pad_width=npad, mode="constant", constant_values=0)

    def _unpad_edges(self, img, pad_edges=20):
        """Remove padding from an image."""
        if img.ndim == 3:
            return img[pad_edges:-pad_edges, pad_edges:-pad_edges, :]
        return img[pad_edges:-pad_edges, pad_edges:-pad_edges]


class CellPoseSegmenter(SegmentationModel):
    """CellPose class to segment images.

    Attributes
    ----------
    `model_name` : Name of the model used for segmentation
    `model` : Model used for segmentation

    Constructor
    ----------
    model : str
        Name of the model used for segmentation

    Methods
    -------
    The CellPoseSegmenter class provides the following methods:
    `segment` : Segment image, e.g. cell classification
    `subsegment` : Subsegment a mask, e.g. nuclei detection only within cells
    """

    def __init__(self, model: str):
        import platform

        from cellpose import models
        from torch.cuda import is_available

        super().__init__(model)  # Store model name
        gpu = False
        if platform.system() == "Darwin" and platform.processor() == "arm":
            import torch

            try:
                device = torch.device("mps")
            except RuntimeError:
                device = None
        else:
            gpu = is_available()
            device = None
        self.model = models.CellposeModel(gpu=gpu, model_type=model, device=device)

    def _predictfun(
        self,
        img: Union[np.ndarray, List[np.ndarray]],
        channels: np.ndarray,
        diameter: int,
        **kwargs,
    ):
        mask = self.model.eval(img, channels=channels, diameter=diameter, **kwargs)[0]
        if isinstance(mask, list):
            for m in mask:
                self._remove_border(m)
            mask = [i.reshape((*i.shape, 1)) for i in mask]
        else:
            self._remove_border(mask)  # Remove objects touching the border
            mask = mask.reshape((*mask.shape, 1))
        return mask

    def subsegment(
        self,
        img,
        channel,
        mask,
        mask_index=0,
        mapping="one-to-one",
        diameter=40,
        pad_edges=20,
        **kwargs,
    ):
        """
        Create a secondary mask for each object in the primary mask.

        Parameters
        ----------
        img : Image
            The image to be segmented
        channel : int
            The channel to be used for segmentation
        mask : Mask
            Mask to subsegment
        mask_index : int, optional
            Index of the mask to subsegment, by default 0
        mapping : str
            Whether each super-object should map to one or multiple sub-objects.
            Must be one of "one-to-one", "one-to-many", "discard", and "return".
            "one-to-one" will label each sub-object with the same label as the super-object.
            "one-to-many" will label each sub-object with a unique label.
            "discard" will discard any super-object that has multiple sub-objects.
            "return" will return a tuple of numpy arrays, where the first element contains
            single-nucleated cells and the second element contains multi-nucleated cells.
        diameter : int, optional
            Expected diameter of objects, by default 40
        pad_edges : int, optional
            Number of pixels to pad the edges of the image, by default 20
            Some models perform better when there is sufficient background.
        kwargs : dict
            Other arguments to pass to the model.

        Returns
        -------
        Secondary mask
        Multi-hit secondary mask (if `mapping` is "return")
        """
        cur_img = img.get_image() if isinstance(img, Image) else img
        cur_img = self._select_layer(cur_img, channel)
        mask_arr = mask.get_mask() if isinstance(mask, Mask) else mask
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, mask_index]
        submask = np.zeros_like(mask_arr)
        if mapping == "return":
            submask_multi = np.zeros_like(mask_arr)

        hit_counts = {"zero": 0, "one": 0, "many": 0}
        cropped_objects = {
            "cropped_img": [],
            "cropped_supermask": [],
            "label": [],
            "coords": [],
        }
        for label, coords in self._loop_over_masks(cur_img, mask_arr):
            # Crop image and mask to object
            cur_img_cropped = self._crop_to_coords(cur_img, coords).copy()
            mask_cropped = self._crop_to_coords(mask_arr, coords).copy()

            # Remove all other pixels not in super-mask
            np.putmask(cur_img_cropped, mask_cropped != label, 0)
            np.putmask(mask_cropped, mask_cropped != label, 0)

            # Pad edges to improve segmentation
            cur_img_cropped = self._pad_edges(cur_img_cropped, pad_edges=pad_edges)
            cropped_objects["cropped_img"].append(cur_img_cropped)
            cropped_objects["cropped_supermask"].append(mask_cropped)
            cropped_objects["label"].append(label)
            cropped_objects["coords"].append(coords)

        # Perform segmentation on cropped images
        new_mask = self.segment(
            cropped_objects["cropped_img"], channel=0, diameter=diameter, **kwargs
        )

        if isinstance(new_mask, Mask):
            new_mask = [new_mask.get_mask()]
        else:
            new_mask = [i.get_mask() for i in new_mask]

        # Loop over new masks and add into one large mask
        for cur_new_mask, super_mask_cropped, label, coords in zip(
            new_mask,
            cropped_objects["cropped_supermask"],
            cropped_objects["label"],
            cropped_objects["coords"],
        ):
            # Remove padding
            new_mask_cropped = self._unpad_edges(cur_new_mask, pad_edges=pad_edges)

            # Count number of sub-objects
            n_sub_objects = sum(np.unique(new_mask_cropped) != 0)

            if n_sub_objects == 0:
                # No objects found
                print("No objects found for label", label)
                hit_counts["zero"] += 1
                continue
            if n_sub_objects > 1:
                msg = f"Multihit for label {label} with {n_sub_objects} objects."
                if mapping == "one-to-one":
                    msg += " Will merge these objects into one."
                if mapping == "discard":
                    msg += " Will discard these objects."
                if mapping == "return":
                    msg += " Will return these objects separately."
                # print(msg)
                hit_counts["many"] += 1
                if mapping == "discard":
                    continue
            else:
                hit_counts["one"] += 1

            # Ensure 2D
            if new_mask_cropped.ndim == 3:
                new_mask_cropped = new_mask_cropped[:, :, 0]
            if super_mask_cropped.ndim == 3:
                super_mask_cropped = super_mask_cropped[:, :, 0]

            # Change label to match super-mask, if desired
            if mapping in "one-to-one" or n_sub_objects == 1:
                new_mask_cropped[
                    np.logical_and(super_mask_cropped, new_mask_cropped)
                ] = label

            # Crop large submask to new assignment area
            if mapping == "return" and n_sub_objects > 1:
                submask_cropped = self._crop_to_coords(submask_multi, coords)
            else:
                submask_cropped = self._crop_to_coords(submask, coords)

            # prevent out-of-bounds predictions
            if mapping == "one-to-one":
                obj = np.logical_and(
                    np.equal(super_mask_cropped, label), new_mask_cropped
                )
            else:
                obj = np.logical_and(super_mask_cropped, new_mask_cropped)

            # Add new mask predictions
            # This operates on `submask` in-place
            submask_cropped[obj] = new_mask_cropped[obj]

        # Cleanup multi mapping masks
        if mapping == "one-to-many":
            supersub = np.stack([mask_arr, submask], axis=2)
            submask = self.fixup_mask_mulitnucleated(supersub, super_mask_index=0)
        elif mapping == "return":
            supersub = np.stack([mask_arr, submask_multi], axis=2)
            submask_multi = self.fixup_mask_mulitnucleated(
                supersub, super_mask_index=0, return_only_submask=False
            )

        # Sanity check
        elif mapping == "one-to-one":
            mismatches = np.logical_not(
                np.logical_or(np.equal(mask_arr, submask), np.equal(submask, 0))
            )
            if np.any(mismatches):
                raise ValueError(
                    "Submask does not match mask."
                    + "\nThis means something went wrong, please contact the developer."
                )

        print("Hit summary:")
        print(
            "\n".join(
                [
                    ": ".join([i, str(j)])
                    for i, j in zip(hit_counts.keys(), hit_counts.values())
                ]
            )
        )
        return (submask, submask_multi) if mapping == "return" else submask

    def fixup_mask_mulitnucleated(
        self, mask, super_mask_index=0, return_only_submask=True
    ):
        """Fix up masks containing multiple subobjects per object"""
        cur_mask = mask if isinstance(mask, np.ndarray) else mask.get_mask()
        if mask.ndim == 2:
            return mask
        elif mask.shape[2] > 2:
            raise ValueError(
                "Currently only supports primary and secondary masks. Run this before creating tertiary masks."
            )

        sub_mask_index = 1 if super_mask_index == 0 else 0
        sub_mask = cur_mask[:, :, sub_mask_index]
        new_sub_mask = np.zeros_like(cur_mask[:, :, sub_mask_index])
        cur_sub_counter = 1
        for label, coords in self._loop_over_masks(
            cur_mask, cur_mask[:, :, super_mask_index]
        ):
            mask_cropped = self._crop_to_coords(cur_mask, coords)
            super_mask_cropped = mask_cropped[:, :, super_mask_index]
            obj = super_mask_cropped == label
            new_sub_mask_cropped = self._crop_to_coords(new_sub_mask, coords)
            sub_mask_cropped = self._crop_to_coords(sub_mask, coords)
            for i in np.unique(sub_mask_cropped[obj]):
                if i == 0:
                    continue
                new_sub_mask_cropped[
                    np.logical_and(obj, sub_mask_cropped == i)
                ] = cur_sub_counter
                cur_sub_counter += 1

        if return_only_submask:
            return new_sub_mask
        elif sub_mask_index == 1:
            return np.stack([cur_mask[:, :, super_mask_index], new_sub_mask], axis=2)
        else:
            return np.stack([new_sub_mask, cur_mask[:, :, super_mask_index]], axis=2)
