#!/usr/bin/env python
# Import libraries
import os
import warnings

import numpy as np
from tifffile import TiffFile, imwrite

from images import Mask, MultiMask
from parsers import BaseMultiParser
from sample_sheet import parse_sample_sheet


def get_background(img, mask, multi_mask=None, mask_index=0):
    cell_mask = mask[:, :, mask_index]
    if multi_mask is not None:
        cell_mask = np.logical_or(cell_mask, multi_mask[:, :, mask_index]).astype(int)

    background = img[cell_mask == 0, :]
    return background


def background_subtract(
    img, mask, multi_mask=None, mask_index=0, background_correction_quantile=0.25
):
    """
    Subtract background of image at a given quantile.

    Parameters
    ----------
    img : np.ndarray
        Image to subtract background from
    mask : np.ndarray
        Mask to use for background subtraction
    multi_mask : MultiMask, optional
        MultiMask to add to stored mask (for accurate background identification), by default None
    mask_index : int, optional
        Index of mask to use for background subtraction, by default 0
    background_correction_quantile : float, optional
        Quantile of background pixel intensity, by default 0.25
    """

    background = get_background(img, mask, multi_mask, mask_index)
    background = np.quantile(background, background_correction_quantile, axis=0)

    for i in range(img.shape[2]):
        bg = background[i]
        img[:, :, i] = (
            img[:, :, i] - np.quantile(bg, background_correction_quantile)
        ).clip(0)
    return img


def background_correct(
    img_path,
    mask_path,
    multi_mask=None,
    mask_index=0,
    background_correction_quantile=0.25,
):
    # Comma seperated lists may be passed when using batched computations
    if isinstance(mask_path, list) and "," in mask_path[0]:
        mask_path = mask_path[0].split(",")
    if multi_mask is not None:
        if isinstance(multi_mask, list) and "," in multi_mask[0]:
            multi_mask = multi_mask[0].split(",")
    if isinstance(img_path, list) and "," in img_path[0]:
        img_path = img_path[0].split(",")
    if isinstance(img_path, str):
        img_path = [img_path]

    # Load images with TiffFile instead of Image class to preserve metadata
    img = [TiffFile(f) for f in img_path]
    # Extract values for correction
    img_data = np.stack([f.asarray() for f in img], axis=-1)
    img_data = img_data / 65535

    # Load in masks
    mask = Mask(mask_path=mask_path).get_mask()

    if multi_mask is not None:
        multi_mask = MultiMask(mask_path=multi_mask).get_mask()

    return img, background_subtract(
        img_data,
        mask,
        multi_mask,
        mask_index=mask_index,
        background_correction_quantile=background_correction_quantile,
    )


def background_correct_sheet(
    sheet_path,
    idx,
    mask_index=0,
    background_correction_quantile=0.25,
):
    for row in parse_sample_sheet(sheet_path, idx=idx):
        # Check that masks exist
        mask_path = row["Mask"][0]
        if not os.path.exists(mask_path):
            warnings.warn(f"{mask_path} does not exist. Skipping sample.")
            continue
        mask_multi_path = row["Mask"][1]
        if not os.path.exists(mask_multi_path):
            mask_multi_path = None

        raw_img, img_correct = background_correct(
            img_path=row["Channel"],
            mask_path=mask_path,
            multi_mask=mask_multi_path,
            mask_index=mask_index,
            background_correction_quantile=background_correction_quantile,
        )

        img_correct = np.split(img_correct, img_correct.shape[-1], axis=-1)
        img_correct = [np.squeeze(img) for img in img_correct]

        for i, (raw_img, cor_img) in enumerate(zip(raw_img, img_correct)):
            out_file = row["ImageCorrected"][i]
            # filter out difficult to write keys
            metadata = {
                k: v
                for k, v in raw_img.stk_metadata.items()
                if isinstance(v, (int, float, str, list))
            }

            out_folder = os.path.dirname(out_file)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            imwrite(out_file, cor_img, metadata=metadata)


if __name__ == "__main__":
    desc = "Subtract background intensity per channel."
    bparser = BaseMultiParser(description=desc, reqs=[])
    bparser.add_args(
        "--background_quantile",
        help="Quantile for background subtraction",
        default=0.25,
        type=float,
    )
    args = bparser.get_args()

    background_correct_sheet(
        args.sample_sheet,
        args.index,
        background_correction_quantile=args.background_quantile,
    )
