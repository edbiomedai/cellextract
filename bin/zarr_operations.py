"""Combine images, labels, and background correction
into one zarr file that can be visualised with MoBIE.
Loosely following the ideas from: https://github.com/fmi-faim/faim-ipa/blob/main/src/faim_ipa/Zarr.py,
with thanks to Joel Lüthi for initial pointers.
"""

import re
from glob import glob
from pathlib import Path
from warnings import warn

import numpy as np
import zarr
from faim_ipa.Zarr import write_labels_to_group
from natsort import natsorted
from numcodecs import Blosc
from ome_zarr.io import parse_url
from skimage.measure import label
from tqdm import tqdm

from images import Mask, MultiMask


def _loop_over_wells(zarr_url, mode="r"):
    zarr_url = Path(zarr_url).resolve()
    storage = parse_url(path=zarr_url, mode=mode).store
    plate = zarr.group(store=storage)
    wells = plate.attrs.asdict()["plate"]["wells"]
    wells = natsorted(wells, key=lambda x: x["path"])
    for well in wells:
        well_name = "".join(well["path"].split("/"))
        group = plate[well["path"]][0]
        yield well_name, plate[well["path"]][0]


def _fix_labels(masks):
    offset = 0
    for i, m in enumerate(masks):
        if i == 0:
            continue
        else:
            offset += np.max(masks[i - 1])
            m[m != 0] += offset
    return masks


def _stitch(elements, cols=2, rows=2, fixup_labels=True):
    if not elements:
        return elements
    elements = [arr.squeeze() for arr in elements]
    if fixup_labels:
        elements = _fix_labels(elements)

    arr = elements[0]

    is3d = elements[0].ndim == 3
    if is3d:
        channels = elements[0].shape[0]
        vpixels = arr.shape[1] * rows
        hpixels = arr.shape[2] * cols
        out = np.zeros((channels, vpixels, hpixels))
    else:
        vpixels = arr.shape[0] * rows
        hpixels = arr.shape[1] * cols
        out = np.zeros((vpixels, hpixels))
    twidth = hpixels // cols
    theight = vpixels // rows

    for i, element in enumerate(elements):
        if is3d:
            out[
                :,
                (i // cols) * theight : ((i // cols) + 1) * theight,
                (i % cols) * twidth : ((i % cols) + 1) * twidth,
            ] = element
        else:
            out[
                (i // cols) * theight : ((i // cols) + 1) * theight,
                (i % cols) * twidth : ((i % cols) + 1) * twidth,
            ] = element
    return out


def _read_labels(label_files, plate_name, well_name, do_stitch=True, multi_nuc=False):
    if multi_nuc:
        reg = re.compile(f".*{plate_name}_.*_{well_name}_[1-4]_mask_multi.npz")
    else:
        reg = re.compile(f".*{plate_name}_.*_{well_name}_[1-4]_mask.npz")

    files = [f for f in label_files if reg.match(f)]

    if not files:
        warn(f"No labels found for {well_name}")
        return []

    mask = [Mask(f).get_mask() for f in files]
    mask = [np.moveaxis(label, -1, 0) for label in mask]

    if not do_stitch:
        return mask

    if len(mask) != 4:
        raise ArgumentError("Stitching currently only supported for 4 sites")

    return _stitch(mask)


def write_segmentation_to_zarr(
    zarr_url, label_dir, label_names=["Cell", "Nuclei", "Cytoplasm"], multi_nuc=False
):
    """
    Merge segmentation masks from a directory into a zarr file.

    Parameters
    ----------
    zarr_url : str
        The url to the zarr file.
    label_dir : str
        The directory containing the labels.
    label_names : list
        The names of the labels.
    multi_nuc : bool
        Whether to use the multi nucleus masks.
    """
    label_files = natsorted(glob((Path(label_dir).resolve() / "*.npz").as_posix()))
    zarr_url = Path(zarr_url).resolve()
    plate_name = zarr_url.stem

    for well_name, well_img in _loop_over_wells(zarr_url=zarr_url, mode="w"):
        mask = _read_labels(
            label_files=label_files,
            plate_name=plate_name,
            well_name=well_name,
            multi_nuc=multi_nuc,
        )
        assert len(mask) == len(
            label_names
        ), "Number of labels does not match number of label names"

        for i, name in enumerate(label_names):
            name = name if not multi_nuc else f"{name}_multi"
            write_labels_to_group(
                labels=mask[[i], :, :].astype(int),
                labels_name=name,
                parent_group=well_img,
                storage_options=dict(
                    chunks=(540, 540),
                    dimension_separator="/",
                    compressor=Blosc(cname="zstd", clevel=6, shuffle=Blosc.BITSHUFFLE),
                    write_empty_chunks=False,
                ),
            )
