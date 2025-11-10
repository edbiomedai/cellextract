#!/usr/bin/env python
import numpy as np
from skimage.morphology import remove_small_objects

from images import Image, MultiMask
from parsers import BaseSuperParser
from sample_sheet import parse_sample_sheet
from segmenters import CellPoseSegmenter


def segment_cellpose_sheet(
    sheet_path,
    idx,
    nuc_model,
    cell_model,
    nuc_channel,
    cell_channel,
    nuc_diameter,
    cell_diameter,
    min_cell_size,
):
    # load models only once to save IO
    cellSegmenter = CellPoseSegmenter(cell_model)
    nucSegmenter = CellPoseSegmenter(nuc_model)

    for row in parse_sample_sheet(sheet_path, idx=idx):
        segment_cellpose(
            img_path=row["Channel"],
            nuc_model=nucSegmenter,
            cell_model=cellSegmenter,
            nuc_channel=nuc_channel,
            cell_channel=cell_channel,
            nuc_diameter=nuc_diameter,
            cell_diameter=cell_diameter,
            min_cell_size=min_cell_size,
            out=row["Mask"][0],
            out_multi_cells=row["Mask"][1],
        )


def segment_cellpose(
    img_path,
    nuc_model,
    cell_model,
    nuc_channel,
    cell_channel,
    nuc_diameter,
    cell_diameter,
    min_cell_size,
    out,
    out_multi_cells,
):
    print(img_path)
    cellSegmenter = (
        CellPoseSegmenter(cell_model) if isinstance(cell_model, str) else cell_model
    )
    mapping = "return" if out_multi_cells else "discard"
    # only load in channels needed for segmentation to reduce IO.
    if len(cell_channel.split(",")) > 1:
        cell_channel = [int(i) for i in cell_channel.split(",")]
    else:
        cell_channel = [int(cell_channel)]
    img_path_subset = [img_path[nuc_channel], *[img_path[i] for i in cell_channel]]
    img = Image(img_path=img_path_subset)

    if len(cell_channel) > 1:
        # maximum intensity projection of all but nuclear channel,
        # then use that channel for segmentation
        seg_img = img.get_image()[:, :, 1:].max(axis=2)
        img.img[:, :, 1] = seg_img
        img.img = img.img[:, :, [0, 1]]
        img.shape = img.img.shape

    # nuc channel must be first channel, otherwise cellpose uses
    # grayscale segmentation. 0-based indexing.
    img.segment(
        cellSegmenter,
        channel=1,
        channel_aux=0,
        diameter=cell_diameter,
        channel_axis=2,
    )

    img.mask.mask = remove_small_objects(img.mask.mask, min_cell_size)
    print(len(np.unique(img.mask.mask)) - 1, "cells detected")
    nucSegmenter = (
        CellPoseSegmenter(nuc_model) if isinstance(nuc_model, str) else nuc_model
    )
    _, multi = img.subsegment(
        nucSegmenter, channel=nuc_channel, diameter=nuc_diameter, mapping=mapping
    )
    if mapping == "return":
        multi_mask = MultiMask(mask_arr=multi[:, :, [0, 1]])
        multi_mask.filter_union()
        multi_mask.create_tertiary()
        multi_mask.write(out_multi_cells)

    img.create_tertiary_mask()
    img.mask.filter_union()

    if img.mask.get_mask()[:, :, 0].sum() == 0:
        print("No nuclei detected, will not produce output file.")
    else:
        img.mask.write(out)


if __name__ == "__main__":
    desc = "Segment cells and nuclei"
    reqs = ["img_path"]
    bparser = BaseSuperParser(description=desc, reqs_single=reqs)

    single = bparser.get_single_parser()

    single.add_args(
        "--out",
        help="Path to save mask, should end in `.npz`",
        required=True,
    )

    bparser.add_args_to_both(
        "--nuc_channel",
        help="Index of nuclear channel (0-based) (default: 0)",
        type=int,
        default=0,
    )

    bparser.add_args_to_both(
        "--cell_channel",
        help="Index of cell channel (0-based) (default: 3). If given a comma-separated list, will combine channels into one by maximum intensity projection.",
        type=str,
        default="3",
    )

    bparser.add_args_to_both(
        "--nuc_model",
        help="Name of pretrained model to use for nuclei (default: nuclei)",
        type=str,
        default="nuclei",
    )

    bparser.add_args_to_both(
        "--cell_model",
        help="Name of pretrained model to use for cells (default: cyto2)",
        type=str,
        default="cyto2",
    )

    bparser.add_args_to_both(
        "--nuc_diameter",
        help="Diameter of nuclei (default: 40)",
        type=int,
        default=40,
    )

    bparser.add_args_to_both(
        "--cell_diameter",
        help="Diameter of cells (default: 100)",
        type=int,
        default=100,
    )

    bparser.add_args_to_both(
        "--min_cell_size",
        help="Minimum volume of cells (default: 100)",
        type=int,
        default=100,
    )

    single.add_args(
        "--out_mask_plot",
        help="Path to save mask plot (default: None, i.e. will not be saved)",
        type=str,
        default=None,
    )

    single.add_args(
        "--out_multi_cells",
        help="Path to save masks for multi-nucleated cells (default: None, i.e. will not be saved)",
        type=str,
        default=None,
    )

    args = bparser.get_args()
    if args.mode == "multi":
        # process multiple image sets
        segment_cellpose_sheet(
            args.sample_sheet,
            args.index,
            args.nuc_model,
            args.cell_model,
            args.nuc_channel,
            args.cell_channel,
            args.nuc_diameter,
            args.cell_diameter,
            args.min_cell_size,
        )
    # process single image
    else:
        segment_cellpose(
            args.img,
            args.nuc_model,
            args.cell_model,
            args.nuc_channel,
            args.cell_channel,
            args.nuc_diameter,
            args.cell_diameter,
            args.min_cell_size,
            args.out,
            args.out_multi_cells,
        )
