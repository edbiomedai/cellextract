#!/usr/bin/env python
from typing import Tuple

from images import TissueImage
from parsers import BaseSingleParser
from segmenters import StarDistSegmenter


def segment_nuclei(
    img_path,
    out,
    stain_type: str = "H&E",
    model_name: str = "StarDist Nucleus Segmentation",
    pretrained_model: str = "2D_versatile_he",
    implementation: str = "big",
    axes: str = "YXC",
    n_tiles: Tuple[int, int, int] = (10, 10, 1),
    block_size: int = 1024,
    min_overlap: int = 32,
    prob_thresh: float = 0.2,
    nms_thresh: float = 0.4,
):
    img = TissueImage(img_path, stain_type=stain_type)

    stardist_segmenter = StarDistSegmenter(
        model=model_name, pretrained_model=pretrained_model
    )
    img.segment(
        stardist_segmenter,
        implementation=implementation,
        axes=axes,
        n_tiles=n_tiles,
        block_size=block_size,
        min_overlap=min_overlap,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )

    img.mask.write(out)


if __name__ == "__main__":
    desc = "Segment cells and nuclei"
    reqs = ["img_path"]
    bparser = BaBaseSingleParserseParser(description=desc, reqs=reqs)

    bparser.add_args(
        "--out",
        help="Path to save mask, should end in `.npz`",
        required=True,
    )

    bparser.add_args(
        "--stain_type",
        help="The type of staining used in the image, should be `H&E` or `H-DAB` (default: H&E)",
        type=str,
        default="H&E",
    )

    bparser.add_args(
        "--pretrained_model",
        help="Name of pretrained model to use for nuclei (default: 2D_versatile_he)",
        type=str,
        default="2D_versatile_he",
    )

    bparser.add_args(
        "--implementation",
        help="Implementation to use for segmentation (default: big)",
        type=str,
        default="big",
    )

    bparser.add_args(
        "--axes",
        help="Axes to use for segmentation (default: YXC)",
        type=str,
        default="YXC",
    )

    bparser.add_args(
        "--n_tiles",
        help="Number of tiles to use for segmentation (default: 10)",
        type=int,
        default=10,
    )

    bparser.add_args(
        "--block_size",
        help="Block size to use for segmentation (default: 1024)",
        type=int,
        default=1024,
    )
    bparser.add_args(
        "--min_overlap",
        help="Minimum overlap to use for segmentation (default: 32)",
        type=int,
        default=32,
    )

    bparser.add_args(
        "--prob_thresh",
        help="Probability threshold to use for segmentation (default: 0.2)",
        type=float,
        default=0.2,
    )

    bparser.add_args(
        "--nms_thresh",
        help="Non-maximum suppression threshold to use for segmentation (default: 0.4)",
        type=float,
        default=0.4,
    )

    args = bparser.get_args()
    n_tiles = (args.n_tiles, args.n_tiles, 1)
    segment_nuclei(
        img_path=args.img,
        out=args.out,
        stain_type=args.stain_type,
        pretrained_model=args.pretrained_model,
        implementation=args.implementation,
        axes=args.axes,
        n_tiles=n_tiles,
        block_size=args.block_size,
        min_overlap=args.min_overlap,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
    )
