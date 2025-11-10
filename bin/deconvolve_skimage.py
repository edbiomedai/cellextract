#!/usr/bin/env python
from numpy import savez_compressed
from stain_deconvolvers import SKImageDeconvolver

from images import TissueImage
from parsers import BaseSingleParser

if __name__ == "__main__":
    desc = "Deconvolve HED image to RGB"
    reqs = ["img_path"]
    bparser = BaseSingleParser(description=desc, reqs=reqs)

    bparser.add_args(
        "--out", help="Path to save image, should end in `.npz`", required=True
    )

    bparser.add_args(
        "--stain_type",
        help="The type of staining used in the image, should be `H&E` or `H-DAB` (default: H&E)",
        type=str,
        default="H&E",
    )

    bparser.add_args(
        "--num_tiles",
        help="Number of tiles to use for deconvolution (default: 1)",
        type=int,
        default=1,
    )

    args = bparser.get_args()

    img = TissueImage(args.img, stain_type=args.stain_type)

    skid = SKImageDeconvolver()

    dec_img = skid.deconvolve(
        img=img.img, num_tiles=args.num_tiles, img_type=args.stain_type
    )

    savez_compressed(args.out, dec_img)
