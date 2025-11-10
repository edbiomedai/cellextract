#!/usr/bin/env python
from numpy import savez_compressed
from stain_normalisers import MinMaxNormaliser

from images import TissueImage
from parsers import BaseSingleParser

if __name__ == "__main__":
    desc = "Normalise staining in image"
    reqs = ["img_path"]
    bparser = BaseSingleParser(description=desc, reqs=reqs)

    bparser.add_args(
        "--out",
        help="Path to save normalised image, should end in `.npz`",
        required=True,
    )

    bparser.add_args("--stain_type", help="Type of stain", required=True)

    args = bparser.get_args()

    img = TissueImage(args.img, args.stain_type)
    min_max_normaliser = MinMaxNormaliser()
    nimg = img.normalise(min_max_normaliser)
    savez_compressed(args.out, nimg)
