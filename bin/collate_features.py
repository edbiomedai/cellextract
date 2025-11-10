#!/usr/bin/env python
from argparse import ArgumentParser

import anndata as ad


def make_parser():
    parser = ArgumentParser(description="Merge h5ad files into one")
    parser.add_argument("--files", nargs="+", help="Input h5ad files")
    parser.add_argument("--out", type=str, help="Output h5ad file")
    parser.add_argument(
        "--mode",
        type=str,
        help="Merge mode [horizontal or vertical]",
        default="horizontal",
    )
    return parser


def main(files, out, mode):
    adata_list = [ad.read_h5ad(f) for f in files]
    adata_list = [a for a in adata_list if a.shape[0] > 0]
    # Make sure sample names are strings
    # Otherwise, AnnData > 0.9 will fail to concat
    for adata in adata_list:
        adata.obs["SampleName"] = adata.obs["SampleName"].astype(str)
    axis = 0 if mode == "vertical" else 1
    adata_coll = ad.concat(adata_list, axis=axis, join="inner", merge="same")
    adata_coll.obs_names_make_unique()
    adata_coll.write(out)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args.files, args.out, args.mode)
