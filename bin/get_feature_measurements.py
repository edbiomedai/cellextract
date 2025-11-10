#!/usr/bin/env python
# Import libraries
import os
import warnings

from background_correct import background_subtract
from features import measure_features
from images import Image, MultiMask
from parsers import BaseSuperParser, _add_granularity_args
from sample_sheet import parse_sample_sheet


def get_feature_measurements(
    mask_path,
    img_path,
    mask_names,
    channel_names,
    sample_name,
    distance,
    features,
    multi_mask=None,
    background_correct=True,
    background_correction_quantile=0.25,
):
    # Comma seperated lists may be passed when using batched computations
    if isinstance(mask_path, list) and "," in mask_path[0]:
        mask_path = mask_path[0].split(",")
    if multi_mask is not None:
        if isinstance(multi_mask, list) and "," in multi_mask[0]:
            multi_mask = multi_mask[0].split(",")
        if isinstance(features, list) and "," in multi_mask_features[0]:
            multi_mask_features = multi_mask_features[0].split(",")
    elif "multinucleated" in features:
        raise ValueError(
            "Multi-mask path required for computing features of multinucleated cells"
        )
    if isinstance(img_path, list) and "," in img_path[0]:
        img_path = img_path[0].split(",")

    # Load in image, mask, and multi-mask if needed
    img = Image(
        img_path=img_path,
        mask_path=mask_path,
        mask_names=mask_names,
        channel_names=channel_names,
    )

    mask = img.mask.get_mask() if img.mask is not None else None
    if multi_mask is not None:
        multi_mask = MultiMask(mask_path=multi_mask, mask_names=mask_names).get_mask()

    if background_correct:
        img.img = background_subtract(
            img.get_image(),
            mask,
            multi_mask,
            mask_index=0,  # cell mask
            background_correction_quantile=background_correction_quantile,
        )

    return measure_features(
        img=img.get_image(),
        mask=mask,
        mask_names=mask_names,
        multi_mask=multi_mask,
        channel_names=channel_names,
        sample_name=sample_name,
        distance=distance,
        features=features,
    )


def get_feature_measurement_sheet(
    sheet_path,
    idx,
    mask_names,
    channel_names,
    distance,
    features,
    background_correct=True,
    background_correction_quantile=0.25,
):
    # Check which features are selected
    mode = []
    image_features = {"image_quality", "count_objects", "multinucleated"}
    if any(f in features for f in image_features):
        mode += ["ImageQC"]
        qc_features = image_features.intersection(features)

    sc_features = {"intensity", "shape", "texture", "granularity"}
    if any(f in features for f in sc_features):
        mode += ["Features"]
        sc_features = sc_features.intersection(features)

    if not mode:
        raise ValueError("No features selected")

    for row in parse_sample_sheet(sheet_path, idx=idx):
        # Check that masks exist
        mask_path = row["Mask"][0]
        if not os.path.exists(mask_path):
            warnings.warn(f"{mask_path} does not exist. Skipping sample.")
            continue
        mask_multi_path = row["Mask"][1]
        if not os.path.exists(mask_multi_path):
            # For pipelines run without multi-mask detection,
            # skip multinucleated feature
            print(
                f"Will not compute multinucleated features for {mask_multi_path} "
                + "because output file is missing"
            )
            mask_multi_path = None
        # Per-image features
        if "ImageQC" in mode:
            res = get_feature_measurements(
                mask_path=row["Mask"][0],
                img_path=row["Channel"],
                mask_names=mask_names,
                channel_names=channel_names,
                sample_name=row["Sample"],
                distance=distance,
                features=qc_features,
                multi_mask=mask_multi_path,
            )
            adata = res["Image"]
            if adata is not None:
                adata.write(row["ImageQC"])

            multi = res["MultiNucleated"]
            if multi is not None:
                multi.write(row["MultiNuc"])

        # Single-cell features
        if "Features" in mode:
            res = get_feature_measurements(
                mask_path=row["Mask"][0],
                img_path=row["Channel"],
                mask_names=mask_names,
                channel_names=channel_names,
                sample_name=row["Sample"],
                distance=distance,
                features=sc_features,
                multi_mask=mask_multi_path,
                background_correct=background_correct,
                background_correction_quantile=background_correction_quantile,
            )
            adata = res["Object"]
            if adata is not None:
                adata.write(row["Features"])


if __name__ == "__main__":
    desc = (
        "Measure features of a label mask and image. "
        + "Currently supports intensity, shape and texture features."
    )
    bparser = BaseSuperParser(description=desc)

    bparser.add_args_to_both(
        "--multi_out",
        help="Output file for multi-mask features",
        required=False,
    )
    bparser.add_args_to_both(
        "--feature",
        nargs="+",
        help="Features to measure, separated by spaces. Must be in 'intensity', 'shape', 'texture', 'granularity', 'image_quality', 'count_objects'",
        required=False,
        default=["intensity", "shape", "texture", "granularity"],
    )
    bparser = _add_granularity_args(bparser)
    bparser.add_args_to_both(
        "--background_correct",
        action="store_true",
    )
    bparser.add_args_to_both(
        "--background_correction_quantile",
        type=float,
        default=0.25,
    )

    args = bparser.get_args()

    if args.mode == "single":
        res = get_feature_measurements(
            args.mask,
            args.img,
            args.mask_names,
            args.channel_names,
            args.sample,
            args.distance,
            args.feature,
            background_correct=args.background_correct,
            background_correction_quantile=args.background_correction_quantile,
        )
        if "multinucleated" in args.feature:
            adata, multi = res
        else:
            adata = res
            multi = None
        if adata is not None:
            adata.write(args.out)

    else:
        get_feature_measurement_sheet(
            args.sample_sheet,
            args.index,
            args.mask_names,
            args.channel_names,
            args.distance,
            args.feature,
            background_correct=args.background_correct,
            background_correction_quantile=args.background_correction_quantile,
        )
