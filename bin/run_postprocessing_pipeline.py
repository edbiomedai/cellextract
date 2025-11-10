import argparse
import os
from glob import glob

import scmorph as sm
from natsort import natsorted

from single_cell_distances import (
    preprocess,
    qc_adata,
    run_kruskal_filter,
    unsupervised_ecdf_distances,
)


def main(
    adata_file,
    qc_file,
    platemap_file,
    metadata_regex,
    neg_control="control_transfection_reagent_only",
    batch_key="PlateID",
    replicate_key="Replicate",
    treatment_key="Treatment",
    qc_threshold=0.15,
):
    sample_name = os.path.basename(adata_file).split("_")[0]
    print("Preprocessing")

    do_batch_correct = batch_key is not None

    existing_output = natsorted(glob(f"./{sample_name}*_features_qc.h5ad"))

    if not existing_output:
        sc = preprocess(
            adata_file,
            platemap_file=platemap_file,
            metadata_regex=metadata_regex,
            do_batch_correct=do_batch_correct,
            neg_control=neg_control,
            batch_key=batch_key,
            replicate_key=replicate_key,
        )

        for i, adata in enumerate(sc):
            prefix = "" if len(sc) == 1 else f"_R{i + 1}"
            print(f"Processing {sample_name}{prefix}")

            if qc_file is not None and os.path.exists(qc_file):
                print("QC...")
                # Perform QC
                sc[i] = qc_adata(
                    adata=adata,
                    qc_file=qc_file,
                    sample_name=sample_name,
                    metadata_regex=metadata_regex,
                    out=".",
                    prefix=prefix,
                    threshold=qc_threshold,
                )
                print("feature filt")
                sc[i] = run_kruskal_filter(
                    adata=sc[i], out=".", sample_name=sample_name, prefix=prefix
                )
            else:
                raise IOError("No QC file provided or found")
    else:
        print("QC already performed, will load in the filtered adata")
        sc = [sm.read(f) for f in existing_output]

    for i, adata in enumerate(sc):
        prefix = "" if len(sc) == 1 else f"_R{i + 1}"
        # Single-cell ECDFs
        print(f"Hit calling {sample_name}{prefix}")
        unsupervised_ecdf_distances(
            adata,
            out=".",
            sample_name=sample_name,
            prefix=prefix,
            group_key=None,
            keep_all_background=False,
            batch_key=batch_key,
            treatment_key=treatment_key,
            neg_control=neg_control,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis pipeline")
    parser.add_argument("--features", help="Path to single-cell features")
    parser.add_argument("--image_features", help="Path to image QC featur file")
    parser.add_argument("--platemap", help="Path to platemap file")
    parser.add_argument(
        "--neg_control",
        help="Name of negative control",
        default="control_transfection_reagent_only",
    )
    parser.add_argument(
        "--batch_key", help="Key for batch correction", default="PlateID"
    )
    parser.add_argument(
        "--treatment_key",
        help="Key for treatment correction",
        default="Treatment",
    )
    parser.add_argument(
        "--replicate_key",
        help="Key for replicate correction. Leave empty if not using replicate experiments.",
        default=None,
    )
    parser.add_argument(
        "--qc_threshold",
        help="Distance threshold for unsupervised QC filtering",
        default=0.15,
    )
    parser.add_argument(
        "--metadata_regex",
        type=str,
        help="Regex used to extract metadata from the file names. Default is '(?P<CellLine>[A-za-z0-9\\-\\s]*)_(?P<TimePoint>[0-9\\-]*)_(?P<PlateLayout>[0-9]{5})_(?P<Well>[A-Z][0-9]{2})_(?P<Site>[0-9])'",
        default=r"(?P<Replicate>R[0-9])-(?P<PlateLayout>[CP][0-9]{1,2})_(?P<CellLine>[A-za-z0-9\-\s]*)_(?P<TimePoint>[0-9\\-]*)_(?P<PlateID>[0-9]{5})_(?P<Well>[A-Z][0-9]{2})_(?P<Site>[0-9])",
        required=False,
    )
    args = parser.parse_args()
    main(
        adata_file=args.features,
        qc_file=args.image_features,
        platemap_file=args.platemap,
        metadata_regex=args.metadata_regex,
        neg_control=args.neg_control,
        batch_key=args.batch_key,
        replicate_key=args.replicate_key,
        treatment_key=args.treatment_key,
    )
