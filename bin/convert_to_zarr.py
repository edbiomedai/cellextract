import argparse
from pathlib import Path

import distributed
from faim_ipa.hcs.acquisition import TileAlignmentOptions
from faim_ipa.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_ipa.hcs.imagexpress import SinglePlaneAcquisition
from faim_ipa.hcs.plate import PlateLayout
from faim_ipa.stitching import stitching_utils


def _spawn_client(threads_per_worker=2, processes=False, n_workers=1):
    return distributed.Client(
        threads_per_worker=threads_per_worker, processes=processes, n_workers=n_workers
    )


def convert_plate_to_zarr(plate_directory, out_dir, client=None):
    client = client or _spawn_client()
    # in nextflow we get the timepoint directory, but we need the plate directory
    # move one level up
    plate_directory = Path(plate_directory).resolve().parent
    print(f"Converting plate {plate_directory} to Zarr")
    plate_name = plate_directory.stem
    print(f"Plate name: {plate_name}")
    plate_acquisition = SinglePlaneAcquisition(
        acquisition_dir=plate_directory,
        alignment=TileAlignmentOptions.GRID,
    )

    converter = ConvertToNGFFPlate(
        ngff_plate=NGFFPlate(
            root_dir=out_dir,
            name=plate_name,
            layout=PlateLayout.I384,
            order_name="order",
            barcode="barcode",
        ),
        yx_binning=1,
        warp_func=stitching_utils.translate_tiles_2d,
        fuse_func=stitching_utils.fuse_mean,
        client=client,
    )

    plate = converter.create_zarr_plate(plate_acquisition)

    converter.run(
        plate=plate,
        plate_acquisition=plate_acquisition,
        well_sub_group="0",
        chunks=(540, 540),
        max_layer=3,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert plate to Zarr")
    parser.add_argument(
        "--plate_directory",
        type=str,
        help="Path to directory containing plate acquisition",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to save Zarr output",
        required=True,
    )
    args = parser.parse_args()

    convert_plate_to_zarr(args.plate_directory, args.out_dir)
