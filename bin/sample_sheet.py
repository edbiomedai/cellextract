import argparse
import glob
import os
import re
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def create_sample_sheet_from_dir(
    input_dir: str = ".",
    file_type: str = ".tif",
    metadata_regex=r"(?P<Plate>[A-Za-z0-9\-\s]*)_(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])",
    channel_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create a sample sheet from a directory of images

    Args:
    input_dir (str): The directory containing the images
    file_type (str): The file type of the images. Default is ".tif"
    metadata_regex (str): The regex used to extract metadata from the file names.
        Default is "(?P<Plate>[A-Za-z0-9\-\s]*)_(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])"
    channel_names (List[str]): The names of the channels. Default is None

    Returns:
    pd.DataFrame: A sample sheet with one row per image set
    """
    cwd = os.getcwd()
    # Change directory so we can derive relative paths
    # and guess directory-depth for metadata extraction
    os.chdir(input_dir)

    # Recurse over directories, find all matching files,
    # removing thumbnails
    files = glob.glob("**/*" + file_type, recursive=True)
    files = [f for f in files if not re.search("thumb", f)]
    files.sort()

    # Divide files into file sets
    sample_sheet = hts_sample_sheet(
        files, metadata_regex=metadata_regex, channel_names=channel_names
    )
    sample_sheet = hts_create_output_names(
        sample_sheet=sample_sheet,
    )
    # Change back to original directory
    os.chdir(cwd)

    return sample_sheet


def hts_sample_sheet(files, metadata_regex, channel_names) -> pd.DataFrame:
    # Get metadata
    meta = hts_meta_from_regex(files, metadata_regex)

    # Create sample sheet
    sample_sheet = pd.DataFrame(meta)
    sample_sheet["file"] = [os.path.abspath(file) for file in files]

    # Create within-plate IDs
    ids = sample_sheet[["Well", "Site"]].apply(lambda x: "_".join(x), axis=1)

    # Add additional cross-plate ID information
    extra_id_cols = [
        i
        for i in ["PlateName", "Plate", "TimePoint", "Group", "PlateID"]
        if i in sample_sheet.columns
    ]
    ids = sample_sheet[extra_id_cols].apply(lambda x: "_".join(x), axis=1) + "_" + ids

    # insert IDs into sample sheet
    sample_sheet.insert(0, "Sample", ids)
    sample_sheet.sort_values("Sample", inplace=True)

    # Pivot to one row per image set
    index = sample_sheet.columns.drop(["file", "ChannelNumber"])
    sample_sheet = sample_sheet.pivot(
        index=index, columns="ChannelNumber", values="file"
    )
    if channel_names is None:
        sample_sheet.columns = ["Channel_" + col for col in sample_sheet.columns]
    else:
        sample_sheet.columns = channel_names

    sample_sheet.reset_index(inplace=True)
    missing = sample_sheet.isna().any(axis=1)
    if missing.any():
        warnings.warn(
            "Some samples had missing files and will be ignored:\n"
            + "\n".join(sample_sheet[missing]["Sample"])
        )
        sample_sheet.drop(sample_sheet[missing].index, inplace=True)
    return sample_sheet


def hts_meta_from_regex(files, regex) -> Dict:
    # Remove folder names
    files_base = [os.path.basename(file) for file in files]

    # Extract metadata from file names
    regex = re.compile(regex)
    matches = [re.search(regex, file) for file in files_base]

    # Place metadata in dictionary
    meta_cols = matches[0].groupdict().keys()
    meta_cols = {
        meta_col: [match.group(meta_col) for match in matches] for meta_col in meta_cols
    }

    # Extract metadata from folder names
    folders = [os.path.dirname(file) for file in files]
    if set(folders) == {""}:  # No folders
        return meta_cols

    folders_s = [folder.split(os.sep) for folder in folders]

    # Sanity check
    folder_s_len = {len(folder_s) for folder_s in folders_s}
    if len(folder_s_len) > 1:
        raise ValueError("Not all folders have the same depth. Check input directory.")

    # Guess type of folder structure
    folder_s_len = int(folder_s_len.pop())
    # folder_s_len == 0 will never be reached because of earlier return statement
    if folder_s_len == 1:
        keys = ["PlateID"]
    elif folder_s_len == 2:
        keys = ["Group", "PlateID"]
    elif folder_s_len == 3:
        keys = ["PlateName", "TimePoint", "PlateID"]
    elif folder_s_len == 4:
        keys = ["PlateName", "Date", "PlateID", "TimePoint"]
    else:
        raise ValueError(
            "Folder structure other than 'PlateName/Date/PlateID/Timepoint',"
            + "'PlateName/TimePoint/PlateID', 'PlateID/', and 'Group/PlateID/'"
            + "not currently implemented."
        )

    meta_folders = dict(zip(keys, zip(*folders_s)))
    # Combine metadata
    return {**meta_folders, **meta_cols}


def hts_create_output_names(sample_sheet) -> pd.DataFrame:
    """Insert columns for output file names for HTS experiments"""
    sample_sheet = create_output_names(sample_sheet)
    sample = sample_sheet["Sample"]
    mask_index = sample_sheet.columns.get_loc("out_mask")
    sample_sheet.insert(mask_index + 1, "out_mask_multi", sample + "_mask_multi.npz")

    return sample_sheet


def create_output_names(sample_sheet) -> pd.DataFrame:
    """Insert columns for output file names"""
    sample = sample_sheet["Sample"].replace(r"\s", "-", regex=True)
    sample_sheet["out_mask"] = sample + "_mask.npz"
    sample_sheet["out_features"] = sample + "_features.h5ad"
    sample_sheet["out_imageQC"] = sample + "_imageQC.h5ad"
    sample_sheet["out_multiNuc"] = sample + "_multiNucleation.h5ad"
    return sample_sheet


def parse_sample_sheet(
    path: Optional[str] = None,
    sheet: Optional[pd.DataFrame] = None,
    idx: Optional[Union[int, List[int]]] = None,
) -> pd.DataFrame:
    if path is None and sheet is None:
        raise ValueError("Either path or sheet must be provided")
    if path is not None and sheet is not None:
        raise ValueError("Only one of path or sheet must be provided")
    if path is not None:
        sample_sheet = pd.read_csv(path, index_col=False)
    else:
        sample_sheet = sheet
    if idx is None:
        idx = np.arange(len(sample_sheet))
    else:
        idx = [idx] if isinstance(idx, int) else idx
    sample_sheet = sample_sheet.iloc[idx, :]
    for row in sample_sheet.to_dict(orient="records"):
        yield categorise_sample_row(row)


def categorise_sample_row(row):
    keys = row.keys()
    channel_keys = [key for key in keys if re.search("^Channel", key)]
    channel_files = [row[key] for key in channel_keys] if channel_keys else None

    mask_keys = [key for key in keys if re.search("^out_mask", key)]
    mask_files = [row[key] for key in mask_keys] if mask_keys else None

    feature_keys = [key for key in keys if re.search("^out_features", key)]
    feature_files = (
        " ".join([row[key] for key in feature_keys]) if feature_keys else None
    )

    out = {
        "Sample": row["Sample"],
        "Channel": channel_files,
        "Mask": mask_files,
        "ImageQC": row["out_imageQC"],
        "MultiNuc": row["out_multiNuc"],
        "Features": feature_files,
    }

    return out


def main(
    input_dir: str = ".",
    file_type: str = ".tiff",
    metadata_regex="(?P<Plate>[A-Za-z0-9\-\s]*)_(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])",
    channel_names: Optional[List[str]] = None,
    output_file: str = "sample_sheet.csv",
) -> None:
    sample_sheet = create_sample_sheet_from_dir(
        input_dir,
        file_type,
        metadata_regex,
        channel_names,
    )
    sample_sheet.reset_index(inplace=True)
    sample_sheet.rename(columns={"index": "Index"}, inplace=True)
    assert all(sample_sheet["Index"] == sample_sheet.index), "Index column not correct"
    sample_sheet.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a sample sheet from a directory of images"
    )
    parser.add_argument("input_dir", type=str, help="Directory containing the images")
    parser.add_argument(
        "--file_type",
        type=str,
        help="File type of the images. Default is '.tif'",
        default=".tif",
        required=False,
    )
    parser.add_argument(
        "--metadata_regex",
        type=str,
        help="Regex used to extract metadata from the file names. Default is '(?P<Plate>[A-Za-z0-9\-\s]*)_(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])'",
        default=r"(?P<Plate>[A-Za-z0-9\-\s]*)_(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])",
        required=False,
    )
    parser.add_argument(
        "--channel_names", type=str, help="Channel names", required=False
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file name. Default is 'sample_sheet.csv'",
        default="sample_sheet.csv",
        required=False,
    )
    args = parser.parse_args()
    main(
        args.input_dir,
        args.file_type,
        args.metadata_regex,
        args.channel_names,
        args.output_file,
    )
