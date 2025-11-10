"""Feature classes to extract features from images and masks.
"""
import re
import warnings
from itertools import product
from typing import Dict, List, Union

import anndata
import numpy as np
import pandas as pd
from mahotas.features import haralick
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import find_boundaries


def intensity_std(mask, img):
    """Calculate the standard deviation of the intensity image within the mask.

    Args:
        mask (matrix): Binary mask of the region
        img (matrix): Intensity image

    Returns:
        float: Standard deviation of the intensity image within the mask
    """
    return np.std(img[mask == 1])


def intensity_median(mask, img):
    """Calculate the median of the intensity image within the mask.

    Args:
        mask (matrix): Binary mask of the region
        img (matrix): Intensity image

    Returns:
        float: Median of the intensity image within the mask
    """
    return np.median(img[mask == 1])


def get_border(mask):
    """Get the inner border of a mask. The inner border is defined as the set of pixels
    that are 1 in the mask, but have at least one 0 neighbour in 2-connectivity.

    Args:
        mask (matrix): Binary mask of the region of interest

    Returns:
        matrix: Binary mask of the inner border of the region of interest
    """
    border = find_boundaries(mask, connectivity=2, mode="inner")
    border[[0, -1], :] = mask[[0, -1], :]
    border[:, [0, -1]] = mask[:, [0, -1]]
    return border


def border_mean(mask, img):
    """Calculate the mean of the intensity image within the inner border of the mask.

    Args:
        mask (matrix): Binary mask of the region of interest
        img (matrix): Intensity image of the region of interest

    Returns:
        float: Mean of the intensity image within the inner border of the mask
    """
    border = get_border(mask)
    return np.mean(img[border])


def border_min(mask, img):
    """Calculate the minimum of the intensity image within the inner border of the mask.

    Args:
        mask (matrix): Binary mask of the region of interest
        img (matrix): Intensity image of the region of interest

    Returns:
        float: Minimum of the intensity image within the inner border of the mask
    """
    border = get_border(mask)
    return np.min(img[border])


def border_max(mask, img):
    """Calculate the maximum of the intensity image within the inner border of the mask.

    Args:
        mask (matrix): Binary mask of the region of interest
        img (matrix): Intensity image of the region of interest

    Returns:
        float: Maximum of the intensity image within the inner border of the mask
    """
    border = get_border(mask)
    return np.max(img[border])


def border_std(mask, img):
    """Calculate the standard deviation of the intensity image within the inner border
    of the mask.

    Args:
        mask (matrix): Binary mask of the region of interest
        img (matrix): Intensity image of the region of interest

    Returns:
        float: Standard deviation of the intensity image within the inner border of the mask
    """
    border = get_border(mask)
    return np.std(img[border])


def border_median(mask, img):
    """Calculate the median of the intensity image within the inner border of the mask.

    Args:
        mask (matrix): Binary mask of the region of interest
        img (matrix): Intensity image of the region of interest

    Returns:
        float: Median of the intensity image within the inner border of the mask
    """
    border = get_border(mask)
    return np.median(img[border])


class Feature:
    """The Feature class is used to manipulate a dataframe. It is
    initialized with two parameters: `mask_names` and `channel_names`, as well as an
    optional `delim` parameter. The class has two attributes, `mask_names` and
    `channel_names`, and one method, `features`.
    It is a superclass and not meant to be instantiated directly.


    Attributes
    -------
    features : Stores any feature measurements that are calculated.
    delim : A string that is used to delimit the module name from the column name. Defaults to "_".

    Constructor
    -------
    delim : str, optional
        A string that is used to delimit the module name from the column name. Defaults to "_".

    Methods
    -------
    `rename_cols()` : Rename columns in a feature dataframe.
    `insert_module_name()` : Insert the module name into the column names of a dataframe.
    """

    def __init__(self, delim="_"):
        super().__init__()
        self.delim = delim
        self.features = None

    def _split_feature_names(
        self, features: Union[pd.Series, List["str"]], feature_delim: str = "_"
    ) -> pd.DataFrame:
        """
        Split feature names into pd.DataFrame

        Parameters
        ----------
        features : pd.Series or list
                Feature names

        feature_delim : str
                Character delimiting feature names

        Returns
        ----------
        pd.DataFrame of feature names split into columns
        """

        features = pd.Series(features)

        df = features.str.split(feature_delim, expand=True)  # split feature names
        df.index = features
        df.columns = [f"feature_{str(i)}" for i in df.columns]
        return df

    def features_to_adata(
        self, df: pd.DataFrame, n_cols_meta: int = 1
    ) -> anndata.AnnData:
        df.reset_index(inplace=True, drop=True)
        df.index = df.index.astype(str)
        return anndata.AnnData(
            df.iloc[:, n_cols_meta:].values,
            obs=df.iloc[:, :n_cols_meta],
            dtype=np.float32,
            var=self._split_feature_names(df.columns[n_cols_meta:]),
        )

    def rename_cols(self, pattern, rep):
        """
        Rename columns of a feature dataframe in-place using regex

        Parameters
        ----------
        pattern : str
            Regex pattern to substitute
        rep : str
            Replacement string
        """
        self.features.rename(columns=lambda x: re.sub(pattern, rep, x), inplace=True)

    def _replace_channel_names(
        self, channel_names, delim, pattern="_(\\d)$", zero_based=True
    ):
        channels = [re.search(pattern, i) for i in self.features.columns]
        channels = [int(i.group(1)) if i is not None else None for i in channels]
        new_channels = [
            channel_names[i if zero_based else i - 1] if i is not None else ""
            for i in channels
        ]
        new_colnames = [
            re.sub(pattern, f"{delim}{j}", i) if j != "" else i
            for i, j in zip(self.features.columns, new_channels)
        ]
        self.features.columns = new_colnames

    def _replace_mask_names(self, mask_names, pattern="^\\d"):
        masks = [re.search("^(\\d)", i) for i in self.features.columns]
        masks = [int(i.group(1)) if i is not None else "" for i in masks]
        new_masks = [mask_names[i] if i != "" else "" for i in masks]
        new_colnames = [
            re.sub(pattern, f"{j}", i) if j != "" else i
            for i, j in zip(self.features.columns, new_masks)
        ]
        self.features.columns = new_colnames

    def insert_module_name(
        self, colnames: List[str], module: str, slot: int = 1
    ) -> pd.DataFrame:
        """
        Insert module name into column names of a dataframe

        Parameters
        ----------
        colnames : List[str]
            List of column names

        module : str
            Module name to insert

        slot : int, optional
            Slot to insert module name into, by default 1

        Returns
        ----------
        pd.DataFrame
            Dataframe with module name inserted into column names
        """

        new_colnames = colnames.tolist()
        for i, j in enumerate(colnames):
            if "ObjectNumber" in j:
                continue
            j = j.split(self.delim)
            j = j[:slot] + [module] + j[slot:]
            j = self.delim.join(j)
            new_colnames[i] = j

        return new_colnames

    def _move_metadata_columns(self):
        meta_cols = np.where(
            [re.search("ObjectNumber$|X$|Y$", i) for i in self.features.columns]
        )[0]
        meta = self.features.iloc[:, meta_cols]
        self.features.drop(meta.columns, axis=1, inplace=True)
        return pd.concat([meta, self.features], axis=1)


class Texture(Feature):
    """
    Compute texture features for each region in the mask.

    Attributes
    ----------
    features : Stores any feature measurements that are calculated.
    delim : A string that is used to delimit the module name from the column name. Defaults to "_".

    Constructor
    ----------
    img : np.ndarray
        Intensity image of the region of interest
    mask : np.ndarray
        A 2D or 3D matrix of integers, where each integer represents a
        different region. The matrix should be the same size as the `img`.
    distance : np.ndarray
        Distance between pixels to be used in the calculation of
        the texture measurements
    mask_names : str or list of str
        A list of strings containing the names of the masks, optional.

    Methods
    ----------
    `measure()` : Compute texture features for each region in the mask.
    """

    def measure(
        self, img, mask, channel_names, mask_names, distance, grey_levels: int = 16
    ):
        """
        Measure texture features for each region in the mask.

        Parameters
        ----------
        img : np.ndarray
            Intensity image of the region of interest
        mask : np.ndarray
            A 2D or 3D matrix of integers, where each integer represents a
            different region. The matrix should be the same size as the `img`.
        channel_names : str or list of str
            A list of strings containing the names of the channels, optional.
        mask_names : str or list of str
            A list of strings containing the names of the masks, optional.
        distance : int
            Distance between pixels to be used in the calculation of
            the texture measurements.
        grey_levels : int
            Number of grey levels to use in the calculation of the
            texture measurements.

        Returns
        ----------
        anndata.Anndata
            An AnnData object containing the texture features for each region in the mask.

        Note
        ----------
        If using a 3D mask matrix, this currently assumes that each mask contains
        identical object numbers. For example, this does not support multiple nuclei per cell.
        """
        img = np.floor(img * grey_levels).astype(int)
        self.features = self._measure_texture(img=img, mask=mask, distance=distance)
        self.features = self._rename_texture_features(channel_names, mask_names)
        return self.features_to_adata(self.features, 1)

    def _rename_texture_features(self, channel_names, mask_names):
        # Add channel names to column names
        self._replace_channel_names(channel_names, self.delim, zero_based=False)
        self._replace_mask_names(mask_names)

        # Rename metadata columns
        self.rename_cols("label", "ObjectNumber")
        self.rename_cols("mask", "Mask")
        self.features.columns = self.insert_module_name(
            self.features.columns, module="Texture"
        )
        return self.features

    def _measure_texture(self, mask, img, distance):
        """Calculate texture measurements for each region in the mask.

        Parameters
        ----------
        mask : np.ndarray
            A 2D or 3D matrix of integers, where each integer represents a
            different region. The matrix should be the same size as the `img`.
        img : np.ndarray
            A 3D matrix of floats, where the first two dimensions
            are the same size as the mask, and the third dimension is the number of channels.
        distance : int
            Distance between pixels to be used in the calculation of
            the texture measurements.

        Returns
        -------
        pd.DataFrame:
            A pandas DataFrame of measurements, where each row is a
            different region, and each column is a different measurement.

        Note
        ----------
        If using a 3D mask matrix, this currently assumes that each mask contains
        identical object numbers. For example, this does not support multiple nuclei per cell.
        """

        FEATURE_NAMES = [
            "AngularSecondMoment",
            "Contrast",
            "Correlation",
            "Variance",
            "InverseDifferenceMoment",
            "SumAverage",
            "SumVariance",
            "SumEntropy",
            "Entropy",
            "DifferenceVariance",
            "DifferenceEntropy",
            "InfoMeas1",
            "InfoMeas2",
        ]

        # Prepare output dataframe
        if mask.ndim < 3:
            ismask3d = False
            nrows = len(np.unique(mask)) - 1
        elif mask.ndim == 3:
            ismask3d = True
            nrows = sum(len(np.unique(mask[:, :, m])) - 1 for m in range(mask.shape[2]))
        else:
            raise ValueError("Mask must be 2D or 3D")

        ncols = 1 + 52 * img.shape[2]
        ncols += 1 if ismask3d else 0

        df = pd.DataFrame(index=range(nrows), columns=range(ncols))

        # If not using multiple masks, pad to 3D mask so we can use the same loop
        if not ismask3d:
            mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

        # Loop through each mask
        row_counter = 0
        for m in range(mask.shape[2]):
            cur_mask = mask[:, :, m]

            # Compute most features
            props = regionprops(cur_mask, img)

            # Loop over objects in mask
            for prop in props:
                region = prop["intensity_image"]
                d = {"label": prop["label"]}  # Built output dictionary
                if ismask3d:
                    d["mask"] = m
                # Loop over channels
                for stain in range(region.shape[2]):
                    # Compute haralick features
                    try:
                        haralick_features = haralick(
                            region[:, :, stain], distance=distance, ignore_zeros=True
                        )
                    except ValueError as error:
                        # Make an empty array of zeros
                        warnings.warn(
                            f"Haralick features could not be calculated. Returning zeros. Returned error was:\n {error}"
                        )
                        haralick_features = np.zeros((1, 13))
                    for i, feature in enumerate(FEATURE_NAMES):
                        # Compute rotation-invariant haralick features
                        d[f"Mean{feature}_{stain+1}"] = haralick_features[:, i].mean()
                        d[f"Std{feature}_{stain+1}"] = haralick_features[:, i].std()
                        d[f"Min{feature}_{stain+1}"] = haralick_features[:, i].min()
                        d[f"Max{feature}_{stain+1}"] = haralick_features[:, i].max()

                df.iloc[row_counter] = list(d.values())

                # Name output columns
                if row_counter == 0:
                    df.columns = d.keys()

                # Prepare for next object
                row_counter += 1

        df = df.convert_dtypes()

        if not ismask3d:
            self.features = df
            return self.features

        # Convert to wide format, where each mask has its own columns
        df_wide = df.pivot(index="label", columns="mask", values=df.columns[2:])

        # Enforce mask-ordered columns
        df_wide.columns.names = ["feature", "mask"]
        df_wide.sort_values(by=["mask", "feature"], axis=1, inplace=True)

        # Merge multi index labels into feature names
        cols = ["_".join([str(chan), feat]) for feat, chan in df_wide.columns.to_list()]
        df_wide.columns = cols
        self.features = df_wide.reset_index()
        return self.features


class Shape(Feature):
    """
    Compute shape features for each region in the mask.

    Attributes
    ----------
    features : Stores any feature measurements that are calculated.
    delim : A string that is used to delimit the module name from the column name. Defaults to "_".

    Constructor
    ----------
    mask : np.ndarray
        A 2D or 3D matrix of integers, where each integer represents a
        different region.
    mask_names : list of str
        A list of strings, where each string is the name of the mask.

    Methods
    ----------
    `measure` : Compute shape features for each region in the mask.
    """

    def measure(
        self, mask, mask_names, img=None, channel_names=None
    ):  # pylint: disable=unused-argument
        """
        Measure shape features for each region in the mask.

        Parameters
        ----------
        mask : np.ndarray
            A 2D or 3D matrix of integers, where each integer represents a
            different region. The matrix should be the same size as the `img`.
        mask_names : str or list of str
            A list of strings containing the names of the masks.
        img : np.ndarray
            Ignored, for compatibility with other feature classes.
        channel_names : list of str
            Ignored, for compatibility with other feature classes.

        Returns
        ----------
        np.ndarray
            A dataframe containing the shape features for each region in the mask.
        """
        self.features, colnames = self._measure_shape(mask)
        self.features = self._rename_shape_features(colnames, mask_names, self.delim)
        return self.features_to_adata(self.features, 3 * len(mask_names))

    def _shape_single_mask(self, mask):
        # Measure nuclei
        measurements = regionprops_table(
            mask,
            properties=[  # TODO check which features are rotational invariant
                "label",
                "centroid",
                "area",
                "convex_area",
                "eccentricity",
                "equivalent_diameter",
                "euler_number",
                "feret_diameter_max",
                "inertia_tensor",
                "inertia_tensor_eigvals",
                "major_axis_length",
                "minor_axis_length",
                "moments",
                "moments_central",
                "moments_hu",
                "moments_normalized",
                "orientation",
                "perimeter",
                "solidity",
            ],
        )

        # Convert measurements to dataframe
        return pd.DataFrame(measurements)

    def _measure_shape(self, mask):
        """Measure shape properties of the regions in a label mask.

        Returns
        -------
        pd.DataFrame:
            Measurements of the shape properties of the regions
        """
        if mask.ndim == 2:
            out = self._shape_single_mask(mask)
            return out, out.columns

        measurements = [
            self._shape_single_mask(mask[:, :, i]) for i in range(mask.shape[2])
        ]
        out = pd.concat(measurements, ignore_index=True, axis=1).reset_index(drop=True)
        return out, measurements[0].columns

    def _rename_shape_features(self, colnames, mask_names, delim="_"):
        new_colnames = list(product(mask_names, colnames))
        new_colnames = [f"{x[0]}{delim}{x[1]}" for x in new_colnames]
        self.features.columns = new_colnames

        self.rename_cols("centroid-0", "CentroidY")
        self.rename_cols("centroid-1", "CentroidX")
        self.rename_cols("label", "ObjectNumber")
        self.rename_cols("moments_central", "MomentsCentral")
        self.rename_cols("moments_normalized", "MomentsNormalized")
        self.rename_cols("moments_hu", "HuMoments")
        self.rename_cols("moments-", "Moments-")
        self.rename_cols("major_axis_length", "MajorAxisLength")
        self.rename_cols("minor_axis_length", "MinorAxisLength")
        self.rename_cols("inertia_tensor_eigvals", "InertiaTensorEigvals")
        self.rename_cols("inertia_tensor", "InertiaTensor")
        self.rename_cols("feret_diameter_max", "FeretDiameterMax")
        self.rename_cols("euler_number", "EulerNumber")
        self.rename_cols("equivalent_diameter", "EquivalentDiameter")
        self.rename_cols("eccentricity", "Eccentricity")
        self.rename_cols("convex_area", "ConvexArea")
        self.rename_cols("area", "Area")
        self.rename_cols("orientation", "Orientation")
        self.rename_cols("perimeter", "Perimeter")
        self.rename_cols("solidity", "Solidity")

        self.features.columns = self.insert_module_name(
            self.features.columns, module="AreaShape"
        )
        return self._move_metadata_columns()


class Intensity(Feature):
    """
    Compute intensity features for each region in the mask.

    Attributes
    ----------
    features : Stores any feature measurements that are calculated.
    delim : A string that is used to delimit the module name from the column name. Defaults to "_".

    Constructor
    ----------
    img : np.ndarray
        Intensity image of the region of interest
    mask : np.ndarray
        A 2D or 3D matrix of integers, where each integer represents a
        different region.
    channel_names : list of str
        A list of strings, where each string is the name of the channel.
    mask_names : list of str
        A list of strings, where each string is the name of the mask.

    Methods
    ----------
    `measure` : Compute intensity features for each region in the mask.
    """

    def measure(self, img, mask, channel_names, mask_names):
        self.features, colnames = self._measure_intensity(img=img, mask=mask)
        self.features = self._rename_intensity_features(
            channel_names, mask_names, colnames
        )
        return self.features_to_adata(self.features, len(mask_names))

    def _intensity_single_mask(self, mask, intensity_img):
        measurements = regionprops_table(
            mask,
            intensity_img,
            properties=["label", "intensity_mean", "intensity_min", "intensity_max"],
            extra_properties=[
                intensity_std,
                intensity_median,
                border_mean,
                border_min,
                border_max,
                border_std,
                border_median,
            ],
        )
        return pd.DataFrame(measurements)

    def _measure_intensity(self, mask, img):
        """Measure the intensity of the regions in the mask.

        Args:
            mask (matrix): Label mask(s) of the regions of interest
            img (matrix): Intensity image(s)

        Returns:
            DataFrame: Measurements of the intensity of the regions in the mask
        """

        if mask.ndim == 2:
            out = self._intensity_single_mask(mask, img)
            return out, out.columns

        measurements = [
            self._intensity_single_mask(mask[:, :, i], img)
            for i in range(mask.shape[2])
        ]
        out = pd.concat(measurements, ignore_index=True, axis=1).reset_index(drop=True)
        return out, measurements[0].columns

    def _rename_intensity_features(
        self,
        channel_names,
        mask_names,
        colnames,
    ):
        new_colnames = list(product(mask_names, colnames))
        new_colnames = [f"{x[0]}{self.delim}{x[1]}" for x in new_colnames]
        self.features.columns = new_colnames

        self._replace_channel_names(channel_names, self.delim, "-(\\d)$")
        self.rename_cols("label", "ObjectNumber")
        self.rename_cols("intensity_mean", "MeanIntensity")
        self.rename_cols("intensity_min", "MinIntensity")
        self.rename_cols("intensity_max", "MaxIntensity")
        self.rename_cols("intensity_std", "StdIntensity")
        self.rename_cols("intensity_median", "MedianIntensity")
        self.rename_cols("border_mean", "MeanIntensityEdge")
        self.rename_cols("border_min", "MinIntensityEdge")
        self.rename_cols("border_max", "MaxIntensityEdge")
        self.rename_cols("border_std", "StdIntensityEdge")
        self.rename_cols("border_median", "EdgeMedian")

        self.features.columns = self.insert_module_name(
            self.features.columns, module="Intensity"
        )
        return self._move_metadata_columns()


class Granularity(Feature):
    """
    Compute granularity features for each region in the mask.

    Attributes
    ----------
    features : Stores any feature measurements that are calculated.
    delim : A string that is used to delimit the module name from the column name. Defaults to "_".

    Constructor
    ----------
    img : np.ndarray
        Intensity image of the region of interest
    mask : np.ndarray
        A 2D or 3D matrix of integers, where each integer represents a
        different region.
    channel_names : list of str
        A list of strings, where each string is the name of the channel.
    mask_names : list of str
        A list of strings, where each string is the name of the mask.

    Methods
    ----------
    `measure` : Compute granularity features for each region in the mask.

    """

    def __init__(self) -> None:
        super().__init__()

    def measure(
        self,
        img,
        mask,
        channel_names,
        mask_names,
        subsample_size=0.25,
        image_sample_size=0.25,
        element_size=10,
        granular_spectrum_length=16,
    ):
        self.features, colnames = self._measure_granularity(
            img=img,
            mask=mask,
            channel_names=channel_names,
            mask_names=mask_names,
            subsample_size=subsample_size,
            image_sample_size=image_sample_size,
            element_size=element_size,
            granular_spectrum_length=granular_spectrum_length,
        )
        self.features = self._rename_granularity_features(
            channel_names, mask_names, colnames
        )
        return self.features_to_adata(self.features, 1)

    def _measure_granularity(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        channel_names: List[str],
        mask_names: List[str],
        subsample_size=0.25,
        image_sample_size=0.25,
        element_size=10,
        granular_spectrum_length=16,
    ) -> pd.DataFrame:
        """
        Measure granularity features for a single mask.

        Args:
            img (np.ndarray): Intensity image(s)
            mask (np.ndarray): Label mask(s) of the regions of interest
            channel_names (list of str): A list of strings, where each string is the name of the channel.
            mask_names (list of str): A list of strings, where each string is the name of the mask.
        """
        from cellprofiler_interface import measure_granularity

        if mask.ndim == 2:
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

        out = measure_granularity(
            img=img,
            mask=mask,
            channel_names=channel_names,
            mask_names=mask_names,
            subsample_size=subsample_size,
            image_sample_size=image_sample_size,
            element_size=element_size,
            granular_spectrum_length=granular_spectrum_length,
        )
        return out, out.columns

    def _rename_granularity_features(
        self,
        channel_names,
        mask_names,
        colnames,
    ):
        """Takes parameters only for compatibility with other feature modules."""
        self.rename_cols("Granularity_", "")

        self.features.columns = self.insert_module_name(
            self.features.columns, module="Granularity"
        )
        return self._move_metadata_columns()


class ImageQuality(Feature):
    def __init__(self) -> None:
        super().__init__()

    def measure(self, img: np.ndarray, channel_names: List[str]) -> pd.DataFrame:
        """Measure image quality features.

        Args:
            img (np.ndarray): Intensity image(s)
            channel_names (list of str): A list of strings, where each string is the name of the channel.

        Returns:
            DataFrame: Measurements of the image quality features
        """
        from cellprofiler_interface import measure_image_quality

        self.features = measure_image_quality(img, channel_name=channel_names)
        return self.features


class CountObjects(Feature):
    """Count number of objects per mask"""

    def __init__(self) -> None:
        super().__init__()

    def measure(self, mask: np.ndarray, mask_names: List[str]) -> pd.DataFrame:
        """Measure the number of objects in each mask.

        Args:
            mask (np.ndarray): Label mask(s) of the regions of interest
            mask_names (list of str): A list of strings, where each string is the name of the mask.

        Returns:
            DataFrame: Measurements of the number of objects in each mask
        """
        assert mask.ndim == 3, "Mask must be 3D"
        assert (
            len(mask_names) == mask.shape[2]
        ), "Number of mask names must match number of masks"

        n_objects = [np.unique(mask[:, :, i]).size - 1 for i in range(mask.shape[2])]
        colnames = [f"{x}{self.delim}Count" for x in mask_names]
        self.features = pd.DataFrame(n_objects, index=colnames).T
        return self.features


class SingleCellMultiNucleated(Feature):
    """Compute per-cell features for multi-nucleated (and clumped) cells"""

    def __init__(self) -> None:
        super().__init__()

    def _crop_to_coords(
        self,
        img,
        coords,
    ):
        """Crop image using coordinates returned by `regionprops`"""
        rows = coords[:, 0]
        cols = coords[:, 1]
        if img.ndim == 3:
            return img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1, :]
        return img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    def _loop_over_masks(self, mask1, mask2):
        """Loop over objects in mask 1, returning the labels and cropped masks"""
        for prop in regionprops(mask1):
            m1crop = self._crop_to_coords(mask1, prop["coords"]).copy()
            m2crop = self._crop_to_coords(mask2, prop["coords"]).copy()
            label = prop["label"]

            # Remove all other pixels not in super-mask
            idx = m1crop != label
            np.putmask(m1crop, idx, 0)
            np.putmask(m2crop, idx, 0)
            yield label, m1crop, m2crop

    def _count_subobject(self, mask, multi_mask, mask_names) -> pd.DataFrame:
        """Count number of subobject per object in first mask.
        Used for counting nuclei per cell"""
        counts = {}
        for label, _, m2 in self._loop_over_masks(mask, multi_mask):
            c = np.unique(m2)
            c = c[c != 0].size
            assert c != 1, f"Label {label} has exactly 1 subobject"

            # Cells in supermask that do not have subobjects will show as 0
            # To compute downstream statistics, we still want to count them,
            # but only as containing a single subobject.
            c = 1 if c == 0 else c
            counts[label] = c

        feat_counts = pd.DataFrame.from_dict(counts, orient="index", columns=["Count"])
        counts_name = f"{mask_names[0]}{self.delim}MultiNucleated{self.delim}{mask_names[1]}Per{mask_names[0]}"
        feat_counts.columns = [counts_name]
        feat_counts.index.name = f"{mask_names[0]}{self.delim}ObjectNumber"
        return feat_counts

    def _shapes(self, mask: np.ndarray, mask_names: List[str]) -> pd.DataFrame:
        """Summarise shape measurements"""
        shape_module = Shape()
        shape_module.measure(mask, mask_names=mask_names)
        return shape_module.features

    def measure(
        self, mask: np.ndarray, multi_mask: np.ndarray, mask_names: List[str]
    ) -> anndata.AnnData:
        """Measure the number of nuclei per cell and associated features
        Note that this assumes that the first mask is the cell mask, and the second mask is the nuclei mask.
        The cytoplasm mask is ignored, if present.

        Args:
            mask (np.ndarray): Label mask(s) of the regions of interest. Must be 2D.
            multi_mask (np.ndarray): Label mask(s) of multinucleated cells. Must be 2D.
            mask_names (list of str): A list of strings, where each string is the name of the mask.

        Returns:
            AnnData: Measurements of the number of nuclei in each cell, and associated features
        """
        assert mask.ndim == 2, "Mask must be 2D"
        assert mask.shape == multi_mask.shape, "Mask and multi-mask must be same shape"
        counts = self._count_subobject(mask, multi_mask, mask_names).reset_index(
            drop=True
        )
        shapes = self._shapes(mask, mask_names=[mask_names[0]]).reset_index(drop=True)
        self.features = pd.concat([counts, shapes], axis=1)
        self.features = self._move_metadata_columns()
        adata = self.features_to_adata(self.features, 4)
        return adata


class MultiNucleated(Feature):
    """Compute per-image features for multi-nucleated (and clumped) cells"""

    def __init__(self) -> None:
        super().__init__()

    def measure(self, mask: np.ndarray, mask_names: List[str]) -> pd.DataFrame:
        """Measure the number of objects in each mask.
        Note that this assumes that the first mask is the cell mask, and the second mask is the nuclei mask.
        The cytoplasm mask is ignored, if present.

        Args:
            mask (np.ndarray): Label mask(s) of the regions of interest
            mask_names (list of str): A list of strings, where each string is the name of the mask.

        Returns:
            DataFrame: Measurements of the number of objects in each mask
        """
        assert mask.ndim == 3, "Mask must be 3D"
        assert mask.shape[2] >= 2, "At least two masks must be provided"

        feat_freq = self._frequencies(mask, mask_names)
        feat_shape = self._shapes(mask, mask_names)
        feat = pd.concat([feat_freq, feat_shape], axis=1)
        self.features = feat
        return feat

    def _frequencies(self, mask: np.ndarray, mask_names: List[str]) -> pd.DataFrame:
        """Compute the average number of nuclei per cell"""
        n_cells = sum(np.unique(mask[:, :, 0]) != 0)
        n_nuclei = sum(np.unique(mask[:, :, 1]) != 0)
        if n_nuclei > 0 and n_cells > 0:
            avg_nuc_per_cells = n_nuclei / n_cells
        else:
            avg_nuc_per_cells = 0
        feat = pd.DataFrame(
            [n_cells, n_nuclei, avg_nuc_per_cells],
            index=[
                f"{mask_names[0]}{self.delim}MultiNucleated_Count",
                f"{mask_names[1]}{self.delim}MultiNucleated_Count",
                f"{mask_names[0]}{self.delim}MultiNucleated{self.delim}{mask_names[1]}PerCellAvg",
            ],
        ).T
        return feat

    def _shapes(self, mask: np.ndarray, mask_names: List[str]) -> pd.DataFrame:
        """Summarise shape measurements"""
        shape_module = Shape()
        feat_shape = shape_module.measure(mask, mask_names=mask_names)
        shape_df = feat_shape.to_df()
        summaries = (
            shape_df.describe()
            .loc[["50%", "std", "min", "max"]]
            .set_index(pd.Index(["median", "std", "min", "max"]))
            .reset_index()
            .melt(id_vars="index")
        )
        summaries["index"] = summaries["index"].str.capitalize()
        summaries["feature"] = summaries["variable"] + "_" + summaries["index"]
        summaries = (
            summaries.drop(columns=["index", "variable"])
            .set_index("feature")
            .T.reset_index(drop=True)
        )
        summaries.columns.name = None
        return summaries


def measure_features(
    img,
    mask,
    channel_names,
    mask_names,
    multi_mask=None,
    sample_name="Sample",
    features=None,
    distance=1,
    subsample_size=0.25,
    image_sample_size=0.25,
    element_size=10,
    granular_spectrum_length=16,
) -> Dict:
    """Measure features of the regions in the mask.

    Args:
        img (matrix): Intensity image(s)
        mask (matrix): Label mask(s) of regions of interest
        channel_names (list of str): A list of strings, where each string is the name of the channel.
        mask_names (list of str): A list of strings, where each string is the name of the mask.
        multi_mask (matrix): Label mask(s) of multinucleated cells, used only when "features" includes "multinucleated".
            Defaults to None.
        sample_name (str): Name of sample. Defaults to "Sample".
        features (str or list of str): A list of features to measure. Defaults to all features.
            Options: "intensity", "shape", "texture", "granularity", "image_quality", "count_objects", "multinucleated"
        distance (int): Distance to use for calculating texture features. Defaults to 1.
        subsample_size (float): Granularity - Subsampling factor for granularity measurements. Defaults to 0.25.
        image_sample_size (float): Granularity - Subsampling factor for background reduction. Defaults to 0.25.
        element_size (int): Granularity - Radius of structuring element. Defaults to 10.
        granular_spectrum_length (int): Granularity - Range of the granular spectrum. Defaults to 16.
    Returns:
        AnnData: Measurements of the regions in the mask
    Note:
        Image quality and object counts will be placed in `.obs`
    """
    # set default features
    feature_list = []
    valid_features = [
        "intensity",
        "shape",
        "texture",
        "granularity",
        "image_quality",
        "count_objects",
        "multinucleated",
    ]
    if features is None or not features:
        features = valid_features
    elif isinstance(features, str):
        features = [features]

    # Sanity check
    invalid_features = set(features) - set(valid_features)
    assert len(invalid_features) == 0, f"Invalid features selected: {invalid_features}"

    # Default names
    if channel_names is None and img is not None:
        channel_names = [f"Channel_{i+1}" for i in range(img.shape[2])]
    if mask_names is None and mask is not None:
        mask_names = [f"Mask_{i+1}" for i in range(mask.shape[2])]

    output_features = dict.fromkeys(["Image", "Object", "MultiNucleated"])

    # go through features and compute, if necessary
    if "intensity" in features:
        mod_intensity = Intensity()
        feature_list.append(
            mod_intensity.measure(
                img=img, mask=mask, channel_names=channel_names, mask_names=mask_names
            )
        )

    if "shape" in features:
        mod_shape = Shape()
        feature_list.append(mod_shape.measure(mask=mask, mask_names=mask_names))

    if "texture" in features:
        mod_texture = Texture()
        features_texture = mod_texture.measure(
            img=img,
            mask=mask,
            channel_names=channel_names,
            mask_names=mask_names,
            distance=distance,
        )
        feature_list.append(features_texture)

    if "granularity" in features:
        mod_granularity = Granularity()
        features_granularity = mod_granularity.measure(
            img=img,
            mask=mask,
            channel_names=channel_names,
            mask_names=mask_names,
            subsample_size=subsample_size,
            image_sample_size=image_sample_size,
            element_size=element_size,
            granular_spectrum_length=granular_spectrum_length,
        )
        feature_list.append(features_granularity)

    if "image_quality" in features:
        mod_qc = ImageQuality()
        features_image_quality = mod_qc.measure(img=img, channel_names=channel_names)

    if "count_objects" in features:
        mod_count = CountObjects()
        features_count = mod_count.measure(mask=mask, mask_names=mask_names)

    if "multinucleated" in features:
        mod_multi = SingleCellMultiNucleated()
        features_multi = mod_multi.measure(
            mask=multi_mask[:, :, 0],
            multi_mask=multi_mask[:, :, 1],
            mask_names=mask_names,
        )
        features_multi.obs.insert(0, "SampleName", sample_name)
        output_features["MultiNucleated"] = features_multi

    # Collect all features
    if feature_list:
        # per-object features
        features_ad = anndata.concat(feature_list, axis=1, merge="unique", join="outer")
        features_ad.obs.insert(0, "SampleName", sample_name)
        features_ad.var = Feature()._split_feature_names(features_ad.var.index)
        output_features["Object"] = features_ad
    else:
        # only image-based features were requested
        features_ad = anndata.AnnData()
        output_features["Object"] = None

    # intialize per-image features
    meta_df = pd.DataFrame()
    nrows = max(1, features_ad.shape[0])

    # add per-image features
    if "image_quality" in features:
        meta_df = pd.concat([features_image_quality] * nrows)

    if "count_objects" in features:
        if meta_df.empty:
            meta_df = pd.concat([features_count] * nrows)
        else:
            meta_df = pd.concat([features_count, meta_df], axis=1)

    # decide if only image-level features were requested
    # if so, return those as AnnData.
    # else, add per-image features into obs slot of per-object features
    if not meta_df.empty:
        if features_ad.shape[0] == 0:
            meta_df.reset_index(drop=True, inplace=True)  # reset index
            meta_df.index = meta_df.index.astype(
                str
            )  # convert index to str to suppress AnnData warnings
            features_ad = anndata.AnnData(
                meta_df.to_numpy(), dtype=np.float32, var=meta_df.columns.to_frame()
            )
            features_ad.obs["SampleName"] = sample_name
            features_ad.var = Feature()._split_feature_names(features_ad.var.index)
            output_features["Image"] = features_ad
        else:
            # Mixed per-object and per-image requested, merge
            features_ad.obs = pd.concat([features_ad.obs, meta_df], axis=1)
            output_features["Object"] = features_ad
    return output_features
