#!/usr/bin/env python
"""Functions to interact with CellProfiler.

These are semi-functional solutions to overcome the heavy dependency
on a GUI for CellProfiler. Instead, we establish minimal functions that create
CellProfiler objects and compute features.
View these functions as fit-for-purpose, not as a general solution.
"""
from typing import List, Optional, Union

import cellprofiler_core
import cellprofiler_core.preferences
import numpy as np
import pandas as pd

cellprofiler_core.preferences.set_headless()


def _make_channel_names(ndim):
    if ndim == 2:
        channel_name = "channel"
    else:
        channel_name = [f"W{i}" for i in range(ndim)]
    return channel_name


def _make_mask_names(ndim):
    if ndim == 2:
        mask_name = "mask"
    else:
        mask_name = [f"Mask{i}" for i in range(ndim)]
    return mask_name


def _module_image_quality(channel_name):
    import cellprofiler.modules.measureimagequality
    from cellprofiler.modules.measureimagequality import (
        MeasureImageQuality as measurement_module,
    )

    module = measurement_module()
    module.images_choice.value = cellprofiler.modules.measureimagequality.O_SELECT

    for i, _ in enumerate(channel_name):
        if i > 0:
            module.add_image_group()
        module.image_groups[i].include_image_scalings.value = False
        module.image_groups[i].image_names.value = channel_name[i]
        module.image_groups[i].use_all_threshold_methods.value = False

    module.set_module_num(1)
    return module


def _module_granularity(
    subsample_size=0.25,
    image_sample_size=0.25,
    element_size=10,
    granular_spectrum_length=16,
):
    import cellprofiler.modules.measuregranularity
    from cellprofiler.modules.measuregranularity import (
        MeasureGranularity as measurement_module,
    )

    module = measurement_module()
    module.subsample_size.value = subsample_size
    module.image_sample_size.value = image_sample_size
    module.element_size.value = element_size
    module.granular_spectrum_length.value = granular_spectrum_length
    return module


# TODO: Add support for list of file sets
def make_img_set(
    pixel_data: np.ndarray,
    masks: Optional[np.ndarray] = None,
    channel_name: Optional[str] = "channel",
):
    import cellprofiler_core.image

    if pixel_data.ndim == 2:
        pixel_data = pixel_data.reshape((*pixel_data.shape, 1))

    # Prepare images and objects
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)

    # TODO: implement all masks
    # At the moment we only retain the first (typically cell) mask
    if masks is None or masks.ndim == 2:
        mask = masks
    elif masks.ndim == 3:
        mask = masks[:, :, 0]
    else:
        raise ValueError("Mask must be 2D or 3D")

    for i, ch in enumerate(channel_name):
        image = cellprofiler_core.image.Image(
            pixel_data[:, :, i], dimensions=2, mask=mask
        )
        image_set.add(ch, image)

    return image_set_list


def make_module(module="MeasureImageQuality", channel_name="channel", **kwargs):
    supported_modules = ["MeasureImageQuality", "MeasureGranularity"]
    if module not in supported_modules:
        raise NotImplementedError(
            f"Module {module} is not supported. Supported modules are {', '.join(supported_modules)}"
        )

    if module == "MeasureImageQuality":
        return _module_image_quality(channel_name)

    return _module_granularity(**kwargs)


def make_pipeline(module):
    import cellprofiler_core.pipeline

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)

    return pipeline


def make_workspace(
    pipeline,
    img_set_list,
    mask=None,
    mask_names=None,
):
    import cellprofiler_core.measurement
    import cellprofiler_core.modules
    import cellprofiler_core.object

    module = pipeline.modules()[0]
    object_set = cellprofiler_core.object.ObjectSet()
    img_set = img_set_list.get_image_set(0)

    # Add masks if needed
    if mask is not None:
        # Support 2D masks
        if mask.ndim == 2:
            mask = mask.reshape((*mask.shape, 1))
        for i in range(mask.shape[2]):
            obj = cellprofiler_core.object.Objects()
            obj.segmented = mask[:, :, i]
            object_set.add_objects(obj, mask_names[i])
            module.objects_list.value.append(mask_names[i])

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        img_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        img_set_list,
    )
    return workspace


def measure_image_quality(
    img: np.ndarray, channel_name: Optional[Union[str, List[str]]] = None
):
    if channel_name is None:
        channel_name = _make_channel_names(img.shape[2])

    module = make_module("MeasureImageQuality", channel_name)
    pipeline = make_pipeline(module)
    img_set = make_img_set(img, channel_name=channel_name)
    wspace = make_workspace(pipeline, img_set)

    module = wspace.module
    module.run(wspace)

    measurements = wspace.get_measurements()
    features = measurements.get_measurement_columns()
    features = [f[1] for f in features]
    fmeasurements = [measurements.get_current_measurement("Image", f) for f in features]
    return pd.DataFrame(fmeasurements, index=features).T


def measure_granularity(
    img: np.ndarray,
    mask: np.ndarray,
    channel_names: Optional[Union[str, List[str]]] = None,
    mask_names: Optional[Union[str, List[str]]] = None,
    subsample_size=0.25,
    image_sample_size=0.25,
    element_size=10,
    granular_spectrum_length=16,
):
    from natsort import natsorted

    if channel_names is None:
        channel_names = _make_channel_names(img.shape[2])

    if mask_names is None:
        mask_names = _make_mask_names(mask.shape[2])

    module = make_module(
        "MeasureGranularity",
        channel_names,
        subsample_size=subsample_size,
        image_sample_size=image_sample_size,
        element_size=element_size,
        granular_spectrum_length=granular_spectrum_length,
    )
    pipeline = make_pipeline(module)
    img_set = make_img_set(img, masks=mask, channel_name=channel_names)
    wspace = make_workspace(pipeline, img_set, mask=mask, mask_names=mask_names)

    module = wspace.module
    for ch in channel_names:
        module.run_on_image_setting(wspace, ch)

    measurements = wspace.get_measurements()
    features = measurements.get_measurement_columns()

    out = []
    for i in range(len(mask_names)):
        cur_features = [f[1] for f in features if f[0] == mask_names[i]]
        cur_measurements = [
            measurements.get_current_measurement(mask_names[i], f) for f in cur_features
        ]
        cur_df = (
            pd.DataFrame(cur_measurements, index=cur_features)
            .dropna(axis=1, how="all")
            .T
        )
        cur_df = cur_df[natsorted(cur_df.columns)]
        cur_df.columns = [f"{mask_names[i]}_{c}" for c in cur_df.columns]
        out.append(cur_df)
    outdf = (
        pd.concat(out, axis=1).reset_index().rename(columns={"index": "ObjectNumber"})
    )
    return outdf
