Nextflow Pipeline to Extract Cells From BioImages
================

This is the repository for pipelines to extract morphological
information from various types of bioimages made by the Khamseh and
Beentjes labs from the University of Edinburgh.

This repository contains two workflows, <IMAGE_WORKFLOW_NAME> for
extracting morphological information from traditional microscopy images
and `workflow_wsi.nf` for extracting morphological information from
whole slide images. Both of these workflows run a backbone of the same
processes that can also be found in this repository, all of which can be
customised by the user.

## Running the Workflows

TODO: Write instructions.

As we have 2 workflows in one project this will probably have to be git
cloning instructions unless we do something clever.

## Workflow Overviews

### <IMAGE_WORKFLOW_NAME>

### `workflow_wsi.nf`

This workflow is for processing whole slide images, which typically tend
to be bigger than normal images and also contain more structural
information. To this end, images within this pipeline are by default
passed to algorithms that handle WSI’s in a more memory efficient
manner.

## Contact

TODO: Decide on contact strategy (i.e. Issues vs. emails etc.)
