cellextract: image-to-feature nextflow pipeline
================

This pipeline processes microscopy images to single-cell feature matrices.
It has found application in [our work](https://doi.org/10.1101/2025.11.14.687149) on phenotypes induced by microRNAs 
and was developed primarily for high-content (Cell Painting) data in mind.

**Features**
- Segmentation with Cellpose-2S, which detects single- and multi-nucleated cells
- Image QC feature extraction (thanks to hooks into CellProfiler)
- CellProfiler-like feature extraction

## Running the Workflows
1. Adapt the `example_nextflow.config` to your needs (see [below](#config))
2. Run the pipeline `nexflow run` (possibly with `-profile` and `-c` options to point to your nextflow profiles and configs)

For example, you may wish to use singularity/docker so you do not have to manage dependencies yourself. To do so, make sure you have singularity installed, then use `nextflow run -c singularity`.

## Config
Steps to adapt the config:
1. Copy the config to one that nextflow will detect automatically: `cp example_nextflow.config nextflow.config`
2. Edit it using your favorite text editor, e.g. `nano nextflow.config`

