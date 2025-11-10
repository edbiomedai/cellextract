nextflow.enable.dsl = 2


// Parameters
// Input and output folders
params.input_dir = "data"
params.output_dir = "output"

// Output file names
params.output_file_prefix = "Sample"

// Segmentation parameters
params.nuc_size = 40 // expected diameter of nucleus, in pixels
params.cell_size = 60 // expected diameter of cell, in pixels
params.min_cell_size = 200 // minimum area in pixels, set to 0 to disable
params.nuc_model = "cyto" // Cellpose model for segmenting nuclei
params.cell_model = "cyto2" // Cellpose model for segmenting cell boundaries
params.dna_channel = 0 // Index of channel containing DNA stain. Often w1 = 0
params.membrane_channel = 3 // Index of channel containing membrane stain. Often w4 = 3

// Names of channels, usually W1-X or named channels (e.g. DNA, ER, ...)
params.channel_names = ["W1","W2","W3","W4","W5"]

// Names of masks after segmentation, usually Cells, Nuclei, Cytoplasm
params.mask_names = ["Cells","Nuclei","Cytoplasm"]

// Measurement options, usually do not need to be modified
params.texture_distance = 1 // while measuring texture, which distance to compute texture at (in pixels)
params.granularity_subsample_measurement = 0.25
params.granularity_subsample_background = 0.25
params.granularity_radius = 10
params.granularity_spectrum_length = 16
// End measurement options

// Batch size for collation of jobs on a cluster. Will run this many fields of view per job.
params.batch_size = 10

// Regular expression for decoding metadata from file names
params.metadata_regex = "(?P<Plate>[A-Za-z0-9\\-\\s]*)_(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])"

// Metadata column name containing treatments
params.treatment_key = "treatment"

// Regular expression used in postprocessing (which is not enabled by default)
params.sample_regex = "(?P<Replicate>R[0-9])-(?P<PlateLayout>P[0-9]{1,2})_(?P<CellLine>[A-za-z0-9\\-\\s]*)_(?P<TimePoint>[0-9\\-]*)_(?P<PlateID>[0-9]{5})_(?P<Well>[A-Z][0-9]{2})_(?P<Site>[0-9])"

// Input file name endings, ".tif" or ".tiff"
params.input_file_type = ".tif"

// End parameters


// Process definitions
include { CreateSampleSheet } from "./modules/process_create_sample_sheet.nf"
include { GetImageQCSheet } from "./modules/process_get_image_quality.nf"
include { CellposeSegmentationSheet } from "./modules/process_find_cellpose_masks.nf"
include { GetFeaturesSheet } from "./modules/process_get_feature_measurements.nf"
include { CollateMeasurementsBatched; CollateMeasurementsBatchedQC; CollateMeasurementsBatchedMultiNuc } from "./modules/process_collate_measurements.nf"
include { INPUT_CHECK } from "./subworkflows/local/input_check.nf"
include { StageData } from "./modules/process_stage_data.nf"

workflow {
    // Get absolute path to input directory
    def input_dir_abs = new File(params.input_dir).getCanonicalPath()

    // Detect all subfolders in input directory
    input_folders = channel
                    .fromPath( input_dir_abs + "/*", type: 'dir', maxDepth:1)

    // Stage data
    StageData(input_folders)
    input_dir_abs = StageData.out.collect()

    // Create sample sheet from folders
    sample_sheet = CreateSampleSheet(input_dir_abs)

    if (workflow.stubRun == false){
        // Check that sample sheet is valid
        sample_sheet = INPUT_CHECK(sample_sheet)

        // Extract data from sample sheet
        sample_sheet.multiMap{it ->
                    index: it.index
                    infiles: it.infiles
                    plate: it.plate
                    }
            .set{ meta }

        indeces = meta.index.collate(params.batch_size)
        input_files = meta.infiles.flatten().collate(params.batch_size * params.channel_names.size())
    } else {
        indeces = [0]
        input_files = Channel.fromPath("stub1.tif")
    }

    // Segmentation
    CellposeSegmentationSheet(CreateSampleSheet.out, indeces, input_files)

    // Measure image QC features
    GetImageQCSheet(CreateSampleSheet.out, CellposeSegmentationSheet.out.idx, CellposeSegmentationSheet.out.mask, CellposeSegmentationSheet.out.multi_mask)

    // Single-cell feature extraction
    GetFeaturesSheet(CreateSampleSheet.out, CellposeSegmentationSheet.out.idx, CellposeSegmentationSheet.out.mask, CellposeSegmentationSheet.out.multi_mask)

    // Collate all measurements from separate jobs
    CollateMeasurementsBatched(GetFeaturesSheet.out.collect())
    CollateMeasurementsBatchedQC(GetImageQCSheet.out.qc.collect())
    CollateMeasurementsBatchedMultiNuc(GetImageQCSheet.out.multi_nucleation.collect())

    // Unstage data to cleanup
    //UnstageData(StageData.out, CollateMeasurementsBatched.out, CollateMeasurementsBatchedQC.out, CollateMeasurementsBatchedMultiNuc.out)
}
