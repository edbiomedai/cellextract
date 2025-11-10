process GetImageQCSheet {
    input:
        path sample_sheet
        val idx
        path mask_files
        path multi_mask_files

    output:
        path("*_imageQC.h5ad"), emit: qc
        path("*_multiNucleation.h5ad"), emit: multi_nucleation

    script:
    """
    # activate singularity env if using
    if [ -f /home/mambauser/.bashrc ]; then
        echo "mamba user detected - assuming singularity usage"
        export MAMBA_SKIP_ACTIVATE=0
        source /home/mambauser/.bashrc
    fi
    python ${projectDir}/bin/get_feature_measurements.py multi \
        --sample_sheet ${sample_sheet} \
        --index ${idx.join(" ")} \
        --channel_names ${params.channel_names.join(" ")} \
        --mask_names ${params.mask_names.join(" ")} \
        --feature image_quality count_objects multinucleated \
        --background_correct \
        --background_correction_quantile 0.25
    """

    stub:
    """
    echo "GetImageQCSheet"
    echo "sample_sheet: ${sample_sheet}"
    echo "idx: ${idx.join(" ")}"
    touch ${params.output_file_prefix}_imageQC.h5ad
    touch ${params.output_file_prefix}_multiNucleation.h5ad
    """
}
