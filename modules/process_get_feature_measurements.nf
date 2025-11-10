process GetFeaturesSheet {
    input:
        path sample_sheet
        val idx
        path mask_files
        path multi_mask_files

    output:
        path("*_features.h5ad")

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
        --mask_names ${params.mask_names.join(" ")} \
        --channel_names ${params.channel_names.join(" ")} \
        --distance ${params.texture_distance} \
        --subsample_size ${params.granularity_subsample_measurement} \
        --image_sample_size ${params.granularity_subsample_background} \
        --element_size ${params.granularity_radius} \
        --granular_spectrum_length ${params.granularity_spectrum_length} \
        --feature intensity shape texture granularity \
        --background_correct \
        --background_correction_quantile 0.25
    """
    stub:
    """
    echo "GetFeaturesSheet"
    echo "sample_sheet: ${sample_sheet}"
    echo "idx: ${idx.join(" ")}"
    touch ${params.output_file_prefix}_features.h5ad
    """
}
