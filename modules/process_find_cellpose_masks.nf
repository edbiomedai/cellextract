process CellposeSegmentationSheet {
    publishDir("${params.output_dir}/mask", mode: 'link', pattern: "*_mask.npz")
    publishDir("${params.output_dir}/mask", mode: 'link', pattern: "*_mask_multi.npz")

    input:
        path sample_sheet
        val idx
        path input_files

    output:
        path("*_mask.npz"), emit: mask
        path("*_mask_multi.npz"), emit: multi_mask
        val idx, emit: idx

    script:
        """
        # activate singularity env if using
        if [ -f /home/mambauser/.bashrc ]; then
            echo "mamba user detected - assuming singularity usage"
            export MAMBA_SKIP_ACTIVATE=0
            source /home/mambauser/.bashrc
        fi

        python ${projectDir}/bin/segment_cellpose.py multi \
        --sample_sheet "${sample_sheet}" \
        --index ${idx.join(" ")} \
        --nuc_channel ${params.dna_channel} \
        --cell_channel ${params.membrane_channel} \
        --nuc_model ${params.nuc_model} \
        --cell_model ${params.cell_model} \
        --nuc_diameter ${params.nuc_size} \
        --cell_diameter ${params.cell_size} \
        --min_cell_size ${params.min_cell_size}
        """

    stub:
        """
        echo "GetFeaturesSheet"
        echo "sample_sheet: ${sample_sheet}"
        echo "idx: ${idx.join(" ")}"
        touch stub_mask.npz
        touch stub_mask_multi.npz
        """
}
