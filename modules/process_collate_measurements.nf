process CollateMeasurementsBatched {
    tag "AggregateFeatures"
    label "batchedCollate"
    publishDir "${params.output_dir}/morphology-measurements/", mode: 'link', pattern: "*_features.h5ad"

    input:
        path(features)

    output:
        file("${params.output_file_prefix}_features.h5ad")

    script:
    arr_features = features.collect({x -> x.join(' ')}).join(' ')
    """
    # activate singularity env if using
    if [ -f /home/mambauser/.bashrc ]; then
        echo "mamba user detected - assuming singularity usage"
        export MAMBA_SKIP_ACTIVATE=0
        source /home/mambauser/.bashrc
    fi

    arr_features=(${arr_features})
    python ${projectDir}/bin/collate_features.py \
        --files "\${arr_features[@]}" \
        --out ${params.output_file_prefix}_features.h5ad \
        --mode vertical
    """
    stub:
    arr_features = features.collect({x -> x.join(' ')}).join(' ')
    """
    arr_features=(${arr_features})
    echo "CollateMeasurementsBatched"
    echo "features: \${arr_features[@]}"
    touch ${params.output_file_prefix}_features.h5ad
    """

}


process CollateMeasurementsBatchedQC {
    tag "AggregateQC"
    label "batchedCollate"
    publishDir "${params.output_dir}/imageQC/", mode: 'link', pattern: "*_features_imageQC.h5ad"

    input:
        path(features)

    output:
        file("${params.output_file_prefix}_features_imageQC.h5ad")

    script:
    arr_features = features.collect({x -> x.join(' ')}).join(' ')
    """
    # activate singularity env if using
    if [ -f /home/mambauser/.bashrc ]; then
        echo "mamba user detected - assuming singularity usage"
        export MAMBA_SKIP_ACTIVATE=0
        source /home/mambauser/.bashrc
    fi

    arr_features=(${arr_features})
    python ${projectDir}/bin/collate_features.py \
        --files "\${arr_features[@]}" \
        --out ${params.output_file_prefix}_features_imageQC.h5ad \
        --mode vertical
    """
    stub:
    arr_features = features.collect({x -> x.join(' ')}).join(' ')
    """
    arr_features=(${arr_features})
    echo "CollateMeasurementsBatchedQC"
    echo "features: \${arr_features[@]}"
    touch ${params.output_file_prefix}_features_imageQC.h5ad
    """

}


process CollateMeasurementsBatchedMultiNuc {
    tag "AggregateMultiNuc"
    label "batchedCollate"
    publishDir "${params.output_dir}/multiNucleation/", mode: 'link', pattern: "*_features_multiNucleation.h5ad"

    input:
        path(features)

    output:
        file("${params.output_file_prefix}_features_multiNucleation.h5ad")

    script:
    arr_features = features.collect({x -> x.join(' ')}).join(' ')
    """
    # activate singularity env if using
    if [ -f /home/mambauser/.bashrc ]; then
        echo "mamba user detected - assuming singularity usage"
        export MAMBA_SKIP_ACTIVATE=0
        source /home/mambauser/.bashrc
    fi

    arr_features=(${arr_features})
    python ${projectDir}/bin/collate_features.py \
        --files "\${arr_features[@]}" \
        --out ${params.output_file_prefix}_features_multiNucleation.h5ad \
        --mode vertical
    """
    stub:
    arr_features = features.collect({x -> x.join(' ')}).join(' ')
    """
    arr_features=(${arr_features})
    echo "CollateMeasurementsBatchedMultiNuc"
    echo "features: \${arr_features[@]}"
    touch ${params.output_file_prefix}_features_multiNucleation.h5ad
    """

}

