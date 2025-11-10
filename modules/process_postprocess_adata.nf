process PostprocessAdata {
    conda "${projectDir}/envs/postprocessing.yaml"
    time "24 h"
    memory "100 GB"
    publishDir "${params.output_dir}/morphology-measurements/", mode: 'link', pattern: "*_features.h5ad"

    input:
        path(image_qc_features)
        path(single_cell_features)

    output:
        file("${params.output_file_prefix}_features_processed.h5ad")

    script:
        """
        python ${projectDir}/bin/run_postprocessing_pipeline.py \
        . ${projectDir} ${params.platemap} ${params.sample_regex}
        """

}