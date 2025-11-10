process CreateSampleSheet {
    publishDir "${params.output_dir}", mode: 'copy'

    input:
        path staging_done

    output:
        path "sample_sheet.csv"

    script:
        input_dir = "${workflow.launchDir}/data/"
        """
        python3 ${projectDir}/bin/sample_sheet.py \\
            ${input_dir} \\
            --file_type "${params.input_file_type}" \\
            --output_file "sample_sheet.csv" \\
            --metadata_regex "${params.metadata_regex}"
        """
    stub:
        """
        echo "CreateSampleSheet"
        touch sample_sheet.csv
        """
}
