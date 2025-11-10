process StageData {
    publishDir "${workflow.launchDir}/data/"
    memory { 4.GB * task.attempt }
    input:
        path input_folder

    output:
        path out_dir

    script:
    out_dir = "${input_folder.getName()}"
    """
    # Stage data by converting symlink to data in two steps.
    mkdir -p staged
    ls -lah .
    rsync -rtL -s --perms --chmod=u+rwx --ignore-existing --exclude "*thumb*" "./${input_folder}" "./staged/"
    unlink "${input_folder}"
    mv -f "staged/${input_folder}" "${input_folder}"
    rm -r staged
    """
    stub:
    out_dir = "${input_folder.getName()}"
    """
    echo "StageData"
    echo "input_folder: ${input_folder}"
    touch "stub1.tif" "stub2.tif"
    """
}
/*
process UnstageData {
    input:
        path input_folder
        value collateMeasurementsBatched_done
        value collateMeasurementsBatchedQC_done
        value collateMeasurementsBatchedMultiNuc_done
    script:
    """
    # Follow symlink and delete files in target directory
    # First from work directory
    find ${input_folder} -type l -exec rm -r {}/ \;

    # Then cleanup leftover symlinks in publishDir
    cd ${workflow.launchDir}
    find "data/" -type l -exec rm -r {}/ \;
    """
    stub:
    """
    echo "UnstageData"
    """
}
*/