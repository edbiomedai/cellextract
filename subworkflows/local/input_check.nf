//
// Check input samplesheet and get read channels
//
// Modeled after nf-core/rnaseq
workflow INPUT_CHECK {
    take:
    samplesheet // file: /path/to/samplesheet.csv

    main:
    samplesheet
        .splitCsv ( header:true, sep:',' )
        .map { create_image_channel(it) }
        .set { image_sets }
    
    emit:
    image_sets
    
}

def create_image_channel(row) {
    // create meta map
    def meta = [:]
    meta.index = row.Index
    meta.sample = row.Sample
    meta.infiles = new ArrayList<String>()
    meta.maskfiles = new ArrayList<String>()
    meta.features = row.out_features
    meta.imageqc = row.out_imageQC
    //FileSearch fileSearch = new FileSearch()
    for (entry : row.entrySet()) {
        if (entry.getKey().startsWith("Channel_")) {
            file = entry.getValue()
            //file = fileSearch.findAbsolutePathWithDir(new File("${params.input_dir}"), file)
            meta.infiles.add(file)
        }
        if (entry.getKey().startsWith("out_mask_")) {
            meta.maskfiles.add(entry.getValue())
        }
    }
    return meta
}
