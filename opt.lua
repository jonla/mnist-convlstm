opt = {
    backend = 'cudnn',
    inplace = true,

    batchSize = 5,
    rho = 15,

    trainingSamples = 5000,
    validationSamples = 1000,

    preTrained = false,
    nBatches = 200,
    sgdState = {learningRate = 0.01},

}
