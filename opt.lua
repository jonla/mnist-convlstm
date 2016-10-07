opt = {
    backend = 'cudnn',
    inplace = true,

    batchSize = 20,
    rho = 15,

    trainingSamples = 1000,

    nBatches = 200,
    sgdState = {learningRate = 0.01},

}
