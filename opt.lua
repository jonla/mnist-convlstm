-- Table with options for running scripts
opt = {
    backend = 'cudnn', -- 'cudnn' or 'nn'
    inplace = true, -- For network ReLUs

    batchSize = 2, -- can be larger with backend 'nn' or large gpu
    rho = 20, -- sequence length, has same effect on memory as batch size

    -- number of samples to include in each dataset:
    trainingSamples = 5000,
    validationSamples = 1000,

    preTrained = true, -- 'true' loads 'model.t7'
    nBatches = 500, -- #mini batches per each 'train()' command
    sgdState = {learningRate = 0.01},

}
