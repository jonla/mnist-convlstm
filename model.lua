require 'ConvLSTM'

local backend
if opt.backend == 'cudnn' then
  require 'cudnn'
  require 'cunn'
  backend = cudnn
else
  backend = nn
end

-- initialization from MSR
-- (This function is not used currently)
-- remove it?
local function MSRinit(net)
    local function init(name)
        for k,v in pairs(net:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            if v.bias then v.bias:zero() end
        end
    end
    -- have to do for both backends
    init'cudnn.SpatialConvolution'
    init'nn.SpatialConvolution'
end 

-- Network architecture
net = nn.Sequential()

encoder = nn.Sequential()
encoder:add(backend.SpatialConvolution(1,32,3,3,1,1,1,1))
encoder:add(backend.SpatialBatchNormalization(32))
encoder:add(backend.ReLU(opt.inplace))
encoder:add(backend.SpatialMaxPooling(2,2,2,2))

encoder:add(backend.SpatialConvolution(32,64,3,3,1,1,1,1))
encoder:add(backend.SpatialBatchNormalization(64))
encoder:add(backend.ReLU(opt.inplace))
encoder:add(backend.SpatialMaxPooling(2,2,2,2))
net:add(encoder)

net:add(nn.ConvLSTM(64,512,opt.rho,3,3,1))
--net:add(nn.ConvLSTM(1,10,opt.rho,7,7,1))

decoder = nn.Sequential()
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(backend.SpatialConvolution(512,256,3,3,1,1,1,1))
decoder:add(backend.SpatialBatchNormalization(256))
decoder:add(backend.ReLU(opt.inplace))
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(backend.SpatialConvolution(256,1,3,3,1,1,1,1))
net:add(decoder)

net:add(backend.Sigmoid())
--MSRinit(net)

-- Make network recurrent
model = nn.Sequencer(net)
model:remember('neither')
model:training()

-- The binomial cross-entropy cost function
criterion = nn.SequencerCriterion(nn.BCECriterion())

-- Put model on GPU if using cudnn
if opt.backend == 'cudnn' then
    model:cuda()
    criterion:cuda()
end
