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

net = nn.Sequential()

encoder = nn.Sequential()
encoder:add(backend.SpatialConvolution(1,16,3,3,1,1,1,1))
encoder:add(backend.SpatialBatchNormalization(16))
encoder:add(backend.ReLU(opt.inplace))
encoder:add(backend.SpatialMaxPooling(2,2,2,2))

encoder:add(backend.SpatialConvolution(16,32,3,3,1,1,1,1))
encoder:add(backend.SpatialBatchNormalization(32))
encoder:add(backend.ReLU(opt.inplace))
encoder:add(backend.SpatialMaxPooling(2,2,2,2))
net:add(encoder)

net:add(nn.ConvLSTM(32,32,opt.rho,3,3,1))
--net:add(nn.ConvLSTM(1,10,opt.rho,7,7,1))

decoder = nn.Sequential()
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(backend.SpatialConvolution(32,16,3,3,1,1,1,1))
decoder:add(backend.SpatialBatchNormalization(16))
decoder:add(backend.ReLU(opt.inplace))
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(backend.SpatialConvolution(16,1,3,3,1,1,1,1))
net:add(decoder)

net:add(backend.Sigmoid())
--MSRinit(net)

model = nn.Sequencer(net)
model:remember('both')
model:training()

criterion = nn.SequencerCriterion(nn.BCECriterion())

if opt.backend == 'cudnn' then
    model:cuda()
    criterion:cuda()
end
