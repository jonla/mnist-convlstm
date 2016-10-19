-- Use floats, not doubles
-- setlocale to avoid decimal to be ","
torch.setdefaulttensortype('torch.FloatTensor')
os.setlocale('en_US.UTF-8')

-- Packages
require 'sys'
require 'image'

dofile 'opt.lua' -- loads options for running in table 'opt'
dofile 'model.lua' -- declares the model 'model'
dofile 'data.lua' -- datasets, and 'test(index,maxTime)

if opt.preTrained then 
    model = torch.load('model.t7')
    if opt.backend == 'cudnn' then model:cuda() end
end

-- Simple test of dataloading and veryfying
-- network input->output
input, label = trainSet:getMinibatch()
output = model:forward(input)

dofile 'train.lua' -- loads train(), plotCost()

-- TRAIN, TEST, SAVE or whatever below here --
--train()
