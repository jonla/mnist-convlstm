torch.setdefaulttensortype('torch.FloatTensor')
os.setlocale('en_US.UTF-8')
dofile 'opt.lua'
require 'sys'
require 'image'
dofile 'model.lua'
dofile 'data.lua'

model = torch.load('model.t7')
if opt.backend == 'cudnn' then model:cuda() end

input, label = getMinibatch()
output = model:forward(input)

dofile 'train.lua'
--train()
