require 'optim'


cost = cost or {}
local batchNumber
local trainingCost

-- High level training function, executing 'opt.nBatches' mini batches
function train()
    trainingCost = {}
    batchNumber = 0
    model:training()
    for i=1,opt.nBatches do
        local inputs, labels = getMinibatch()
        trainBatch(inputs, labels)
    end
    model:clearState()
    collectgarbage()
    
    -- Calculate avergae cost and store
    local s = 0
    for i, val in pairs(trainingCost) do
        s = s + val
    end
    --table.insert(overallCost.training, s/#trainingCost)
end

local timer = torch.Timer()
local dataTimer = torch.Timer()
parameters, gradParameters = model:getParameters()


-- Function for training a single mini batch
function trainBatch(inputs, labels)
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    local trainMask
    local err, outputs

    feval = function(x)
        model:zeroGradParameters()
        outputs = model:forward(inputs)

        err = criterion:forward(outputs, labels)
        local gradOutputs = criterion:backward(outputs, labels)
        model:backward(inputs, gradOutputs)
        return err, gradParameters
    end
    optim.sgd(feval, parameters, opt.sgdState)
    batchNumber = batchNumber + 1
    table.insert(cost,err)
    table.insert(trainingCost, err)
    print(('Minibatch: [%d/%d]\t Time %.4f Cost %.4f, DataLoadingTime %.3f'):format(
        batchNumber, opt.nBatches, timer:time().real, err, dataLoadingTime))

    dataTimer:reset()
    collectgarbage()
end

function plotCost(avgWidth)
    if not gnuplot then
        require 'gnuplot'
    end

    local avgWidth = avgWidth or 50
    local costT = torch.Tensor(cost)
    local costX = torch.range(1, #cost)
    local nAvg = (#cost - #cost%avgWidth)/avgWidth
    local costAvg = torch.Tensor(nAvg)
    local costAvgX = torch.range(1, nAvg):mul(avgWidth)

    for i = 1,nAvg do
        costAvg[i] = costT[{{(i-1)*avgWidth+1, i*avgWidth}}]:mean()
    end
    --plots = {costT, costAvg}
    gnuplot.plot({'Mini batch cost',costX, costT},
                {'Mean over ' .. avgWidth .. ' batches', costAvgX, costAvg})
end



