require 'optim'


cost = cost or {}
validationCost = validationCost or {}
local batchNumber
local trainingCost

-- High level training function, executing 'opt.nBatches' mini batches
-- Function also validates one batch per training batch
-- (can be commented out)
function train()
    trainingCost = {}
    batchNumber = 0
    model:training()
    for i=1,opt.nBatches do
        local inputs, labels = trainSet:getMinibatch()
        trainBatch(inputs, labels)
        local inputs, labels = valSet:getMinibatch()
        validateBatch(inputs, labels)
    end
    model:clearState()
    collectgarbage()
    
    -- Calculate avergae cost and store
    local s = 0
    for i, val in pairs(trainingCost) do
        s = s + val
    end
end

local timer = torch.Timer()
local dataTimer = torch.Timer()
parameters, gradParameters = model:getParameters()


-- Function for training a single mini batch
function trainBatch(inputs, labels)
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

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

function validateBatch(inputs, labels)
    collectgarbage()
    local outputs = model:forward(inputs)
    local err = criterion:forward(outputs, labels)
    table.insert(validationCost,err)
    collectgarbage()
end

-- Plots training and validation cost
function plotCost(avgWidth)
    local avgWidth = avgWidth or 50
    if not gnuplot then
        require 'gnuplot'
    end

    local function avgCost(costT)
        local nAvg = (#cost - #cost%avgWidth)/avgWidth
        local costAvg = torch.Tensor(nAvg)
        local costAvgX = torch.range(1, nAvg):mul(avgWidth)

        for i = 1,nAvg do
            costAvg[i] = costT[{{(i-1)*avgWidth+1, i*avgWidth}}]:mean()
        end
        return costAvgX, costAvg
    end
    local costT = torch.Tensor(cost)
    local costX = torch.range(1, #cost)
    local costAvgX, costAvg = avgCost(costT)

    local costTV = torch.Tensor(validationCost)
    local costXV = torch.range(1, #validationCost)
    local costAvgXV, costAvgV = avgCost(costTV)

    gnuplot.plot({'Training batch',costX, costT},
                 {'Training,  ' .. avgWidth .. '-avg', costAvgX, costAvg},
                 {'Validation batch',costXV, costTV},
                 {'Validation,  ' .. avgWidth .. '-avg', costAvgXV, costAvgV})
end



