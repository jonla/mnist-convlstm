require 'dataset-mnist'



trainData, trainLabels = mnist.loadTrainSet(opt.trainingSamples, opt.rho)
testData, testLabels = mnist.loadTestSet(10, 100)


function getMinibatch()
    local h = trainData[1][1]:size(3)
    local w = trainData[1][1]:size(4)
    local indecies = torch.Tensor(opt.batchSize)
    indecies:random(1,trainData:size())
    local label = torch.Tensor(opt.batchSize, 1, h, w):cuda()
    for i=1,opt.batchSize do
        label[i] = trainLabels[indecies[i]]:cuda()
    end

    local inputBatch = {}
    local labelBatch = {}
    for i=1,opt.rho do
        local frames = torch.Tensor(opt.batchSize, 1, h, w):cuda()
        for j=1,opt.batchSize do
            frames[{{j}}] = trainData[indecies[j]][i]:cuda()
        end
        table.insert(inputBatch, frames)
        table.insert(labelBatch, label)
    end
    return inputBatch, labelBatch
end

function test(index, maxTime)
    local index = index or 1
    local sequence = testData[index]
    local input = {}
    for i=1,maxTime or #testData[1] do
        table.insert(input, sequence[i]:cuda())
    end
    model:evaluate()
    output = model:forward(input)
    model:training()

    for i,frame in pairs(output) do
        output[i] = torch.cat({sequence[i], frame:float(), testLabels[{{index}}]}, 4)
        --table.insert(input, testData[index][i]:cuda())
    end

    simulateSequence(output)
end


function simulateSequence(sequence)
    local zoom = 8
    local h = sequence[1]:size(3)*zoom
    local w = sequence[1]:size(4)*zoom
    local qtwidget = require 'qtwidget'
    win = qtwidget.newwindow(w,h)
    for i, frame in pairs(sequence) do
        image.display{image = frame, win=win, zoom=zoom}
        sys.sleep(0.75)
    end
end
