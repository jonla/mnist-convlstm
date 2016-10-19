require 'dataset-mnist'

-- Create datasets and store in RAM
trainSet = mnist.loadTrainSet(opt.trainingSamples, opt.rho)
valSet = mnist.loadTestSet(opt.validationSamples, opt.rho)
testSet = mnist.loadTestSet(10, 100)

-- Function to test inference on one of the test sequences
function test(index, maxTime)
    local index = index or 1
    local sequence = testSet.data[index]
    local input = {}
    for i=1,maxTime or #testData[1] do
        table.insert(input, sequence[i]:cuda())
    end
    model:evaluate()
    output = model:forward(input)
    model:training()

    for i,frame in pairs(output) do
        output[i] = torch.cat({sequence[i], frame:float(), testSet.label[index][i]}, 4)
        --table.insert(input, testData[index][i]:cuda())
    end

    simulateSequence(output)
end


-- Function to play given sequence
function simulateSequence(sequence)
    local zoom = 8
    local h = sequence[1]:size(3)*zoom
    local w = sequence[1]:size(4)*zoom
    local qtwidget = require 'qtwidget'
    win = qtwidget.newwindow(w,h)
    for i, frame in pairs(sequence) do
        image.display{image = frame, win=win, zoom=zoom}
        sys.sleep(0.25)
    end
end
