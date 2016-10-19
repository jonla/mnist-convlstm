require 'torch'
require 'image'
require 'paths'

mnist = {}

mnist.path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
mnist.path_dataset = 'mnist.t7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train_32x32.t7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test_32x32.t7')

function mnist.download()
   if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
      local remote = mnist.path_remote
      local tar = paths.basename(remote)
      os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

function mnist.loadTrainSet(maxLoad, sequenceLength)
   return mnist.loadDataset(mnist.path_trainset, maxLoad, sequenceLength)
end

function mnist.loadTestSet(maxLoad, sequenceLength)
   return mnist.loadDataset(mnist.path_testset, maxLoad, sequenceLength)
end

function mnist.loadDataset(fileName, maxLoad, sequenceLength)
   mnist.download()

   local f = torch.load(fileName, 'ascii')
   local data = f.data:type(torch.getdefaulttensortype())
   --local labels = f.labels

   local nExample = f.data:size(1)
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
   end
   data = data[{{1,nExample},{},{},{}}]
   --data:add(-data:mean())
   --data:mul(1/data:std())
   data:add(-data:min())
   data:mul(1/data:max())
   --labels = labels[{{1,nExample}}]
    local h = data:size(3)
    local w = data:size(4)
   print('<mnist> done')

    local function expandToSequence(img, length)

        -- Draw dancing digits on blank canvas
        local iw = w*2/4 -- image scale is 1/2
        local iwC = w - iw
        local px = torch.random(1,iwC+1)
        local py = torch.random(1,iwC+1)
        local dx = torch.random(-1,1)
        local dy = torch.random(-1,1)
        local img = image.scale(img[1], iw)

        local labelSequence = {}
        for i=1,length do
            -- Flip direction if dancing of the edge
            if px+dx > iwC+1 or px+dx < 1 then
                dx = (-1)*dx
            end
            if py+dy > iwC+1 or py+dy < 1 then
                dy = (-1)*dy
            end
            px = px + dx
            py = py + dy
            local frame = torch.zeros(1,1,h,w)
            frame[{1,{1},{py,py+iw-1},{px,px+iw-1}}] = img
            table.insert(labelSequence, frame)
        end

        -- Occlude image with curtain
        local cH = math.floor(h/3) -- curtain of height h/3
        local step = 2 -- steplentgh between frames
        local lc = torch.random(1,h-cH)
        local dc = torch.random(0,1)*2 - 1 -- 1:down, -1: up

        dataSequence = {}
        for i=1,length do
            if cH+lc+dc*step > h or lc + dc*step < 1 then
                -- Flip direction of curtain if out of bounds
                dc = (-1)*dc
            end
            lc = lc + dc*step
            local frame = labelSequence[i]:clone()
            frame[{{1},{1},{lc,lc+cH}}]:fill(0)
            table.insert(dataSequence, frame)
        end
        return dataSequence, labelSequence
    end

   local inputset = {}
   local labelset = {}
   for i=1,nExample do
       local dataSequence, labelSequence = expandToSequence(data[{{i}}], sequenceLength)
       table.insert(inputset, dataSequence)
       table.insert(labelset, labelSequence)
   end

   local dataset = {}
   dataset.data = inputset
   dataset.label = labelset
   function dataset:size()
      return nExample
   end

    function dataset:getMinibatch()
        local indecies = torch.Tensor(opt.batchSize)
        indecies:random(1,dataset:size())
        --local label = torch.Tensor(opt.batchSize, 1, h, w):cuda()
        --for i=1,opt.batchSize do
        --    label[i] = trainLabels[indecies[i]]:cuda()
        --end

        local inputBatch = {}
        local labelBatch = {}
        for i=1,opt.rho do
            local frames = torch.Tensor(opt.batchSize, 1, h, w):cuda()
            local labels = torch.Tensor(opt.batchSize, 1, h, w):cuda()
            for j=1,opt.batchSize do
                frames[{{j}}] = dataset.data[indecies[j]][i]:cuda()
                labels[{{j}}] = dataset.label[indecies[j]][i]:cuda()
            end
            table.insert(inputBatch, frames)
            table.insert(labelBatch, labels)
        end
        return inputBatch, labelBatch
    end

   return dataset, labelset
end

