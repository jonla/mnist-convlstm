require 'torch'
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
   print('<mnist> done')

    local function expandToSequence(img, length)
        local h = img:size(3)
        local w = img:size(4)
        local cH = math.floor(h/4) -- curtain of height h/4
        local step = 2 -- steplentgh between frames
        local location = torch.random(1,h-cH)
        local direction = torch.random(0,1)*2 - 1 -- 1:down, -1: up

        -- Allocate sequence
        local sequence = {}
        for i=1,length do
            table.insert(sequence, img:clone())
        end

        for i=1,length do
            if cH+location+direction*step > h or location + direction*step < 1 then
                -- Flip direction of curtain if out of bounds
                direction = (-1)*direction
            end
            location = location + direction*step
            local frame = sequence[i]
            frame[{{1},{1},{location,location+cH}}]:fill(0)
        end
        return sequence
    end

   local dataset = {}
   for i=1,nExample do
       table.insert(dataset, expandToSequence(data[{{i}}], sequenceLength))
   end

   function dataset:size()
      return nExample
   end

   return dataset, data
end

