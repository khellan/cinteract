require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

-- Name channels for convenience
channels = {'y','u','v'}

-- Directory lister copied from http://stackoverflow.com/questions/5303174/get-list-of-directory-in-a-lua
-- Added skipping of here and parent elements
function scandir(directory)
   local i, t, popen = 0, {}, io.popen
   for filename in popen('ls -a "'..directory..'"'):lines() do
      if filename:sub(1, 1) ~= "." then
         i = i + 1
         t[i] = filename
      end
   end
   return t
end

function loadJPEGImage(path)
   -- Copied from the torch image documentation at https://github.com/torch/image
   local fin = torch.DiskFile(path, 'r')
   fin:binary()
   fin:seekEnd()
   local file_size_bytes = fin:position() - 1
   fin:seek(1)
   local img_binary = torch.ByteTensor(file_size_bytes)
   fin:readByte(img_binary:storage())
   fin:close()
   -- Then when you're ready to decompress the ByteTensor:
   im = image.decompressJPG(img_binary)
   
   return im
end

function convertCMYK2RGB(source)
   local target = torch.Tensor(3, source:size(2), source:size(3))
   for i = 1, source:size(2) do
      for j = 1, source:size(3) do
         local c = source[1][i][j]
         local m = source[2][i][j]
         local y = source[3][i][j]
         local k = source[4][i][j]

         local r = 255 * (1 - c) * (1 - k)
         local g = 255 * (1 - m) * (1 - k)
         local b = 255 * (1 - y) * (1 - k)
         target[1][i][j] = r
         target[2][i][j] = g
         target[3][i][j] = b
      end
   end
   return target
end

function expandData(source)
   local target = torch.Tensor(#channels, source:size(2), source:size(3))
   for i = 1, #channels do
      target[i] = source[1]
   end
   
   return target
end

function loadLabelData(path, width, height)
   local list = scandir(path)
   print("list length: " .. #list)
   print("#channels  : " .. #channels)
   print("height     : " .. height)
   print("width      : " .. width)
   print("elements   : " .. #list * #channels * width * height)
   print("data = torch.Tensor(" .. #list .. ", " .. #channels .. ", " .. width .. ", " .. height .. ")")
   local samples = torch.Tensor(#list, #channels, width, height)
   for i, file in ipairs(list) do      
      im = loadJPEGImage(path..'/'..file)
      if im:size(1) == 1 then
         im = expandData(im)
      elseif im:size(1) == 4 then
         im = convertCMYK2RGB(im)
      elseif im:size(1) ~= 3 then
         print(file)
         print(im:size())
      end
      samples[i] = im
   end

   return samples:float()
end

function preprocessData(samples)
   for i = 1, samples:size(1) do
      samples[i] = image.rgb2yuv(samples[i])
   end
end

function findNorms(dataset)
   mean = {}
   std = {}
   for i, channel in ipairs(channels) do
      -- normalize each channel globally:
      mean[i] = dataset.samples[{ {},i,{},{} }]:mean()
      std[i] = dataset.samples[{ {},i,{},{} }]:std()
   end
   dataset.mean = mean
   dataset.std = std
end

function normalizeGlobally(dataset)
   for i,channel in ipairs(channels) do
      -- normalize each channel globally:
      dataset.samples[{ {},i,{},{} }]:add(-dataset.mean[i])
      dataset.samples[{ {},i,{},{} }]:div(dataset.std[i])
   end
end

function normalizeLocally(dataset)
   -- Define the normalization neighborhood:
   neighborhood = image.gaussian1D(13)

   -- Define our local normalization operator (It is an actual nn module, 
   -- which could be inserted into a trainable model):
   normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

   -- Normalize all channels locally:
   for c = 1, #channels do
      for i = 1,dataset:size() do
         dataset.samples[{ i,{c},{},{} }] = normalization:forward(dataset.samples[{ i,{c},{},{} }])
      end
   end
end

function normalizeData(dataset)
   findNorms(dataset)
   normalizeGlobally(dataset)
   normalizeLocally(dataset)
end

EMPTY_FLOAT_TENSOR = torch.FloatTensor()

function createDataset(samples, labels)
   return {
      samples = samples,
      labels = labels,
      size = function() if samples:dim() > 0 then return samples:size(1) else return 0 end end
   }
end

function emptyDataset()
   return createDataset(torch.FloatTensor(), torch.DoubleTensor())
end

function labelData(samples, label)
   local labels = torch.DoubleTensor(samples:size(1))
   for i = 1, samples:size(1) do
      labels[i] = label
   end
   labels:resize(samples:size(1))
   return createDataset(samples, labels)
end

function sampleAppend(s1, s2)
   local offset = 0
   if s1:dim() > 0 then offset = s1:size(1) end
   s1:resize(offset + s2:size(1), s2:size(2), s2:size(3), s2:size(4))
   for i = 1, s2:size(1) do
      s1[offset + i] = s2[i]
   end
   return s1
end

function labelAppend(l1, l2)
   local offset = 0
   if l1:dim() > 0 then offset = l1:size(1) end
   l1:resize(offset + l2:size(1))
   for i = 1, l2:size(1) do
      l1[offset + i] = l2[i]
   end
   return l1
end

function dataAppend(d1, d2)
   offset = d1:size(1)
   sampleAppend(d1.samples, d2.samples)
   labelAppend(d1.labels, d2.labels)
end

function loadData(path, width, height)
   local classes = {}
   local allData = emptyDataset()
   local labelDirs = scandir(path)
   for _, label in ipairs(labelDirs) do
      table.insert(classes, label)
      print("labelSamples = loadLabelData(" .. path .. '/' .. label .. ", " .. width .. ", " .. height .. ")")
      local samples = loadLabelData(path .. '/' .. label, width, height)
      local newData = labelData(samples, label)
      dataAppend(allData, newData)
   end
   return allData, classes
end

function printStatistics(name, samples)
   for i, channel in ipairs(channels) do
      mean = samples[{ {},i }]:mean()
      std = samples[{ {},i }]:std()

      print(name .. ' samples, ' .. channel .. '-channel, mean: ' .. mean)
      print(name .. ' samples, ' .. channel .. '-channel, standard deviation: ' .. std)
   end
end

function visualizeData(samples)
   if itorch then
      first256Samples_y = samples[{ {1,256},1 }]
      first256Samples_u = samples[{ {1,256},2 }]
      first256Samples_v = samples[{ {1,256},3 }]
      itorch.image(first256Samples_y)
      itorch.image(first256Samples_u)
      itorch.image(first256Samples_v)
   else
      print("For visualization, run this script in an itorch notebook")
   end
end
