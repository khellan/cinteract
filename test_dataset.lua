require 'torch'

mytest = {}
tester = torch.Tester()
paths.dofile('dataset.lua')

function mytest.scandir()
   local currentDir = scandir('.')
   local found = false
   for i, filename in ipairs(currentDir) do
      if filename == 'dataset.lua' then found = true end
   end
   tester:assert(found, 'scandir in dataset reads the right directory')
end

-- This implicitly tests createDataset, labelData, tableConcat
function mytest.mergeData()
   ds1 = {data = {'a', 'b'}, labels = {0, 0}}
   ds2 = {data = {'b', 'c'}, labels = {1, 1}}
   ds0 = mergeData({ds1, ds2})
   tester:asserteq(ds0.data[1], 'a', 'ds0.data[1] == a')
   tester:asserteq(ds0.data[2], 'b', 'ds0.data[2] == b')
   tester:asserteq(ds0.data[3], 'b', 'ds0.data[3] == b')
   tester:asserteq(ds0.data[4], 'c', 'ds0.data[4] == c')
   tester:asserteq(#ds0.data, 4, '#ds0.data == 4')
   tester:asserteq(ds0.labels[1], 0, 'ds0.labels[1] == 0')
   tester:asserteq(ds0.labels[2], 0, 'ds0.labels[2] == 0')
   tester:asserteq(ds0.labels[3], 1, 'ds0.labels[3] == 1')
   tester:asserteq(ds0.labels[4], 1, 'ds0.labels[4] == 1')
   tester:asserteq(#ds0.labels, 4, '#ds0.labels == 4')   
   tester:asserteq(ds0.size(), 4, 'ds0.size() == 4')
end

function mytest.expandData()
   table = {
      {{1, 128, 255},
      {32, 64, 96}}
   }
   source = torch.Tensor(table)

   expanded = expandData(source)
   tester:asserteq(expanded[1][1][1], 1,   "r channel equals grey original")
   tester:asserteq(expanded[1][1][2], 128, "r channel equals grey original")
   tester:asserteq(expanded[1][1][3], 255, "r channel equals grey original")
   tester:asserteq(expanded[1][2][1], 32,  "r channel equals grey original")
   tester:asserteq(expanded[1][2][2], 64,  "r channel equals grey original")
   tester:asserteq(expanded[1][2][3], 96,  "r channel equals grey original")

   tester:asserteq(expanded[2][1][1], 1,   "g channel equals grey original")
   tester:asserteq(expanded[2][1][2], 128, "g channel equals grey original")
   tester:asserteq(expanded[2][1][3], 255, "g channel equals grey original")
   tester:asserteq(expanded[2][2][1], 32,  "g channel equals grey original")
   tester:asserteq(expanded[2][2][2], 64,  "g channel equals grey original")
   tester:asserteq(expanded[2][2][3], 96,  "g channel equals grey original")

   tester:asserteq(expanded[3][1][1], 1,   "b channel equals grey original")
   tester:asserteq(expanded[3][1][2], 128, "b channel equals grey original")
   tester:asserteq(expanded[3][1][3], 255, "b channel equals grey original")
   tester:asserteq(expanded[3][2][1], 32,  "b channel equals grey original")
   tester:asserteq(expanded[3][2][2], 64,  "b channel equals grey original")
   tester:asserteq(expanded[3][2][3], 96,  "b channel equals grey original")

   tester:asserteq(expanded:size(1), 3, "three channels")
   tester:asserteq(expanded:size(2), 2, "width is 2 pixels")
   tester:asserteq(expanded:size(3), 3, "heigth is 3 pixels")
end

function mytest.convertCMYK2RGB()
   table = {
      {{0, 0.125}},    -- C
      {{0.25, 0.375}}, -- M
      {{0.5, 0.625}},  -- Y
      {{0.75, 0.875}}  -- K
   }
   source = torch.Tensor(table)
   
   expanded = convertCMYK2RGB(source)
   tester:asserteq(math.floor(expanded[1][1][1] + 0.5), 64,   "r channel is correct")
   tester:asserteq(math.floor(expanded[1][1][2] + 0.5), 28,   "r channel is correct")

   tester:asserteq(math.floor(expanded[2][1][1] + 0.5), 48,   "g channel is correct")
   tester:asserteq(math.floor(expanded[2][1][2] + 0.5), 20,   "g channel is correct")

   tester:asserteq(math.floor(expanded[3][1][1] + 0.5), 32,   "b channel is correct")
   tester:asserteq(math.floor(expanded[3][1][2] + 0.5), 12,   "b channel is correct")

   tester:asserteq(expanded:size(1), 3, "three channels")
   tester:asserteq(expanded:size(2), 1, "width is 1 pixel")
   tester:asserteq(expanded:size(3), 2, "heigth is 2 pixels")
end

tester:add(mytest)
tester:run()
