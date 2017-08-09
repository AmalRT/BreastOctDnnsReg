function [labels,scores] = valFnc(net, dataPath)

%Function used to get the class probabilties given a network and a
%(validation) data path.

g = gpuDevice(1);
reset(g);

load(dataPath);
net = vl_simplenn_move(net, 'gpu');
net.layers{end}.type = 'softmax';
labels= images.labels(images.set==3);




res = vl_simplenn(net,gpuArray(single(images.data(:,:,:,images.set==3))));
res2= reshape(res(end).x,2,length(labels));
scores = res2';

scores = gather(scores);

