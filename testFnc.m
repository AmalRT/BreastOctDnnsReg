function [labels,scores,test_time] = testFnc(net, testFolder)
%Function used to get the class probabilties given a network and a
%(test) data path.

d_c= dir([testFolder, 'Cancer']);
d_c(1:2) =[];
d_n= dir([testFolder, 'Normal']);
d_n(1:2) =[];
pos = length(d_c);
neg = length(d_n);

net = vl_simplenn_move(net, 'gpu');
net.layers{end}.type = 'softmax';

labels= [ones(1,pos),2*ones(1,neg)];
scores = gpuArray(zeros(pos+neg,2));

t_st = tic;



for i=1:pos
    im = gpuArray(single(importdata([testFolder,'Cancer/', d_c(i).name])));    
res = vl_simplenn(net,single(im));
res2= reshape(res(end).x,2,1);
scores(i,:) = res2';

end;



for i=1:neg
    im= gpuArray(single(importdata([testFolder,'Normal/', d_n(i).name])));    
res = vl_simplenn(net,single(im));
res2= reshape(res(end).x,2,1);
scores(pos+i,:) = res2';
end;


test_time= toc(t_st);
scores = gather(scores);

