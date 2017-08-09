function imdbsConstructFnc(k, dataFolder, config, extractAir)

% This function is used to construct the databases in matconvnet format
%k = number of folds used for cross-validation. For 5-fold
%           cross-validation, k = 5.
%dataForlder = where to save the constructed database
%config= 'a-c', 'b-c', 'c-a' or 'c-b'
%       designates the training-test samples. As a and b come from the 
%       same patient, the use of these data configurations insures that 
%       we train and test on samples from different patients.
%extractAir= 'true' or 'false'
%       If extractAir is true, then the function will extract the air 
%       patches. These patches can be used as a third class for classification. 
%       This is not the strategy that we adopt in the paper, as the air patches
%       can be confusing, and as the surface detection is enough to remove
%       the air part during test.

d = dir(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Training/',config,'/Cancer']);
images.data = [];
images.set = [];
images.labels = [];

disp('Constructing global database');
for i=3:length(d)   
    im = importdata(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Training/',config,'/Cancer/',d(i).name]);
    images.data(:,:,:,i-2) = single(im);
    images.set(end+1) = 1;
    images.labels(end+1) = 1;
end;
l = size(images.data,4);
nc = l;
disp('done for cancer - training');

d = dir(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Training/',config,'/Normal']);

for i=3:length(d)
    
    im =importdata(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Training/',config,'/Normal/',d(i).name]);
    images.data(:,:,:,l+i-2) = single(im);
    images.set(end+1) = 1;
    images.labels(end+1) = 2;
end;
l = size(images.data,4);
nn = l-nc;
disp('done for normal - training');

if extractAir
    d = dir(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Training/',config,'/Air']);
    
    for i=3:length(d)
        
        im = importdata(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Training/',config,'/Air/',d(i).name]);
        images.data(:,:,:,l+i-2) = single(im);
        images.set(end+1) = 1;
        images.labels(end+1) = 3;
    end;
    l = size(images.data,4);
    disp('done for air - training');
end;


d = dir('/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Regularization/Tissue');

for i=3:length(d)
    
    im =importdata(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Regularization/Tissue/',d(i).name]);
    images.data(:,:,:,l+i-2) = single(im);
    images.set(end+1) = 4;
    images.labels(end+1) = 0;
end;
l = size(images.data,4);
disp('done for regularization');
if extractAir
    d = dir('/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Regularization/Air');
    
    for i=3:length(d)
        
        im = importdata(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Training/Air/',d(i).name]);
        images.data(:,:,:,l+i-2) = single(im);
        images.set(end+1) =4;
        images.labels(end+1) = 0;
    end;
    l = size(images.data,4);
    disp('done for air - regularization');
end;


images.data_mean = mean(images.data,4);
meta.sets = {'train', 'val', 'test'};
meta.classes = {1,2};

I1 = find(images.set ==1);
I2 = find(images.set == 3);
I3 =  find(images.set == 4);
I = [I1,I2,I3];
images.data = images.data(:,:,:,I);
images.labels = images.labels(I);
images.set = images.set(I);

disp('Constructing k-folds for cross validation');

foldSz1 = floor(nc/k);
foldSz2 = floor(nn/k);
for i=1:k
    disp(['Fold ',num2str(i)]);
    valIdx= [((i-1)*foldSz1+1:i*foldSz1),nc+((i-1)*foldSz2+1:i*foldSz2)];
    images.set(valIdx)=3;
    I1 = find(images.set ==1);
    I2 = find(images.set == 3);
    I3 =  find(images.set == 4);
    I = [I1,I2,I3];
    images.data = images.data(:,:,:,I);
    images.labels = images.labels(I);
    images.set = images.set(I);
    save([dataFolder,'imdb',num2str(i),'.mat'], 'images', 'meta', '-v7.3');
    [~,I_org] =sort(I);
    images.data = images.data(:,:,:,I_org);
    images.labels = images.labels(I_org);
    images.set = images.set(I_org);
    images.set(valIdx) = 1;
end;

images.set(images.set==3) =1;
disp('Adding test data')
d = dir(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Test/',config,'/Cancer']);

for i=3:length(d)   
    im = importdata(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Test/',config,'/Cancer/',d(i).name]);
    images.data(:,:,:,l+i-2) = single(im);
    images.set(end+1) = 3;
    images.labels(end+1) = 1;
end;
l = size(images.data,4);
disp('done for cancer - test');

d = dir(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Test/',config,'/Normal']);

for i=3:length(d)
    
    im =importdata(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Test/',config,'/Normal/',d(i).name]);
    images.data(:,:,:,l+i-2) = single(im);
    images.set(end+1) = 3;
    images.labels(end+1) = 2;
end;
l = size(images.data,4);
disp('done for normal - test');

if extractAir
    d = dir(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Test/',config,'/Air']);
    
    for i=3:length(d)
        
        im = importdata(['/home/amal-rt/Documents/Back-up-2/OCT_NN/Data/Test/',config,'/Air/',d(i).name]);
        images.data(:,:,:,l+i-2) = single(im);
        images.set(end+1) = 3;
        images.labels(end+1) = 3;
    end;
    
    disp('done for air - test');
end;

images.data_mean = mean(images.data,4);
I1 = find(images.set ==1);
I2 = find(images.set == 3);
I3 =  find(images.set == 4);
I = [I1,I2,I3];
images.data = images.data(:,:,:,I);
images.labels = images.labels(I);
images.set = images.set(I);


mkdir(dataFolder);
save([dataFolder,'imdb.mat'], 'images', 'meta', '-v7.3');
