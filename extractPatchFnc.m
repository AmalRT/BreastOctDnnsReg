function t_end = extractPatchFnc(config, display, extractAir,reg)

%This function is used to extract the patches for training, test, and
% regularization (using data distribution).

%config= 'a-c', 'b-c', 'c-a' or 'c-b'
%       designates the training-test samples. As a and b come from the 
%       same patient, the use of these data configurations insures that 
%       we train and test on samples from different patients.
%display= 'true' or 'false'
%       Put this variable to true if you want to display the surface
%       detection.
%extractAir= 'true' or 'false'
%       If extractAir is true, then the function will extract the air 
%       patches. These patches can be used as a third class for classification. 
%       This is not the strategy that we adopt in the paper, as the air patches
%       can be confusing, and as the surface detection is enough to remove
%       the air part during test.
%reg= 'true' or 'false'
%       designates if regularization patches need to be extracted. These
%       are needed when you want to use the data distrubtion regularization
%       method.

%---------------------------------------------------------------------
%Prepare folder names
%source_pth_*: contain the OCT images that are already (manually) cropped 
%    to contain only the useful information
%save_pth_*: contain the paths where the patches are saved
%---------------------------------------------------------------------
source_pth_a = '/media/amal-rt/D83CB42A3CB40612/Breast_OCT/20161005/Sample-a';
source_pth_b = '/media/amal-rt/D83CB42A3CB40612/Breast_OCT/20161005/Sample-b';
source_pth_c = '/media/amal-rt/D83CB42A3CB40612/Breast_OCT/20161005/Sample-c';
source_reg = '/media/amal-rt/D83CB42A3CB40612/Breast_OCT/20161005/Regularization';
save_pth_tr = ['/home/amal-rt/Documents/Back-up-2/OCT_NN/DataNN/Training/',config,'/'];
save_pth_test = ['/home/amal-rt/Documents/Back-up-2/OCT_NN/DataNN/Test/',config,'/'];
save_pth_reg = '/home/amal-rt/Documents/Back-up-2/OCT_NN/DataNN/Regularization/';
types = {'Cancer', 'Normal'};
types_reg = {'Tissue'};
if extractAir
    types{3} = 'Air';
    types_reg{2} = 'Air';
end;

%---------------------------------------------------------------------
%Prepare folders
%---------------------------------------------------------------------

for i=1:length(types)
    mkdir([save_pth_tr,types{i}]);
    mkdir([save_pth_test, types{i}]);
end;
if reg
    for i=1:length(types_reg)
        mkdir([save_pth_reg  , types_reg{i}]);
       
    end;
end;
disp('Folders created')

switch config
    case 'a-c'
        training = source_pth_a;
        test = source_pth_c;
    case 'b-c'
        training = source_pth_b;
        test = source_pth_c;
    case 'c-a'
        training = source_pth_c;
        test = source_pth_a;
    case 'c-b'
        training = source_pth_c;
        test = source_pth_b;
    otherwise
        error('unknown configuration');
end;
g = fspecial('gaussian', 10, 3);        

%---------------------------------------------------------------------
% Extracting training patches
%---------------------------------------------------------------------
disp('Beginning extracting patches');
if extractAir
    save_pth2 = [save_pth_tr,types{3}];
    nb2 = 1;
end;
for tp= 1:2
    nb1 = 1;
    source_pth = [training,'/',types{tp}];
    save_pth1 = [save_pth_tr,types{tp}];
    d = dir(source_pth);
    k =1;
    while d(k).name(1) ~= '2'
        d(k) = [];
    end;
    for i=1:3:length(d)-2
        disp(['Training patches for ',types{tp}, ': Images ', num2str(i), ' to ', num2str(i+2), 'out of ', num2str(length(d))]);
        filename = [source_pth,'/', d(i).name];
        load(filename);
        c = ceil(border_det( imfilter(im, g)));
        filename = [source_pth,'/', d(i+1).name];
        im(:,:,2) = importdata(filename);
        c(:,2) = ceil(border_det(imfilter(im(:,:,2), g)));
        filename = [source_pth,'/', d(i+2).name];
        im(:,:,3) =importdata(filename);
        c(:,3) = ceil(border_det( imfilter(im(:,:,3), g)));
        if display
            imagesc(im(:,:,3)), colormap gray;
            hold on, plot(c(:,3),'r');
        end;
        step = 128;
        for j =1:step:size(im,2)-63
            k1 = max(max(c(j:j+63,:)));
            k2 = max(max(max(c(j:j+63,:))) - 30,1);
            for k = k1:step:min(size(im,1)-63,k1+193)
                im_ = im(k:k+63,j:j+63,:);
                im_ = imresize(im_, [32, 32]);
                if nb1<10
                    savename = [save_pth1,'/patch_000', num2str(nb1),'.mat'];
                else if  nb1<100
                        savename = [save_pth1,'/patch_00', num2str(nb1),'.mat'];
                    else if nb1<1000
                            savename = [save_pth1,'/patch_0', num2str(nb1),'.mat'];
                        else
                            savename = [save_pth1,'/patch_', num2str(nb1),'.mat'];
                        end;
                    end;
                end;
                save(savename,'im_');
                nb1 = nb1+1;
            end;
            if extractAir
                for k = 1:step:k2-63
                    im_ = im(k:k+63,j:j+63,:);
                    im_= imresize(im_,[32,32]);
                    if nb2<10
                        savename = [save_pth2,'/patch_000', num2str(nb2),'.mat'];
                    else if  nb2<100
                            savename = [save_pth2,'/patch_00', num2str(nb2),'.mat'];
                        else if nb2<1000
                                savename = [save_pth2,'/patch_0', num2str(nb2),'.mat'];
                            else
                                savename = [save_pth2,'/patch_', num2str(nb2),'.mat'];
                            end;
                        end;
                    end;
                    save(savename,'im_');
                    nb2 = nb2+1;
                end;
            end;
            
            if display
                drawnow;
                pause(0.05);
            end;
        end;
    end;
end;

%---------------------------------------------------------------------
% Extracting test patches
%---------------------------------------------------------------------
t = tic;
if extractAir
    save_pth2 = [save_pth_test,types{3}];
    nb2 = 1;
end;
for tp= 1:2
    nb1 = 1;
    source_pth = [test,'/',types{tp}];
    save_pth1 = [save_pth_test,types{tp}];
    d = dir(source_pth);
    k =1;
    while d(k).name(1) ~= '2'
        d(k) = [];
    end;
    for i=1:3:length(d)-2
        disp(['Test patches for ',types{tp}, ': Images ', num2str(i), ' to ', num2str(i+2), 'out of ', num2str(length(d))]);
        filename = [source_pth,'/', d(i).name];
        load(filename);
        c = ceil(border_det( imfilter(im, g)));
        filename = [source_pth,'/', d(i+1).name];
        im(:,:,2) = importdata(filename);
        c(:,2) = ceil(border_det(imfilter(im(:,:,2), g)));
        filename = [source_pth,'/', d(i+2).name];
        im(:,:,3) =importdata(filename);
        c(:,3) = ceil(border_det( imfilter(im(:,:,3), g)));
        if display
            imagesc(im(:,:,3)), colormap gray;
            hold on, plot(c(:,3),'r');
        end;
        
        step = 32;
        for j =1:step:size(im,2)-63
            k1 = max(max(c(j:j+63,:)));
            k2 = max(max(max(c(j:j+63,:))) - 30,1);
            for k = k1:step:min(size(im,1)-63,k1+129)
                im_ = im(k:k+63,j:j+63,:);
                im_ = imresize(im_, [32, 32]);
                if nb1<10
                    savename = [save_pth1,'/patch_000', num2str(nb1),'.mat'];
                else if  nb1<100
                        savename = [save_pth1,'/patch_00', num2str(nb1),'.mat'];
                    else if nb1<1000
                            savename = [save_pth1,'/patch_0', num2str(nb1),'.mat'];
                        else
                            savename = [save_pth1,'/patch_', num2str(nb1),'.mat'];
                        end;
                    end;
                end;
                save(savename,'im_');
                nb1 = nb1+1;
            end;
            if extractAir
                for k = 1:step:k2-63
                    im_ = im(k:k+63,j:j+63,:);
                    im_= imresize(im_,[32,32]);
                    if nb2<10
                        savename = [save_pth2,'/patch_000', num2str(nb2),'.mat'];
                    else if  nb2<100
                            savename = [save_pth2,'/patch_00', num2str(nb2),'.mat'];
                        else if nb2<1000
                                savename = [save_pth2,'/patch_0', num2str(nb2),'.mat'];
                            else
                                savename = [save_pth2,'/patch_', num2str(nb2),'.mat'];
                            end;
                        end;
                    end;
                    save(savename,'im_');
                    nb2 = nb2+1;
                end;
            end;
        end;
        if display
            drawnow;
            pause(0.05);
        end;
    end;
end;
t_end = toc(t);
disp('Extracting patches done');

%---------------------------------------------------------------------
% Extracting regularization patches
%---------------------------------------------------------------------
if reg
if extractAir
    save_pth2 = [save_pth_reg,types_reg{2}];
    nb2 = 1;
end;
    nb1 = 1;
    source_pth = source_reg;
    save_pth1 = [save_pth_reg,types_reg{1}];
    d = dir(source_pth);
    k =1;
    while d(k).name(1) ~= '2'
        d(k) = [];
    end;
    for i=1:3:length(d)-2
        disp(['Training patches for ',types_reg{1}, ': Images ', num2str(i), ' to ', num2str(i+2), 'out of ', num2str(length(d))]);
        filename = [source_pth,'/', d(i).name];
        im = importdata(filename);
        c = ceil(border_det( imfilter(im, g)));
        filename = [source_pth,'/', d(i+1).name];
        im(:,:,2) = importdata(filename);
        c(:,2) = ceil(border_det(imfilter(im(:,:,2), g)));
        filename = [source_pth,'/', d(i+2).name];
        im(:,:,3) =importdata(filename);
        c(:,3) = ceil(border_det( imfilter(im(:,:,3), g)));
        if display
            imagesc(im(:,:,3)), colormap gray;
            hold on, plot(c(:,3),'r');
        end;
        step = 64;
        for j =1:step:size(im,2)-63
            k1 = max(max(c(j:j+63,:)));
            k2 = max(max(max(c(j:j+63,:))) - 30,1);
            for k = k1:step:min(size(im,1)-63,k1+193)
                im_ = im(k:k+63,j:j+63,:);
                im_ = imresize(im_, [32, 32]);
                if nb1<10
                    savename = [save_pth1,'/patch_000', num2str(nb1),'.mat'];
                else if  nb1<100
                        savename = [save_pth1,'/patch_00', num2str(nb1),'.mat'];
                    else if nb1<1000
                            savename = [save_pth1,'/patch_0', num2str(nb1),'.mat'];
                        else
                            savename = [save_pth1,'/patch_', num2str(nb1),'.mat'];
                        end;
                    end;
                end;
                save(savename,'im_');
                nb1 = nb1+1;
            end;
            if extractAir
                for k = 1:step:k2-63
                    im_ = im(k:k+63,j:j+63,:);
                    im_= imresize(im_,[32,32]);
                    if nb2<10
                        savename = [save_pth2,'/patch_000', num2str(nb2),'.mat'];
                    else if  nb2<100
                            savename = [save_pth2,'/patch_00', num2str(nb2),'.mat'];
                        else if nb2<1000
                                savename = [save_pth2,'/patch_0', num2str(nb2),'.mat'];
                            else
                                savename = [save_pth2,'/patch_', num2str(nb2),'.mat'];
                            end;
                        end;
                    end;
                    save(savename,'im_');
                    nb2 = nb2+1;
                end;
            end;
            
            if display
                drawnow;
                pause(0.05);
            end;
        end;
    end;
end;


