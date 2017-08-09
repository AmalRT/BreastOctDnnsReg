function [pred,t] = predict_im(path, ov1, ov2, ov3, T ,s)

%Function used to predict the class probabilities for overlapping patches
%in mixed tissue images: ovx designates the number of overlapping pixels
%(=size - ovx) in the dimension x.

load(path);
disp('loading data');
testFolder = '/media/amal-rt/D83CB42A3CB40612/Breast_OCT/20161005/Test/';
d = dir(testFolder);
d(1:2) =[];

% Contrast correction. ImgRange and ImgOffset are set manually to get the
% best image rendering. 
ImgRange = 60;
ImgOffset = 40;
for i =1:60
    im = importdata([testFolder,d(i).name]);
    AbsBuffer = im*256;
    ScaledImg = (256/ImgRange)*(AbsBuffer-ImgOffset);
    ScaledImg(find(ScaledImg>256)) = 255;
    ScaledImg(find(ScaledImg<2)) = 2;
    ScaledImg = ScaledImg/255;
    testDat(:,:,i) = single(ScaledImg);
end;
disp('data loaded');
[m,n,p] = size(testDat);
pred_ = zeros(m,n,p);
occ_ = zeros(m,n,p);
%Beginning of prediction
t0 = tic();
net.layers{end}.type = 'softmax';
%Surface detection
g = fspecial('gaussian', 10,3);
for j = 1:ov3:p-2
    c(:,1) = ceil(border_det(imfilter(testDat(:,:,j),g)));
    c(:,2) = ceil(border_det(imfilter(testDat(:,:,j+1),g)));
    c(:,3) = ceil(border_det(imfilter(testDat(:,:,j+2),g)));
    c = max(c,1);
%Predict and add probabilities, count the number of times a pixel is used     
    for i = 1:ov1:n-(s-1)
        disp(['Sample begins at col ', num2str(i), ' out of ', num2str(n), ' and depth ', num2str(j), ' out of ', num2str(p)]);
        k1 = ceil(mean(mean(c(i:i+(s-1),:))));
        k2 = max(max(c(i:i+(s-1),:)));
        for k = k1:ov2:min(k2+321, m-(s-1))
            im_ = testDat(k:k+(s-1),i:i+(s-1),j:j+2);
            if s~=32
                im_= imresize(im_, [32 32]);
            end;
            res2 = vl_simplenn(net,im_);
            res3 = reshape(res2(end).x,2,1);
            if res3(1) >= T
                class = 1;
            else
                class = 2;
            end;
            pred_(k:k+(s-1),i:i+(s-1),j:j+2) = pred_(k:k+(s-1),i:i+(s-1),j:j+2) + class*ones(s,s,3);
            occ_(k:k+(s-1),i:i+(s-1),j:j+2) = occ_(k:k+(s-1),i:i+(s-1),j:j+2) + ones(s,s,3);
        
            
        end;
        
    end;
    for i=1:n
        pred_(1:c(i,1),i,j) = 0;
        pred_(c(i,1)+384:end,i,j) = 0;
        pred_(1:c(i,2),i,j+1) = 0;
        pred_(c(i,1)+384:end,i,j+1) = 0;
        pred_(1:c(i,3),i,j+2) = 0;
        pred_(c(i,1)+384:end,i,j+2) = 0;
    end;
end;
% Average predictions
pred = pred_./occ_;
t = toc(t0);
save('/home/amal-rt/Documents/Back-up-2/OCT_NN/testDB1.mat','testDat','-v7.3');
