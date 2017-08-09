function c = border_det(im)

% Function used for surface detection
% It takes im the input image and follows the following steps:
%      1. Apply a Sobel edge detctor
%      2. Modify the detected edge in order to keep one and only one 
%       pixel per vertical line
%      3. Divide the vertical lines to 10 subsets (line i is assigned to
%       the set i mod 10).
%      4. Apply a spline interpolation on each of the the subsets, the
%      average them --> This operartions will reduce the effect of
%      outliers.
%      5. Apply a Rolling Ball of radius 10.
% The number of subsets and the value of the radius give satisfying results
% for our data, but can be modified if needed.

im_g = mat2gray(im);
bd_test = edge(im_g, 'sobel');
bd_improved = bd_test;
l = size(im,2);
for i=1:l
    I = find(bd_improved(:,i) == 1);
    if isempty(I)
        if i == 1
            bd_improved(ceil(size(im,1)/2),1) = 1;
        else
            bd_improved(:,i) = bd_improved(:,i-1);
        end;
    else
        if length(I) == 1 || I(1) > 20
            bd_improved(I(1)+1:end,i) = 0;
        else 
            bd_improved(1:I(2)-1,i) = 0;
            bd_improved(I(2)+1:end,i) = 0;
        end;
    end;
end;
y = zeros(l,1);
for i = 1:l
    y(i) = find(bd_improved(:,i)==1);
end;
c_all = zeros(l,10);
for i = 1:10
    x = (i:10:l);
    yi = y(x);
    c_all(:,i) = ceil(spline(x,yi, (1:l)'));
end;
c = mean(c_all,2) + 10;
c = Rolling_FilterB(c,10, 1);