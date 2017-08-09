function output = Rolling_FilterB(input,radius,deltax)

%THIS CODE IS TAKEN FROM THE MATLAB CENTRAL FORUM

%performs rolling ball filter for ball below input
%input is a column vector of sample values
%radius is the radius of the ball rolling under graph
%deltax is spacing of sample points

N = size(input,1); %input(1),...,input(N)
K=floor(radius/deltax); %K is number of neighbors
output = input - radius; %Constrain from each point

for k=1:K %Constrain from kth left/right neighbors
  st = k + 1; en = N - k; %start and end markers
  V = input-sqrt(radius^2 - (k*deltax)^2);
  output(1:en) = min( output(1:en),V(st:N) ); %left
  output(st:N) = min( output(st:N),V(1:en) ); %right
end

