function [Y_n x_n_bins]=rebin3(x,Y,varargin)
%Rebins existing data Y(x) into Y_new(x_new).
%[Y_new x_new] = rebin(x,Y,nbins)
%or      Y_new = rebin(x,Y,x_new)
% November 27, 2008.  Andrey Boguslavskiy
%N.B. This is a BETA. Works in general but some stuff is still unfinished..
%P.S. "rebin2" can handle 2D Y(x) data.
%N.B. 'x_new' must be a monotonically _increasing_ array 
%P.S. "rebin3" "does not care" about orientation of the "Y" array. It
%chooses the dimention with the length==length(x);

%last edited May 13,2009 Andrey Boguslavskiy
if length(varargin)<1, eval('help rebin'); Y_n=[]; return; end

transpose=0;
if size(Y,1) ~=length(x), if size(Y,2)==length(x), Y=Y'; transpose=1;end;end;

switch length(varargin{1})
    case 1
        n=varargin{1}; %number of bins
        x_n_grid = linspace(x(1),x(end),n+1); %size of the grid is +1 of x_n_bins
        x_n_bins = x_n_grid(2:end);
    otherwise

        if x(1)>x(end), x=x(end:-1:1); Y=Y(end:-1:1,:); end
        
        x_n_bins = varargin{1};
        x_n_grid=[(2*x_n_bins(1)-x_n_bins(2)) x_n_bins]; %create extra point for the x_n_grid
        n=length(x_n_bins);
end

Y_size=length(x);%length(Y);
Y_n = zeros(n,size(Y,2));

x_n_indx=interp1(x,1:length(x),x_n_grid); %array of fractional indices
for i=1:n+1, if isnan(x_n_indx(i)), x_n_indx(i)=1; else break; end; end %clean NaNs in front. They appear in case of accidental extrapolation
for i=(n+1):-1:1, if isnan(x_n_indx(i)), x_n_indx(i)=Y_size-1; else break; end; end %clean NaNs at the tail
%IMPORTANT! : programming is not finished yet! "Y_size-1" on the previous
%line is a quick fix, a hack, which leads to artefacts in "extrapolated" data, instead of zeros.
%Right way to fix the problem is to have "Y_size" here but to work out
%stuff 10 lines below this line. Decide on how you want bins sampling area
%to be located with respect to bin position(to the left, or centered)? 


    for i=2:n+1          
        jb=floor(x_n_indx(i-1));
        jt=floor(x_n_indx(i));
        sum=zeros(1,size(Y,2));      
        for j=(jb+1):jt
            sum=sum + Y(j,:);
        end
        if jt==Y_size %for the slowest tof bin
            sum=sum + (jb+1-x_n_indx(i-1))*(Y(jb+1,:)+Y(jb,:))/2 + (x_n_indx(i)-jt)*(Y(jt,:));
        else % for the rest
            sum=sum + (jb+1-x_n_indx(i-1))*(Y(jb+1,:)+Y(jb,:))/2 + (x_n_indx(i)-jt)*(Y(jt,:)+Y(jt+1,:))/2;
        end
        Y_n(i-1,:)=sum;
    end

if transpose, Y_n=Y_n'; end;