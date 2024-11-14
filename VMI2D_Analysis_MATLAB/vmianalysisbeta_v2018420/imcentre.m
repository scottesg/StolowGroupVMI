function [x0,y0]=imcentre(var,verbosity)
%[x0,y0]=imcentre(full_file_name) or
%[x0,y0]=imcentre(data_array)

%ab20160513,20161027

sz=size(var);
if sz(1)>1,
    data=var;
else
    data=imread(var);
    if isempty(data), disp(['imcentre(): file ' var ' is not found']); return; end;
end

%data_inv=rot90(data,2);
%C=xcorr2(double(data),double(data_inv));
A=double(data); %C = conv2(A, rot90(conj(A),2));
%C = conv2(A, A);               %extremely slow in R2007b. reasonable in R2010b
%C = convolve2(A, A);           %dunno. requires matlab>=2013.
C = conv_fft2(A,A);             %very fast!
[y0,x0]=find(C==max(max(C)));

x0=round(x0/2);
y0=round(y0/2);

%y0x0=[y0,x0];

if exist('verbosity','var'),
figure;imagesc(data); hold on; circle(x0,y0,220); circle(x0,y0,150); circle(x0,y0,100); 
line([x0 x0],[1 2*y0],'linestyle',':','color','r');
line([1 2*x0],[y0 y0],'linestyle',':','color','r');
hold off; axis image; title(['[x0,y0]=' num2str(x0) ', ' num2str(y0)]);
end;
end

%usage examples:
%1):    imcentre(vmistats.imavgs.img-vmistats.imavgs.bgb,'');
%2):    [x0,y0]=imcentre(vmistats.imavgs.img-vmistats.imavgs.bgb);


%figure; surf(flipud(vmistats.imavgs.img-vmistats.imavgs.bgb)); shading interp; axis image; colormap(colorcube); 
