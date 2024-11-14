function vmistats=vmicropdifstk(vmistats,cropsize)
%function vmistats=vmicropdifstk(vmistats[,cropsize])
%   Finds center of the vmi rings and symmetrically cropps image around it.
%   When crop size is specified (like:  vmistats=vmicropdifstk(vmistats,[450,450]);
%   it defines the size of the image, else the largest possible square
%   centered on the vmi rings centre is auto-determined and applied.

%NB for now assuming the center of the vmi rings is always between pixels (therefore use of 'floor'), 
%   thus the cropped square image will have even number of pixels.
%   Need to think if there is a better way to deal with the case when the
%   vmi centre is inbetween 2 pixels (the tricky part is what to do if it 
%   is in-between two pixels along one dimension and right on a pixel on 
%   another dimension)
%ab201606-07

% %IMAGE CENTRE
% disp('Finding image center');
% if length(vmistats.imavgs.bgb)~=1, data=vmistats.imavgs.img-vmistats.imavgs.bgb; else data=vmistats.imavgs.img;end;
% %data_inv=rot90(data,2);
% %C=xcorr2(double(data),double(data_inv));
% A=double(data); %C = conv2(A, rot90(conj(A),2));
% %C = conv2(A, A);               %extremely slow in R2007b. reasonable in R2010b
% %C = convolve2(A, A);           %dunno. requires matlab>=2013.
% C = conv_fft2(A,A);             %very fast!
% [y0,x0]=find(C==max(max(C)));
% % x0=round(x0/2);
% % y0=round(y0/2);
% x0=floor(x0/2);
% y0=floor(y0/2);
% [x0,y0]
% vmistats.vmicentre=[y0,x0];
% disp('Done.');
y0=vmistats.vmicentre(1);
x0=vmistats.vmicentre(2);


%CROP
[szy,szx]=size(vmistats.imstks.difstk{1});
% y0=vmistats.vmicentre(1);
% x0=vmistats.vmicentre(2);

if ~exist('cropsize','var')||isempty(cropsize),
    halfcrop=min([x0-1,szx-x0,y0-1,szy-y0]);
else
    halfcrop=cropsize(1)/2;
end;

delaysN=length(vmistats.imstks.difstk);

for j=1:delaysN, 
    vmistats.imstks.difstk{j}=vmistats.imstks.difstk{j}(y0-halfcrop+1:y0+halfcrop,x0-halfcrop+1:x0+halfcrop); 
    %vmistats.imavgs.difstk=vmistats.imavgs.difstk+vmistats.imstks.difstk{j}/delaysN;
end;
vmistats.imavgs.difstk=vmistats.imavgs.difstk(y0-halfcrop+1:y0+halfcrop,x0-halfcrop+1:x0+halfcrop);
%%%
% vmistats.imavgs.img=vmistats.imavgs.img(y0-halfcrop+1:y0+halfcrop,x0-halfcrop+1:x0+halfcrop);
% vmistats.imavgs.bgb=vmistats.imavgs.bgb(y0-halfcrop+1:y0+halfcrop,x0-halfcrop+1:x0+halfcrop);
% vmistats.imavgs.imgp=vmistats.imavgs.imgp(y0-halfcrop+1:y0+halfcrop,x0-halfcrop+1:x0+halfcrop);
% vmistats.imavgs.bgbp=vmistats.imavgs.bgbp(y0-halfcrop+1:y0+halfcrop,x0-halfcrop+1:x0+halfcrop);
% vmistats.imavgs.imge=vmistats.imavgs.imge(y0-halfcrop+1:y0+halfcrop,x0-halfcrop+1:x0+halfcrop);
% vmistats.imavgs.bgbe=vmistats.imavgs.bgbe(y0-halfcrop+1:y0+halfcrop,x0-halfcrop+1:x0+halfcrop);
%%%
vmistats.difstk_imsize=size(vmistats.imstks.difstk{1});
return;
