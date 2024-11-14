function [betaout fitdata]=vmifitbeta_stk(stkin,r,center,th1i,th2i,dthi)
% [betaout fitdata]=vmifitbeta_stk(stkin,r,[center],[th1i],[th2i],[dthi])



if ~iscell(stkin), stkin={stkin};end;
if ~exist('center','var')||length(center)<2, center=size(stkin{1})/2;end;
%xc=center(1); yc=center(2);
if ~exist('th1i','var')||isempty(th1i), th1i=0;end;
if ~exist('th2i','var')||isempty(th2i), th2i=90;end;
if ~exist('dthi','var')||isempty(dthi), dthi=1;end;

nz=length(stkin);
nr=length(r)-1;

if nz==1, 
    %betaout=zeros(length(r)-1,4);
    betaout=vmifitbeta_img(stkin{1},r,center,th1i,th2i,dthi);
elseif nr==1,
    betaout=zeros(nz,4);
    for i=1:nz,
        betaout(i,:)=vmifitbeta_img(stkin{i},r,center,th1i,th2i,dthi);
        disp([num2str(i) '/' num2str(nz)]);
    end;
else    
    betaout=zeros(nz,length(r)-1,4);
    for i=1:nz,
        %[betaout{i} fitdata{i}]=vmifitbeta_img(stkin{i},r,center,th1i,th2i,dthi);
        betaout(i,:,:)=vmifitbeta_img(stkin{i},r,center,th1i,th2i,dthi);
        disp([num2str(i) '/' num2str(nz)]);
    end;
end;
    
fitdata=[];
return
