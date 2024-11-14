function [betaout fitdata]=vmifitbeta_img(yin,ri,center,th1i,th2i,dthi)
%   [betaout fitdata]=vmifitbeta_img(yin,r,[center],[th1i],[th2i],[dthi])
%   calculate beta coefficients from abel inverted image yin.
%   th1,th2 and dth in degrees.
%   center = 2-number vector of center of the image [pix]
%   r=vector defining the bins of interest (along radius dimension). 
%   betaout(length(r)-1,4)= matrix of beta coefficients for all the radial bins 
%
%   this is Andrey's mod of Doug's fit betas to vmi data ("betaclc2.m"))
%   20160809ab: modded for speed and clarity 
%   (also, btw, the betaout is the row vector now). 
%   Using another im2pol function from matlabcentral by Bjorn Gustavsson ->
%   much faster and no a bit awkward restiction on max r being <=(rmax-5)
%   The legendre polynomials are now written out explicitely, for fun, as in
%   Wikipedia
%
%   20160809ab: "fitdata" output is foobar-ed at the moment. This is to be
%   sorted out in the next few days.

if ~exist('center','var')||length(center)<2, center=size(yin)/2;end;
xc=center(1); yc=center(2);
if ~exist('th1i','var')||isempty(th1i), th1i=0;end;
if ~exist('th2i','var')||isempty(th2i), th2i=90;end;
if ~exist('dthi','var')||isempty(dthi), dthi=1;end;

persistent th1 th2 dth y
if isempty(y) || th1 ~= th1i || th2 ~= th2i || dth ~= dthi,
    th1 = th1i; th2 = th2i; dth = dthi;
    th = th1:dth:th2;
    x=cos(pi*th/180); 
    P0=ones(size(th));
    P2=1/2*(3*x.^2-1);
    P4=1/8*(35*x.^4-30*x.^2+3);
    P6=1/16*(231*x.^6-315*x.^4+105*x.^2-5);
    y=[P0;P2;P4;P6];
end
% warning('off','MATLAB:rankDeficientMatrix');

%r=r1:r2;
r=ri(1):ri(end);
phi=(90-(th1:dth:th2))*pi/180;  %this is because in Legendre polynomial - the 
%angle is counted from polarization axis (we keep it vertical on the image),
%but the img2pol_bjorn assumes usual convention that polar angle is counted
%of the abscissa axis. Thus phi=90-theta;
persistent R Phi
if isempty(R)|| isempty(Phi) || size(R,2)~=length(r) || size(Phi,1)~=length(phi),
%if isempty(R)|| isempty(Phi) || sum(R(1,:)~=r) || sum(Phi(:,1)~=phi'),
    [R,Phi] = meshgrid(r,phi);
end;
yinp=img2pol_bjorn(yin,xc,yc,R,Phi,'cubic');

betaout=zeros(length(ri)-1,4);
for i=1:(length(ri)-1)
    %fitdata=sum(yinp,2)/(r2-r1+1); %average signal between r1 and r2
    r1=ri(i);
    r2=ri(i+1);
    fitdata=sum(yinp(:,(r1:r2)-ri(1)+1),2)/(r2-r1+1); %average signal between r1 and r2
    betaout(i,:)=(y' \ fitdata)'; %least square fit
                         %transposing -> trying to arrange for "betaout" to be a row vector
end