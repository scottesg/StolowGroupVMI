function vmistats=vmidetgaincorr(vmistats,detgainprofile,verbosity)

%I guess it would make sense to apply this correction to the po and eo
%average images as well?

if detgainprofile(end-3:end)=='.png',
    detgain=double(imread(detgainprofile,'png'))/2^16;
elseif detgainprofile(end-3:end)=='.mat',
    tmp=load(detgainprofile);
    detgain=tmp.detgain;
end;

detgain=smoothimg(detgain,10);
detgain(detgain<0.05)=0.05;
%%%
% vmistats.imstks.difstk_raw=vmistats.imstks.difstk;
% vmistats.imavgs.difstk_raw=vmistats.imavgs.difstk;
%%%
delaysN=length(vmistats.delays);
for j=1:delaysN,vmistats.imstks.difstk{j}=vmistats.imstks.difstk{j}./(detgain);end;
vmistats.imavgs.difstk=vmistats.imavgs.difstk ./ detgain;
%%%
vmistats.imavgs.img=vmistats.imavgs.img ./ detgain;
vmistats.imavgs.bgb=vmistats.imavgs.bgb ./ detgain;
vmistats.imavgs.imgp=vmistats.imavgs.imgp ./ detgain;
vmistats.imavgs.bgbp=vmistats.imavgs.bgbp ./ detgain;
vmistats.imavgs.imge=vmistats.imavgs.imge ./ detgain;
vmistats.imavgs.bgbe=vmistats.imavgs.bgbe ./ detgain;
%%%

if exist('verbosity','var'),
    %figure; surf((vmistats.imavgs.img-vmistats.imavgs.bgb)./detgain);shading interp
    
    tmp1=vmistats.imavgs.img-vmistats.imavgs.bgb;
    tmp2=tmp1./(detgain);
    figure('position',[15 560 1800 420]);
    subplot(1,3,1); imagesc(detgain);title('gainprofile');% surf(detgain);shading interp; axis tight; view([0 90]);
    subplot(1,3,2); imagesc(tmp1);title('avg(img-bgb)'); %surf(tmp1);shading interp; axis tight; view([0 90]);
    subplot(1,3,3); imagesc(tmp2);title('avg(img-bgb)/gainprofile'); %surf(tmp2);shading interp; axis tight; view([0 90]);
end