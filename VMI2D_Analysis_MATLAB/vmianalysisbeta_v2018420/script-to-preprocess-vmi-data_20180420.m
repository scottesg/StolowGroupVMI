% VMI data preprocessing toolbox worksheet created to facilitate handling 
% of the data taken on the VMI machine in Femto 3-4 lab at NRC,Ottawa
% AB 2016/July-Aug
% See the toolbox description in separate file: 
% "vmi analysis toolbox functions list.txt"

%ver 0.4 (20170316)

%(changes since v0.3: 
%       +(on Kevin's request): support of correction for 5th harmonic decay
%       and additional bgnd  subtraction for abel-inverted stk. (Use before
%       fitting beta-parameters) 
%       +(on Kevin's request): few tweaks to support loading and displaying
%       a single-delay data  
%)


%%
clear all; close all;
%cd 'e:\vmisoft\vmianalysisbeta20160809'
cd 'C:\Users\Scott\Python\VMI'
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%% preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%folderin=uigetdir('C:\vmidata\rawdata\','pick the folder with raw data') 
%subfolder='20180417\03-xenon-(t0 scan)';
%subfolder='20180417\05-butadiene(same t0 as in scan03)';
subfolder='data\20241010\20241010_cs2test3';

scans=-1; %this means 'load all the scans'
%scans='[1:6,8:18,20:end]';
%scans=[1:4];
folderin=['C:\Users\Scott\Python\VMI\' subfolder]; if ~isdir(folderin), disp(['Folder ''' folderin ''' not found.']);return;end;
folderout=['C:\Users\Scott\Python\VMI\reduceddataml\' subfolder];if ~isdir(folderout), mkdir(folderout);end;

%read-in the raw data, create the stack
vmistats = vmigetstats(folderin, scans);

%%% plot the traces to decide which scans are good
vmiplottraces4(vmistats); 

%%% if the auto-centering algorithm does not seem to do a good job => fix the center values manually
%vmistats.vmicentre=[233,321]; 
%vmistats.vmicentre=[224,310];   

%%% plot some images to judge what BGND is worth subtracting
vmiplotaverages(vmistats);


%% subtract BG, correct for gain nonuniformity, crop, rotate, export the resulting .difstk

%vmistats=vmiget160decayrate(vmistats,drange);

%%% subract background (.stk(s) -> .difstk)
%'peb' is a binary string defining what background needs to be subtracted (pumP,probE,Beam) 
%peb='000'; %no bg subtraction
%peb='001'; %subtract 'nobeam'
%peb='100'; %subtract 'pumponly'
%peb='101'; %subtract 'pumponly' and 'nobeam'
%peb='010'; %subtract 'probeonly'
%peb='011'; %subtract 'probeonly' and 'nobeam'
%peb='111'; %subtract 'pumponly','probeonly' and 'nobeam'

peb='001';
vmistats=vmibgsubtr(vmistats,peb);


%%% correct for detector gain nonuniformity
%detgainprofile=['C:\vmidata\reduceddata\08-Nov-2016\20161107_detector_response\detgain.mat'];
%detgainprofile=['C:\vmidata\reduceddata\08-Nov-2016\20161107_detector_response\detgain.png'];
%detgainprofile=['C:\vmisoft\analysis\detgain20161108.mat'];
%vmistats=vmidetgaincorr(vmistats,detgainprofile);%,'');


%%% crop the .difstk images
%vmistats.vmicentre=[254,322];    %hardwired centre when needed:
%vmistats=vmicropdifstk(vmistats,[450 450]); % keep the size EVEN NUMBERs
vmistats=vmicropdifstk(vmistats);

%

%%% rotate the .difstk images 
vmistats=vmirotate(vmistats,70);%0); %the exact angle will be automatically searched for in the proximity of the "suggested' angle [deg]
%vmistats=vmirotate(vmistats,100,'exact'); %the exact angle will be automatically searched for in the proximity of the "suggested' angle [deg]
%vmirotate(vmistats,100,'auto','preview'); %use this syntax to preview if the angle is right [deg] 


%%% and export to .difstk or .bin format for further analysis in Doug's GUI
% %exportedto=vmiexport(vmistats,folderout,'bin');
% exportedto=vmiexport(vmistats,folderout,'difstk');
% if ~isempty(exportedto), disp(exportedto);end;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% inverse Abel using Ben's code. Extract TR-PES. 

vmistats=vmiiabel_stk(vmistats);


%% plot TR-PES (r[pix],t[ps])

Ir=vmistats.Ir;
delays=-vmistats.delays; 
r=vmistats.r;

%%% correct for decay of 5th harmonic (assumed dropping linearly)
%R is a multiplier that has to be applied to a last delay in scan compare
%to first delay of the same scan to compensate for the decrease of the 5th
%harmonic intensity (assuming 1-photon process) 
% R=vmiget160decayrate(vmistats,[]);
% %R=vmiget160decayrate(vmistats,[-0.5,0.15]);
% Ir=Ir.*(linspace(1,R,length(delays))'*ones(size(r)));

%%% remove time-independent background as defined by trpes between delay1 and delay2 [ps]
%Ir=flattenbg(Ir,-1,-0.2,delays);

%%% smooth 
%Ir=smoothimg(Ir,1);

%plot
figure; 
if length(delays)>1,
    surf(r,delays,Ir); shading interp; axis tight; view([0 90]); lighting phong;
    ylabel('delays, [ps]');
    title('TRPES');
    %camlight
else
    plot(r,Ir);
end
xlabel('ring radius, [pix]');

%%% export to .CSV (compatible with PESview)
% fout=[vmistats.subfolder '.trpes.r[pix].csv'];
% csvwrite([folderout '\' fout],[2 r; delays Ir]);


%% TR-PES (r[pix],t[ps]): plot time evolutions of few energy-slices 

rbinsa=[1 30; 31 100; 150 200]; %[pix]
r=vmistats.r;

irbinsa=geti(rbinsa,r);
Ir_rbinsa=[];legends={};
for i=1:size(rbinsa,1),
    Ir_rbinsa=[Ir_rbinsa, sum(Ir(:,irbinsa(i,1):irbinsa(i,2)),2)/(irbinsa(i,2)-irbinsa(i,1)+1)];
    legends{i}=['r[pix]= ' num2str(rbinsa(i,1)) '..' num2str(rbinsa(i,2))];
end

%%% renormalize
%Ir_rbinsa=Ir_rbinsa./(ones(size(Ir_rbinsa,1),1)*sum(Ir_rbinsa,1)); %on integral
%Ir_rbinsa=Ir_rbinsa./(ones(size(Ir_rbinsa,1),1)*max(Ir_rbinsa));  %on max
Ir_rbinsa=Ir_rbinsa./(ones(size(Ir_rbinsa,1),1)*(max(Ir_rbinsa)-min(Ir_rbinsa)));  %on max-min

%%% plot
figure; plot(delays,Ir_rbinsa);
xlabel('delays, [ps]');
title('Evolution of trpes slices with p-p delay');
legend(legends);


%% TR-PES (r[pix],t[ps]): plot PES at few pp-delays 

dbinsa=[-0.1 0.1; -0.5 -0.2; -2.0 -0.8]; %[ps]
r=vmistats.r;

idbinsa=geti(dbinsa,delays); idbinsa=sort(idbinsa,2);
Ir_dbinsa=[];legends={};
for i=1:size(rbinsa,1),
    Ir_dbinsa=[Ir_dbinsa; sum(Ir(idbinsa(i,1):idbinsa(i,2),:),1)/(idbinsa(i,2)-idbinsa(i,1)+1)];
    legends{i}=['[' num2str(dbinsa(i,1)) '..' num2str(dbinsa(i,2)) ']ps'];
end

%%% renormalize
%Ir_dbinsa=Ir_dbinsa./(sum(Ir_dbinsa,2)*ones(1,size(Ir_dbinsa,2))); %on integral
%Ir_dbinsa=Ir_dbinsa./(max(Ir_dbinsa,[],2)*ones(1,size(Ir_dbinsa,2)));  %on max
%Ir_dbinsa=Ir_dbinsa./((max(Ir_dbinsa,[],2)-min(Ir_dbinsa,[],2))*ones(1,size(Ir_dbinsa,2)));  %on max-min

%%% plot
figure; plot(r,Ir_dbinsa');
xlabel('r, [pix]');
title('Evolution of trpes slices with p-p delay');
legend(legends);


%% %export4magda
% %create *processed.mat file
% vmiexport4magda(vmistats,[folderout '\' vmistats.subfolder]);
% 
% %calibration file
% K = 1.3528e-004; %for 5eV  % K=KE/(R)^2
% fout=[vmistats.subfolder '.repeits.cal'];
% fid=fopen([folderout '\' fout],'wt');
% fprintf(fid,'A=%g\nE0=0\nt0=1\n',1/2/K);
% fclose(fid);


%% plot TR-PES (ke[eV],t[ps])

Ir=vmistats.Ir;
r=vmistats.r;
delays=-vmistats.delays;
delaysN=length(delays);
ke2r = @(ke,K) round(sqrt(ke/K)); 
r2ke = @(r,K) ((r.^2)*K);

%%%%get the calibration constant 
%(a)from a file
%K=vmical('e:\vmisoft\analysis\test2.vmical.txt')
%K=vmical('20180417.xe.vmical.txt')
%or
%(b)define from UR voltage
%UR=521;%1022; %[Volt] 
%K=UR*1e-3/(62)^2;   %Varun&Kevin's trick to calculate the calibration constant from UR (=repeller voltage)
%or
%(c)define K it explicitely
%K=2.0661e-004; %a made-up number so that 10eV electrons will make R=220pixels: K=KE[eV]/(Radius[pix])^2
%K=1.1284e-004; 
%K=0.87/(112.4)^2
%K=9.5367e-005;
K=8.046433567210871e-05

%%%rebin into equidistant KE bins
KEs=linspace(0,1.12,101); %KE bins in [eV]: "from", "to", N_of_bins"
bins=sqrt(KEs/K);
Ir_ke = rebin3(r,Ir,bins);

%%
%%%remove time-independent background as defined by trpes between delay1 and delay2 [ps]
%Ir_ke=flattenbg(Ir_ke,-0.5,-0.3,delays);


%%%plot
figure;
if length(delays)>1,
    surf(KEs,delays(1:end),Ir_ke(1:end,:)); shading interp; axis tight; view([0 90]); %lighting phong; %camlight
    %figure; surf(KEs,1:delaysN,Ir_ke(1:end,:)); shading interp; axis tight; view([0 90]); lighting phong;
    ylabel('delays, [ps]');
    title('TRPES');
else
    plot(KEs,Ir_ke);
end;
xlabel('eKE, [eV]');

%%%store 
vmistats.Ir_ke=Ir_ke;
vmistats.ke=KEs;
vmistats.K=K;

%%export to .CSV (compatible with PESview)
fout=[vmistats.subfolder '.trpes.ke.csv'];
csvwrite([folderout '\' fout],[0 KEs; delays Ir_ke]);

%% TR-PES (ke[eV],t[ps]): plot time evolutions of few energy-slices 
kebinsa=[0 1; 2 3; 4 5]; %[eV]
K=K;

rbinsa=floor(ke2r(kebinsa,K)); %K = vmi calibration constant
irbinsa=geti(rbinsa,r);

Ir_rbinsa=[];legends={};
for i=1:size(rbinsa,1),
    Ir_rbinsa=[Ir_rbinsa, sum(Ir(:,irbinsa(i,1):irbinsa(i,2)),2)];
    %legends{i}=['ke[eV]= ' num2str(kebinsa(i,1)) '..' num2str(kebinsa(i,2))];
    legends{i}=['ke[eV]= ' num2str(r2ke(rbinsa(i,1),K),2) '..' num2str(r2ke(rbinsa(i,2),K),2)]; %actual range
end

%%%renormalize
%Ir_rbinsa=Ir_rbinsa./(ones(size(Ir_rbinsa,1),1)*sum(Ir_rbinsa,1)); %on integral
%Ir_rbinsa=Ir_rbinsa./(ones(size(Ir_rbinsa,1),1)*max(Ir_rbinsa));  %on max
Ir_rbinsa=Ir_rbinsa./(ones(size(Ir_rbinsa,1),1)*(max(Ir_rbinsa)-min(Ir_rbinsa)));  %on max-min

%%%plot
figure; plot(delays,Ir_rbinsa);
xlabel('delays, [ps]');
title('Evolution of trpes slices with p-p delay');
legend(legends);



%% TR-PES (ke[eV],t[ps]): plot PES at few pp-delays 

%dbinsa=[0.50 0.60; 0.70 0.90; 1.00 1.20]; %[ps]
dbinsa=[-0.1 0.1; -0.5 -0.3; 0.2 1.0]; %[ps]

idbinsa=geti(dbinsa,delays); idbinsa=sort(idbinsa,2);
Ir_ke_dbinsa=[];legends={};
for i=1:size(rbinsa,1),
    Ir_ke_dbinsa=[Ir_ke_dbinsa; sum(Ir_ke(idbinsa(i,1):idbinsa(i,2),:),1)/(idbinsa(i,2)-idbinsa(i,1)+1)];
    legends{i}=['[' num2str(dbinsa(i,1)) '..' num2str(dbinsa(i,2)) ']ps'];
end

%%%renormalize if needed
%Ir_ke_dbinsa=Ir_ke_dbinsa./(sum(Ir_ke_dbinsa,2)*ones(1,size(Ir_ke_dbinsa,2))); %on integral
%Ir_ke_dbinsa=Ir_ke_dbinsa./(max(Ir_ke_dbinsa,[],2)*ones(1,size(Ir_ke_dbinsa,2)));  %on max
%Ir_ke_dbinsa=Ir_ke_dbinsa./((max(Ir_ke_dbinsa,[],2)-min(Ir_ke_dbinsa,[],2))*ones(1,size(Ir_ke_dbinsa,2)));  %on max-min

%%%plot
figure('name','Evolution of trpes slices with p-p delay'); 
plot(KEs,Ir_ke_dbinsa');
xlabel('ke, [eV]');
title('Evolution of trpes slices with p-p delay');
legend(legends);


%% before fitting beta-parameters to the abel-inverted images 
%  it is necessary to remove pump-only and probe-only background.
%  If enough pump-only and probe-only data was collected - the subtraction
%  can be done already at the vmibgsubtr(vmistats,peb) step. If not - then
%  we need to use the pump-probe data at 'negative delays' when it is
%  assumed that all the p-p signal is faded and what's left is only due to
%  pump-only + probe-only  
%  (note that first, one needs to correct for decaying 5th harmonics. This
%  is taken care of by supplying R determined by vmiget160decayrate() )

%vmistats.imstks.idifstk_bgfree=vmistats.imstks.idifstk; 
vmistats.imstks.idifstk_bgfree = flattenbg_stk(vmistats.imstks.idifstk,-1,-0.3,vmistats.delays);  %subtract background, but no correction for 5th harmonics yield decay
%vmistats.imstks.idifstk_bgfree = flattenbg_stk(vmistats.imstks.idifstk,-1,-0.2,vmistats.delays,1);
%vmistats.imstks.idifstk_bgfree = flattenbg_stk(vmistats.imstks.idifstk,-1,-0.2,vmistats.delays,R); %correct for decay of 5th harmonics yield, then subtract background



%% fit beta-parameters and export them: one KE_ROI_slice for all delays

stk=vmistats.imstks.idifstk_bgfree; %stack of Abel inverted image, created by "iabel2D_ab"
delays=-vmistats.delays;
delaysN=length(delays);

%%%specify range of KEs for the beta param fits
kerange=[0 5]; %[eV]
rrange=ke2r(kerange,K); %K = vmi calibration constant
rmax=vmistats.difstk_imsize(1)/2;
rrange(rrange>rmax)=rmax;
kerange=r2ke(rrange,K);
%%%or give the range of radii [pix] directly 
%rrange=[135,145];
%rrange=[85,105];

%%%fit
beta = vmifitbeta_stk(stk,rrange);

%%%plot
figure('name',['fit of betas for ke=[' num2str(kerange(1),3) '..' num2str(kerange(2),3) ']eV' ' (r=[' num2str(rrange(1)) '..' num2str(rrange(2)) ']pix)']); 
subplot(1,2,1); 
plot(delays,beta);
xlabel('delay, ps');
%plot(1:delaysN,beta);
%xlabel('delay point #');
legend({'b_0','b_2','b_4','b_6'});

beta2=beta./(beta(:,1)*ones(1,size(beta,2)));
beta2( (beta(:,1)<0.05*max(beta(:,1))) ,:)=NaN;
subplot(1,2,2); 
plot(delays,beta2);
xlabel('delay, ps');
%plot(1:delaysN,beta2);
%xlabel('delay point #');
legend({'\beta_0_0/\beta_0_0','\beta_2_0/\beta_0_0','\beta_4_0/\beta_0_0','\beta_6_0/\beta_0_0'});
%title(['[' num2str(kerange(1)) '-' num2str(kerange(2)) ']eV' ' ([' num2str(rrange(1)) '-' num2str(rrange(2)) ']pix)']);

%%%export
fout=[vmistats.subfolder '.betas(r=[' num2str(rrange(1)) '-' num2str(rrange(end)) ']pix).csv'];
if ~isdir(folderout), mkdir(folderout);end;
%csvwrite([folderout '\' fout],[delays beta]);

%%%save into the vmistats structure
vmistats.betas=beta;
vmistats.betasnorm=beta2;



%% fit beta-parameters and export them: all KE bins at around t0

stk=vmistats.imstks.idifstk_bgfree; %stack of Abel inverted image, created by "iabel2D_ab"
delays=vmistats.delays;
delaysN=length(delays);

%%%"t0" range of delays
trange=[-0.1,0.1];   %"t0" range of delays
%itrange=threshold1Darray_v(trange,delays);
itrange=geti(trange,delays); itrange=sort(itrange);

%%%ke bins
%ke=linspace(0,5,50); %ke bins
%OR, use the same KE bins as above then
ke=KEs;

%however, shouldn't fit betas for radii>rmax aka biggest circle to fully fit the ccd chip
rmax=vmistats.difstk_imsize(1)/2;
kemax=r2ke(rmax,K);
ke=ke(ke<=kemax);

rbins=ke2r(ke,K); %aka: rbins=floor(sqrt(KEs/K));

%%%average over "trange of delays before fitting
imtmp=zeros(size(stk{1})); for it=itrange(1):itrange(2),imtmp=imtmp+stk{it}; end; %imtmp={imtmp};%imtmp=mat2cell(imtmp);

%%%fit
beta = vmifitbeta_stk(imtmp,rbins);

%%%ke's bin's centres
kei=ke(1:end-1)+diff(ke)/2;

%%%plot
figure; 
subplot(1,2,1); 
%plot(ke(1:end-1),beta);
plot(kei,beta);
xlabel('KE, eV');
legend({'\beta_0_0','\beta_2_0','\beta_4_0','\beta_6_0'});
%normalize on beta00
beta2=beta./(beta(:,1)*ones(1,size(beta,2))); 
beta2( (beta(:,1)<0.01*max(beta(:,1))),:)=NaN;
subplot(1,2,2); 
%plot(ke(1:end-1),beta2);
plot(kei,beta2);
xlabel('KE, eV');
legend({'\beta_0_0/\beta_0_0','\beta_2_0/\beta_0_0','\beta_4_0/\beta_0_0','\beta_6_0/\beta_0_0'});

%%%export
fout=[vmistats.subfolder '.betas(t=[' num2str(trange(1)) ',' num2str(trange(2)) ']ps).csv'];
% if ~isdir(folderout), mkdir(folderout);end;
% csvwrite([folderout '\' fout],[ke(1:end-1)' beta]);

% %test load&plot the exported betas
% tmp=csvread([folderout '\' fout]);
% figure;plot(tmp(:,1),tmp(:,2:end));
% xlabel('KE, eV');
% legend({'\beta_0_0','\beta_2_0','\beta_4_0','\beta_6_0'});


%% fit beta to whole [delays,kes] surface and plot 3 normalized surfaces (beta20/beta00, beta40/beta00, beta60/beta00)

stk=vmistats.imstks.idifstk_bgfree; %stack of Abel inverted image, created by "iabel2D_ab"

%define ranges to fit
ke=linspace(0,6,41); %ke bins
%OR, use the same KE bins as above then
%ke=KEs;
%ke=ke(2:end); %not interested in zeke els
%
%however, shouldn't attempt fitting betas for radii>rmax aka biggest circle to fully fit the ccd chip
rmax=vmistats.difstk_imsize(1)/2;
kemax=r2ke(rmax,K);
ke=ke(ke<=kemax);
%
rbins=ke2r(ke,K); %aka: rbins=floor(sqrt(KEs/K));

%range of delays to be fitted
delays=-vmistats.delays;
trange=[-.5 1];
itrange=geti(trange,delays); itrange=sort(itrange);
itvector=itrange(1):itrange(2);
tvector=delays(itvector);

%%%fit
beta = vmifitbeta_stk(stk,rbins);

%%%plot
b0=squeeze(beta(:,:,1));
offmask=(b0<0.01*max(max(b0)));
beta20=squeeze(beta(:,:,2))./b0;  beta20( offmask )=NaN; 
beta40=squeeze(beta(:,:,3))./b0;  beta40( offmask )=NaN; 
beta60=squeeze(beta(:,:,4))./b0;  beta60( offmask )=NaN; 

%%%ke's bin's centres
kei=ke(1:end-1)+diff(ke)/2;
%kei=ke(1:end-1);

figure('name','beta surfaces'); 
subplot(2,2,1); 
surf(kei,tvector,b0(itvector,:)); shading interp; axis tight; lighting phong; colorbar; title('b_0'); view([0 90]); %ylim([-1,1])
subplot(2,2,2); 
surf(kei,tvector,beta20(itvector,:)); shading interp; axis tight;  lighting phong; colorbar; title('\beta_2_0/\beta_0_0'); view([0 90]);caxis([-1,2]);
subplot(2,2,3); 
surf(kei,tvector,beta40(itvector,:)); shading interp; axis tight;  lighting phong; colorbar; title('\beta_4_0/\beta_0_0'); view([0 90]);
subplot(2,2,4); 
surf(kei,tvector,beta60(itvector,:)); shading interp; axis tight;  lighting phong; colorbar; title('\beta_6_0/\beta_0_0'); view([0 90]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot 1D plots of the average spectra 

vmistats=vmiplotPESa(vmistats);


%% convert image into polar coordinates for shits and giggles

in_Img=vmistats.imavgs.img-vmistats.imavgs.bgb-(vmistats.imavgs.imgp-vmistats.imavgs.bgbp);%imin;

x0 = vmistats.vmicentre(2);%450/2;%118;
y0 = vmistats.vmicentre(1);%450/2;%154;
r = 0:.5:450/2;%150;
%phi = [0:.25:360]*pi/180;
phi=[0:0.5:360]*pi/180;
[R,Phi] = meshgrid(r,phi);
polImg = img2pol_bjorn(in_Img,x0,y0,R,Phi,'linear');
figure;
% subplot(1,2,1)
% surf(x0+R.*cos(Phi),y0+R.*sin(Phi),polImg),shading flat
% subplot(1,2,2)
%imagesc(polImg)
pcolor(r,phi/pi*180,polImg); shading flat;


%% %% inverse Abel of a single image using Ben's code.
% 
% imin=vmistats.imavgs.difstk;
% %imin=rot90(imin,1);
% imin=smoothimg(imin,2);
% [Ir,r,imout]=iabel2D_ab(imin,'inv'); 
% figure; subplot(1,2,1); imagesc(imin); axis image; title('imavgs.difstk');
%         subplot(1,2,2); imagesc(imout); axis image; title('iabel(imavgs.difstk)');
% figure; plot(r,Ir); 
% %figure; imagesc(imout); axis image;
% 
% %%
% K=vmical('e:\vmidata\rawdata\452V.txt')
% %K=2.0661e-004; %a made-up number so that 10eV electrons will make R=220pixels
% %UR=520; %[Volt] 
% %K=UR*1e-3/(62)^2;   %Varun&Kevin's trick to calculate the calibration constant from UR (=repeller voltage)
% %
% KEs=linspace(0,5,75);
% bins=sqrt(KEs/K);
% Ir_ke = rebin3(r,Ir,bins);
% %Ir_ke=flattenbg(Ir_ke,-0.5,-0.1,delays);
% figure; plot(KEs,Ir_ke(1:end,:));
% %%
% %I449=Ir_ke;
% %I452=Ir_ke;
% %I460=Ir_ke;
% %figure; plot([I449; I452; I460]');legend({'449V','452V','460V'});title('UE (@UR=520V, UR=0V, UR1=520V)');

