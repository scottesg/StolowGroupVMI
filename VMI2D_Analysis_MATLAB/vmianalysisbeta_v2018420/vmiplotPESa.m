function vmistats=vmiplotPESa(vmistats)

angle=vmistats.angle;
y0=vmistats.vmicentre(1);
x0=vmistats.vmicentre(2);
[szy szx]=size(zeros(vmistats.imsize));
halfcrop=min([x0-1,szx-x0,y0-1,szy-y0]);

list={'vmistats.imavgs.img-vmistats.imavgs.bgb'}; listdesc={'pp'};
if sum(sum(vmistats.imavgs.imgp)), list=[list,'vmistats.imavgs.imgp-vmistats.imavgs.bgbp']; listdesc=[listdesc,'po']; end;
if sum(sum(vmistats.imavgs.imge)), list=[list,'vmistats.imavgs.imge-vmistats.imavgs.bgbe']; listdesc=[listdesc,'eo']; end;    
for i=1:length(list),
    imin=eval(list{i});
    imin=imin(y0-halfcrop+1:y0+halfcrop,x0-halfcrop+1:x0+halfcrop);
    imin=imrotate(imin,angle,'bilinear','crop');
    imin=smoothimg(imin,2);
    [Ir{i},r{i},imout]=iabel2D_ab(imin,'inv');
end
Ir1=cell2mat(Ir);
Ir1=reshape(Ir1,size(Ir{1},2),[]);
r1=cell2mat(r);
r1=reshape(r1,size(r{1},2),[]);

figure; plot(r1,Ir1); legend(listdesc); xlabel('r, pix');

vmistats.spectra.Ir=Ir1;
vmistats.spectra.r=r1;
vmistats.spectra.desc=listdesc;


if isfield(vmistats,{'ke','K'}),
    KEs=vmistats.ke;
    K=vmistats.K;
    bins=sqrt(KEs/K);
    for i=1:length(r),
        Ir1_ke(:,i) = rebin3(r{i},Ir{i},bins);
    end;
    figure; plot(KEs,Ir1_ke); legend(listdesc); xlabel('eKE, eV');
    
    vmistats.spectra.Ir_ke=Ir1_ke;
    vmistats.spectra.ke=KEs;
end;