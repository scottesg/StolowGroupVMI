function vmistats=vmiiabel_stk(vmistats)
%   vmistats=vmiiabel_stk(vmistats)
%   wrapper function to conveniently iabel the whole stack of images
%   20160809 ab
%   requires: preprocessed data in vmstats (.difstk{delaysN}(:,:))   
%   produces: .idifstk, .Ir, .r
%   requires functions: "iabel2D_ab.m", "smoothimg.m"

delays=vmistats.delays;
delaysN=length(vmistats.delays);
Ir=zeros(delaysN,vmistats.difstk_imsize(1)/2); % half as many points as in the noninverted image
imout=cell(delaysN,1);
for j=1:delaysN,
    imin=vmistats.imstks.difstk{j};
    imin=smoothimg(imin,2);
    %tic
    [Ir(j,:),r,imout{j}]=iabel2D_ab(imin,'inv'); 
    %toc
    disp([num2str(j) '    ' num2str(delays(j)) 'ps      '  num2str(sum(sum(imin))) '     ' num2str(sum(Ir(j,:)))]); 
end
vmistats.imstks.idifstk=imout; 
vmistats.Ir=Ir;
vmistats.r=r;
delays=delays;
