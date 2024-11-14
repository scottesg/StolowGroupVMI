function vmistats=vmirotate(vmistats,angle,mode,target)
%vmistats=vmirotatedifstk(vmistats,angle[,mode[,target]])
%rotate the .difstk images
% angle [deg]
% mode='auto','exact'
% target='difstk','preview', or image

%ab20161122

if ~exist('mode','var')||isempty(mode), mode='auto'; end;
if ~exist('target','var'), target='difstk'; end;
if ~exist('angle','var'),
    if isfield(vmistats,'angle'), angle=vmistats.angle;
    else disp('Approx angle is required. Exiting.'); return; end
end
    
switch mode
    case {'exact','fix','fixed'}
        angle=angle;
    case 'auto'
        n=vmistats.difstk_imsize(1);
        imin=vmistats.imavgs.difstk+rot90(vmistats.imavgs.difstk,2);
        %angle=110;
        arange=[angle-30:0.5:angle+30];
        %ROI=annroimask([450 450],225, 225, 0,150);
        ROI=annroimask(vmistats.difstk_imsize,vmistats.difstk_imsize(1)/2, vmistats.difstk_imsize(2)/2, 0,150);
        for iangle=1:length(arange),
            im=imrotate(imin,arange(iangle),'bilinear','crop');
            im=im.*ROI;
            %subplot(1,2,1); imagesc(im); axis image; title(num2str(arange(iangle)));
            d3(iangle)=sum(sum((im(:,1:n/2)-im(:,end:-1:(n/2+1))).^2));
        end;
        %figure; plot(arange,d3)
        
        iangle=find(d3==min(d3));
        angle=arange(iangle(1));
        disp(['Image(s) will be rotated by ' num2str(angle) 'deg']);
    otherwise
        disp('whaat?');
end;


%if ischar(target),
switch target
    case 'difstk'
        delaysN=length(vmistats.imstks.difstk);
        for j=1:delaysN, vmistats.imstks.difstk{j}=imrotate(vmistats.imstks.difstk{j},angle,'bilinear','crop'); end;
        vmistats.imavgs.difstk=imrotate(vmistats.imavgs.difstk,angle,'bilinear','crop');
        vmistats.angle=angle;

    case 'preview'
        if isempty(vmistats.imstks.difstk), disp('no difstk been defined yet. run "vmibgsubtr" and "vmicropdifstk" first'); return; end;
        in=vmistats.imavgs.difstk;
        out=imrotate(in,angle,'bilinear','crop');
        figure('name',['preview: roatating ' num2str(angle) 'deg']); 
        subplot(1,2,1); imagesc(in); axis image; title('before rotating');
        subplot(1,2,2); imagesc(out); axis image; title(['after rotating ' num2str(angle) 'deg']);
        hold on; 
        line([vmistats.difstk_imsize(1)/2-1 vmistats.difstk_imsize(1)/2-1],[1 vmistats.difstk_imsize(2)],'linestyle',':','color','r');
        line([vmistats.difstk_imsize(1)/2+2 vmistats.difstk_imsize(1)/2+2],[1 vmistats.difstk_imsize(2)],'linestyle',':','color','r');
        line([1 vmistats.difstk_imsize(1)],[vmistats.difstk_imsize(2)/2-1 vmistats.difstk_imsize(2)/2-1],'linestyle',':','color','r'); 
        line([1 vmistats.difstk_imsize(1)],[vmistats.difstk_imsize(2)/2+2 vmistats.difstk_imsize(2)/2+2],'linestyle',':','color','r'); 
        hold off;
end;
%else 
%    imout=imrotate(imin,angle,'bilinear','crop');
    
