function savedto = vmiexport(vmistats,folderout,format)

if ~exist('format','var'), format='bin'; end;
if ~isdir(folderout), mkdir(folderout);end;

delaysN=length(vmistats.imstks.difstk);

switch format
    case 'bin'
        folder=[folderout '\' vmistats.subfolder '\'];if ~isdir(folder), mkdir(folder);end;
        for j=1:delaysN,
            fname=['test._' num2str(vmistats.delays(j)) 'ps.difave.bin'];
            fid=fopen([folder fname],'wb');
            fwrite(fid,single(vmistats.imstks.difstk{j}),'float32');
            fclose(fid);
            dlmwrite([folderout '\delays.dat'],vmistats.delays); %legacy support
        end;
    case 'difstk'
        fname=[folderout '\' vmistats.subfolder];
        fid=fopen([fname '.difstk'],'wb'); 
        fwrite(fid,int32(delaysN),'int32'); %delays dimension
        fwrite(fid,int32(size(vmistats.imstks.difstk{1},1)),'int32'); %pix size v
        fwrite(fid,int32(size(vmistats.imstks.difstk{1},2)),'int32'); %pix size h
        for j=1:delaysN,
            fwrite(fid,single(vmistats.imstks.difstk{j}),'float32');
        end;
        fclose(fid);
        dlmwrite([fname '.delays'],vmistats.delays);
        dlmwrite([folderout '\delays.dat'],vmistats.delays); %legacy support
    case 'idifstk'
        %disp(['format ' format ' is not yet supported']);
    case 'trpes'
    case 'betas'
    otherwise
        disp(['format ' format ' is not yet supported']);
end;

savedto=[];
return;

%000106-Apr-2016_143906._-0.5ps.difave.bin