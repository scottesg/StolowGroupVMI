function stats=vmigetstats(folder,scans) %[traces,centre,imaverage]

flist=dir([folder '\*.png']);
if isempty(flist), error(['No .png files were found in "' folder '"']); return;end;
%[tmp,idx]=sort([flist.datenum]); %sort by date
[tmp,idx]=sort(char(flist.name)); %sort by name
%%[tmp,idx]=sort_nat(cellstr(char(flist.name)));
%flist(idx(i)).name

%-----------------------------------------------------------------------
%BUILD A "DATABASE" OF USEABLE FILES IN THE FOLDER ONE LINE PER MEASUREMENT: DELAY, PEB, PASS, FNAME
disp('Building a "database" of useable files in the folder one line per measurement: delays, peb, pass, fname');
tic
%-----------------
%the first pass - to get the range of delays used in the dataset. (Not elegant but makes scan# determination more robust. For example the code will still work correctly when some delays are missing in the first scan.)
delays=[]; %delay_ps=[]; 
for i=1:length(flist)
    fname=flist(idx(i)).name;
%     switch fname(end-7:end-4)
%         case '.img'
%             peb=bin2dec('111'); %7
%         case {'.bgb','.bgr'}
%             peb=bin2dec('110'); %6
%         case 'imgp'
%             peb=bin2dec('101'); %5
%         case 'bgbp'
%             peb=bin2dec('100'); %4
%         case 'imge'
%             peb=bin2dec('011'); %3
%         case 'bgbe'
%             peb=bin2dec('010'); %2
%         otherwise
%             continue;
%     end
    delay_ps=str2delay(fname); if isempty(delay_ps), continue; end;
    if isempty(find(delays == delay_ps,1)), delays=[delays; delay_ps]; end;
end
delaysN=length(delays);
maxdelay=max(delays); mindelay=min(delays);
if delaysN>1,
    if delays(1)>delays(2),delays=sort(delays,'descend');else delays=sort(delays);end;
end;
%-----------------
%the second pass - collecting the rest info
clear db 
scansN=1; j=0; %delaysN=0; %tmp2=[];
delay_ps=NaN;
for i=1:length(flist)
    %fname=flist(i).name;
    fname=flist(idx(i)).name;
    switch fname(end-7:end-4)
        case '.img'
            peb=bin2dec('111'); %7
        case {'.bgb','.bgr'}
            peb=bin2dec('110'); %6
            tmp=delay_ps;
            %if isempty(tmp2), tmp2=delay_ps; end;
        case 'imgp'
            peb=bin2dec('101'); %5
        case 'bgbp'
            peb=bin2dec('100'); %4
        case 'imge'
            peb=bin2dec('011'); %3
        case 'bgbe'
            peb=bin2dec('010'); %2
        otherwise
            continue;
    end
    delay_ps=str2delay(fname); if isempty(delay_ps), continue; end;
    %if isempty(find(delays == delay_ps,1)), delays=[delays; delay_ps]; delaysN=delaysN+1; maxdelay=max(delays); mindelay=min(delays);
    %%elseif delay_ps<tmp, scansN=scansN+1; end; %note: this assumes accending delay scan
    %elseif (peb==6),
    if (peb==6),
        if (delay_ps==tmp) || ... %alternating directions
                ((delay_ps==maxdelay)||(delay_ps==mindelay)) && ((tmp==maxdelay)||(tmp==mindelay)), %non-alternating descending or ascending
            scansN=scansN+1;
        end;
        %if (delay_ps==tmp2), scansN=scansN+1; end; %this will work only for unidirectional scans (both accending and descending)
    end;
    j=j+1;
    db(j).idx=j;
    db(j).del=delay_ps;
    db(j).peb=peb;
    db(j).scan=scansN;
    db(j).fname=fname;
end
%char(db(1:2).fname)
%db(:).pass
disp('Done.'); toc
%---------------------------------------------------------------------


%----------------------------------------------------------------------
%PARSING THE SCANS MASK
disp('Parsing the scans mask..');

if isempty(scans), scans=[1:scansN];
elseif ischar(scans), 
    str=scans;
    %str=strrep(str,'-',':');
    str=strrep(str,';',',');
    str=strrep(str,'end',num2str(scansN));
    if str(1)~='[', str=['[' str]; end;
    if str(end)~=']',str=[str ']']; end;
    scans=eval(str);
    if scans(end)>scansN, 
        scanstmp=scans; scans=[]; 
        for i=1:length(scanstmp), 
            if scanstmp(i)<=scansN, scans(i)=scanstmp(i); end;
        end;
    end;
    scansN=length(scans);
    %array2range(scans)
elseif isnumeric(scans)
    if find(scans(scans<0)), scans=[1:scansN]; end;
end;

scans
disp('Done.'); toc
%----------------------------------------------------------------------


%----------------------------------------------------------------------
%LOADING FILES AND EXTRACTING TRACES
disp('Extracting traces...');

% %scans to be loaded
% scans=[1:scansN];  %all
% %scans=[3,20:21];  %a subset of scans
% %scans=[1:9,11:21];
%scans=scans( (scans>0)&(scans<=scansN));
masks.scans=zeros(size([db.scan])); for i=1:length(scans), masks.scans = masks.scans | ([db.scan]==scans(i)); end;
% masks.img=([db(:).peb]==7);
% masks.imgp=([db(:).peb]==5);
% masks.bgbp=([db(:).peb]==4);
% masks.all=ones(1,length(db)); %=all

%which mask is to be used
mask=masks.scans;
%mask=masks.scans & (masks.imgp | masks.bgbp);
%mask=masks.imgp; 
%mask=masks.all & masks.scans & (~masks.imgp);
%mask=masks.all;

clear traces
traces=cell(7,1);
for i=1:7, traces{i}=[];end; %111,110,101,100,011,010, (001),(000)

img_stk=cell(delaysN,1);  bgb_stk=cell(delaysN,1); 
% imgp_stk=cell(delaysN,1); bgbp_stk=cell(delaysN,1); 
% imge_stk=cell(delaysN,1); bgbe_stk=cell(delaysN,1); 
i=find(mask,1); sz=size(imread([folder '\' db(i).fname],'png'));
for i=1:delaysN, 
    img_stk{i}=zeros(sz);
%     imgp_stk{i}=zeros(sz);
%     imge_stk{i}=zeros(sz);
    bgb_stk{i}=zeros(sz);
%     bgbp_stk{i}=zeros(sz);
%     bgbe_stk{i}=zeros(sz);
end;
img_weights=zeros(1,delaysN);
bgb_weights=zeros(1,delaysN);
imgp_weights=zeros(1,delaysN);
bgbp_weights=zeros(1,delaysN);
imge_weights=zeros(1,delaysN);
bgbe_weights=zeros(1,delaysN);

img_avg=0; bgb_avg=0; 
imgp_avg=0;bgbp_avg=0;
imge_avg=0;bgbe_avg=0; 

Nimgs2load=length(find(mask)); iter=0;

for i=find(mask)
    iter=iter+1;
    disp(['#' num2str(i) '  (' num2str(mod(iter-1,Nimgs2load)+1) '/' num2str(Nimgs2load) ')']);
    data=imread([folder '\' db(i).fname],'png');
    %j=delaysN*(db(i).scan-1) + find(delays==db(i).del,1);
    idelay=find(delays==db(i).del,1);
    j=delaysN*(db(i).scan-1) + idelay;
    traces{8-db(i).peb}=[traces{8-db(i).peb} ; j sum(sum(data)) i]; %************************* last column=file index in the db(for tracking the filename)
    
    switch (db(i).peb)
        case 7 %img
            img_stk{idelay}=img_stk{idelay}+double(data);
            img_weights(idelay)=img_weights(idelay)+1;
        case 6 %bgb
            bgb_stk{idelay}=(bgb_stk{idelay}+double(data));
            bgb_weights(idelay)=bgb_weights(idelay)+1;
         case 5 %imgp
%             imgp_stk{idelay}=(imgp_stk{idelay}+double(data));
             imgp_weights(idelay)=imgp_weights(idelay)+1;
         case 4 %bgbp
%             bgbp_stk{idelay}=bgbp_stk{idelay}+double(data);
             bgbp_weights(idelay)=bgbp_weights(idelay)+1;
         case 3 %imge
%             imge_stk{idelay}=imge_stk{idelay}+double(data);
             imge_weights(idelay)=imge_weights(idelay)+1;
         case 2 %bgbe
%             bgbe_stk{idelay}=bgbe_stk{idelay}+double(data);
             bgbe_weights(idelay)=bgbe_weights(idelay)+1;
    end
    
%     switch (db(i).peb)
%         case 7 %img
%             k7=k7+1; img_avg=(img_avg*(k7-1)+double(data))/k7;
%         case 6 %bgb
%             k6=k6+1; bgb_avg=(bgb_avg*(k6-1)+double(data))/k6;
%         case 5 %imgp
%             k5=k5+1; imgp_avg=(imgp_avg*(k5-1)+double(data))/k5;
%         case 4 %bgbp
%             k4=k4+1; bgbp_avg=(bgbp_avg*(k4-1)+double(data))/k4;
%         case 3 %imge
%             k3=k3+1; imge_avg=(imge_avg*(k3-1)+double(data))/k3;
%         case 2 %bgbe
%             k2=k2+1; bgbe_avg=(bgbe_avg*(k2-1)+double(data))/k2;
%     end

    switch (db(i).peb)
        case 7 %img
            img_avg=img_avg+double(data);
        case 6 %bgb
            bgb_avg=bgb_avg+double(data);
        case 5 %imgp
            imgp_avg=imgp_avg+double(data);
        case 4 %bgbp
            bgbp_avg=bgbp_avg+double(data);
        case 3 %imge
            imge_avg=imge_avg+double(data);
        case 2 %bgbe
            bgbe_avg=bgbe_avg+double(data);
    end

end;

for idelay=1:delaysN, 
    if img_weights(idelay)~=0, img_stk{idelay}=img_stk{idelay}/img_weights(idelay); end;
    if bgb_weights(idelay)~=0, bgb_stk{idelay}=bgb_stk{idelay}/bgb_weights(idelay); end;
    %if imgp_weights(idelay)~=0, imgp_stk{idelay}=imgp_stk{idelay}/imgp_weights(idelay); end;
    %if bgbp_weights(idelay)~=0, bgbp_stk{idelay}=bgbp_stk{idelay}/bgbp_weights(idelay); end;
    %if imge_weights(idelay)~=0, imge_stk{idelay}=imge_stk{idelay}/imge_weights(idelay); end;
    %if bgbe_weights(idelay)~=0, bgbe_stk{idelay}=bgbe_stk{idelay}/bgbe_weights(idelay); end;
end;
if sum(img_weights)~=0, img_avg=img_avg/sum(img_weights);end;
if sum(bgb_weights)~=0, bgb_avg=bgb_avg/sum(bgb_weights);end;
if sum(imgp_weights)~=0, imgp_avg=imgp_avg/sum(imgp_weights);end;
if sum(bgbp_weights)~=0, bgbp_avg=bgbp_avg/sum(bgbp_weights);end;
if sum(imge_weights)~=0, imge_avg=imge_avg/sum(imge_weights);end;
if sum(bgbe_weights)~=0, bgbe_avg=bgbe_avg/sum(bgbe_weights);end;


%for j=1:delaysN, find(bgbp_weights~=0)
disp('Done.'); toc
%---------------------------------------------------------------------

%(moved this bit to the vmicropdifstk() function
%----------------------------------------------------------------------
%IMAGE CENTRE
disp('Finding image center');
if length(bgb_avg)~=1, data=img_avg-bgb_avg; else data=img_avg;end;
%data_inv=rot90(data,2);
%C=xcorr2(double(data),double(data_inv));
A=double(data); %C = conv2(A, rot90(conj(A),2));
%C = conv2(A, A);               %extremely slow in R2007b. reasonable in R2010b
%C = convolve2(A, A);           %dunno. requires matlab>=2013.
C = conv_fft2(A,A);             %very fast!
[y0,x0]=find(C==max(max(C)));
% x0=round(x0/2);
% y0=round(y0/2);
x0=floor(x0/2);
y0=floor(y0/2);


% pad=zeros(size(A));
% B=[pad pad pad; pad A pad; pad pad pad];
% C =  ifftshift(ifft2(fft2(fftshift(B)).^2));   %fast but incorrect.. why??
% [y0,x0]=find(C==max(max(C)));
% x0=round(x0/3);
% y0=round(y0/3);
% clear B C

% C = conv2FFT(A,A);            %fast but wrong also :(
% [y0,x0]=find(C==max(max(C)));


[x0,y0]
toc
% % figure;imagesc(data); hold on; circle(x0,y0,220); circle(x0,y0,100); hold off; axis image; title(['[x0,y0]=' num2str(x0) ', ' num2str(y0)]);
%----------------------------------------------------------------------


stats.folder=folder; t1=findstr(folder,'\'); stats.subfolder=folder(t1(end)+1:end);
stats.delays=delays;
stats.scans=scans;
stats.imsize=size(img_avg);
stats.vmicentre=[y0,x0];
stats.traces=traces;
stats.imavgs.img=img_avg;
stats.imavgs.bgb=bgb_avg;
stats.imavgs.imgp=imgp_avg;
stats.imavgs.bgbp=bgbp_avg;
stats.imavgs.imge=imge_avg;
stats.imavgs.bgbe=bgbe_avg;

stats.imstks.img=img_stk;
stats.imstks.bgb=bgb_stk;
% stats.imstks.imgp=imgp_stk;
% stats.imstks.bgbp=bgbp_stk;
% stats.imstks.imge=imge_stk;
% stats.imstks.bgbe=bgbe_stk;

stats.weights.img=img_weights;
stats.weights.bgb=bgb_weights;
% stats.weights.imgp=imgp_weights;
% stats.weights.bgbp=bgbp_weights;
% stats.weights.imge=imge_weights;
% stats.weights.bgbe=bgbe_weights;

stats.db=db;
%stats.mask=mask;
%stats.db2=db2;
return;

