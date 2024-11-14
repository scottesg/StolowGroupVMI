function vmistats=vmibgsubtr(vmistats,bgsubtr)
%bgsubtr - a three digit binary string 'peb'

peb=bin2dec(bgsubtr);
delaysN=length(vmistats.imstks.img);

switch peb
    case 0
        vmistats.imstks.difstk=vmistats.imstks.img;
    case 1
        for j=1:delaysN, vmistats.imstks.difstk{j}=vmistats.imstks.img{j}-vmistats.imstks.bgb{j};end;
    case 3
        %for j=1:delaysN, vmistats.imstks.difstk{j}=(vmistats.imstks.img{j}-vmistats.imstks.bgb{j})-(vmistats.imstks.imge{j}-vmistats.imstks.bgbe{j}); end;
        for j=1:delaysN, vmistats.imstks.difstk{j}=(vmistats.imstks.img{j}-vmistats.imstks.bgb{j})-(vmistats.imavgs.imge-vmistats.imavgs.bgbe); end;
    case 4
        %for j=1:delaysN, vmistats.imstks.difstk{j}=(vmistats.imstks.img{j}-vmistats.imstks.bgb{j})-(vmistats.imstks.imge{j}-vmistats.imstks.bgbe{j}); end;
        for j=1:delaysN, vmistats.imstks.difstk{j}=vmistats.imstks.img{j}-vmistats.imavgs.imgp; end;
    case 5
        %for j=1:delaysN, vmistats.imstks.difstk{j}=(vmistats.imstks.img{j}-vmistats.imstks.bgb{j})-(vmistats.imstks.imgp{j}-vmistats.imstks.bgbp{j}); end;
        for j=1:delaysN, vmistats.imstks.difstk{j}=(vmistats.imstks.img{j}-vmistats.imstks.bgb{j})-(vmistats.imavgs.imgp-vmistats.imavgs.bgbp); end;
    case 7
        %for j=1:delaysN, vmistats.imstks.difstk{j}=(vmistats.imstks.img{j}-vmistats.imstks.bgb{j})-(vmistats.imstks.imgp{j}-vmistats.imstks.bgbp{j})-(vmistats.imstks.imge{j}-vmistats.imstks.bgbe{j}); end;
        for j=1:delaysN, vmistats.imstks.difstk{j}=(vmistats.imstks.img{j}-vmistats.imstks.bgb{j})-(vmistats.imavgs.imgp-vmistats.imavgs.bgbp)-(vmistats.imavgs.imge-vmistats.imavgs.bgbe); end;
    otherwise
        disp('not implemented ');
end

vmistats.imavgs.difstk=zeros(size(vmistats.imstks.img{1}));
for j=1:delaysN, 
    vmistats.imavgs.difstk=vmistats.imavgs.difstk+vmistats.imstks.difstk{j}/delaysN;
end;


