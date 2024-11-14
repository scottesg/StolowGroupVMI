function [dataout]=flattenbg(datain,delay1,delay2,delays)
%function [dataout]=flattenbg(datain,del1,del2,delays)
%'delays' is expected to be a column,i.e. delays(delaysN,1)  
%'datain' - datain(delaysN,KEsN)
%
%ab20160712: a rewrite of the old function
flip=0;
if size(delays,1)~=length(delays), delays=delays';end;
if size(datain,1)~=length(delays), if size(datain,2)~=length(delays), disp('check dimensionalities of the input arrays'); return;end;
                                   flip=1;
                                   datain=datain';
end

idelay1=floor( threshold1Darray(delay1,delays) );
idelay2=floor( threshold1Darray(delay2,delays) );
if idelay2<idelay1, i=idelay2;idelay2=idelay1;idelay1=i;clear i;end;

mask=zeros(size(delays));
mask(idelay1:idelay2)=1;
bg=(mask'*datain/sum(mask));
dataout=datain-ones(size(delays))*bg;
if flip, dataout=dataout';end;
return;
