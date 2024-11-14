function [stkout]=flattenbg_stk(stkin,delay1,delay2,delays,R)
%[stkout]=flattenbg_stk(stkin,delay1,delay2,delays [,R])
%  same syntax as in flattenbg() but for stk data
%  with extra (optional) parameter R that defines scaling of signal 
%  necessary for correcting for the decay of 5th harmonic output.  

%ab20170317

if ~exist('R','var')||isempty(R), R=1;end;

stkout=stkin;

if size(delays,1)~=length(delays), delays=delays';end;
if length(stkin)~=length(delays), disp('check dimensionalities of the input arrays'); stkout=stkin; return;end;
idelay1=floor( threshold1Darray(delay1,delays) );
idelay2=floor( threshold1Darray(delay2,delays) );
if idelay2<idelay1, i=idelay2;idelay2=idelay1;idelay1=i;clear i;end;

if ~iscell(stkin), stkin={stkin};end;
nz=length(stkin);

scale=linspace(1,R,length(delays));
for i=1:nz,
    stkin{i}=stkin{i}*scale(i);
end;

bg=zeros(size(stkin{1}));
for i=idelay1:idelay2,
    bg=bg+stkin{i};
end;
bg=bg/(idelay2-idelay1+1);

for i=1:nz,
    stkout{i}=stkin{i}-bg;
end;


%for i=1:length(vmistats.imstks.idifstk),
%    
%tmpstk=(vmistats.imstks.idifstk)
%Ir=Ir.*(linspace(1,R,length(delays))'*ones(size(r)));