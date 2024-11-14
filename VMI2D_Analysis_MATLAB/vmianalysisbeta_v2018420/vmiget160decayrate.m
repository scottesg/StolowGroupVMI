function R=vmiget160decayrate(vmistats,drange,verbosity)
%R=vmiget160decayrate(vmistats [,drange])
%where drange is the range of pp delays to be used as a reference 
%if drange is not provided then signal at all delays is used.
%
%The function offers a take on correction of pp signal decay due to 5th
%hamonic power dropping as a function of time between refills
%The decay is (for now) assumed to be a constant percent drop per scan (for now)
%
%ab20161026
%last edit 20170206 (bugfix)

if ~exist('drange','var')||isempty(drange), drange=[vmistats.delays(1) vmistats.delays(end)]; end;
drange=sort(drange);

for iscan=vmistats.scans%1:(max([vmistats.db.scan])),
    maskpp=find(([vmistats.db.peb]==7)&([vmistats.db.del]>=drange(1))&([vmistats.db.del]<=drange(end))&([vmistats.db.scan]==iscan)); %pp
    maskng=find(([vmistats.db.peb]==6)&([vmistats.db.del]>=drange(1))&([vmistats.db.del]<=drange(end))&([vmistats.db.scan]==iscan)); %ppng
    pp(iscan)=0;
    ng(iscan)=0;
    for i=maskpp, %i index runs through all files in the folder
        %data=imread([vmistats.folder '\' vmistats.db(i).fname],'png');
        %pp(iscan)=sum(sum(data));
        pp(iscan)=vmistats.traces{1}(find(vmistats.traces{1}(:,3)==i),2);
    end;
    for i=maskng, 
        %data=imread([vmistats.folder '\' vmistats.db(i).fname],'png');
        %ng(iscan)=sum(sum(data));
        ng(iscan)=vmistats.traces{2}(find(vmistats.traces{2}(:,3)==i),2);
    end;
end


aa=pp-ng;
aa_avg=sum(aa)/length(aa(aa>0));
sc=aa_avg./aa;
sc(aa==0)=NaN;

if exist('verbosity','var'),
    figure;
    subplot(2,2,1); plot([pp;ng;pp-ng]','.-'); xlabel('scan #'); title('total yield vs scan #'); legend({'pp','ppng','pp-ppng'});  axis tight;
    % figure; plot([aa;aa.*sc]','.-');
    subplot(2,2,3); plot(sc,'.-'); xlabel('scan #'); title('multiplier needed to bring yields to the average');  axis tight;
    % figure; plot(1./sc,'.-'); title('1/sc');
    % figure; plot(diff(sc),'.-'); title('diff(sc)');
end

%
rate=[]; %scale factor change per scan
for i=1:(length(sc)-1)
if (sc(i+1)>0.9*sc(i)), %rate=[rate sc(i+1)-sc(i)]; 
                         rate=[rate sc(i+1)/sc(i)];% else rate=[rate 0];
end; %skip refils

end;
rate
R=sum(rate)/length(rate(rate>0)); %avg rate
R

if exist('verbosity','var'), subplot(2,2,4); plot(rate','.-'); title('VUV decay compensation rate'); hold on; plot([1, length(rate)], R*[1 1],'k--'); hold off; axis tight; legend({'rate','R=avg(rate)'}); end;




%vmistats.R=R; %R is a correction factor changing from the beginning of a scan (1) to the end (1+R)
