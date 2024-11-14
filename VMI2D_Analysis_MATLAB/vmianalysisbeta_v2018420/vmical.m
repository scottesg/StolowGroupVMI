function K=vmical(file)
%ab2017
%last edit 20180418: added plot of the fit when parsing a 'R-KE' file

fid=fopen(file);
str=fgets(fid);
if sum(findstr(str,'R-KE')), format='R-KE';
elseif sum(findstr(str,'K')), format='K';
else fclose(fid); disp('unknown format'); K=NaN; return;
end;
data=[];
str=fgets(fid);
while ischar(str)
    a=sscanf(str,'%f')';
    if ~isempty(a),data=[data; a];end;
    str=fgets(fid);
end;
fclose(fid);


switch format
    case {'R-KE'}
        xdata=data(:,1);
        ydata=data(:,2);
        %K=fit
        fun = @(p) sum((ydata - (p(1)*xdata.^2)).^2);
        pguess = [1];
        [p,fminres] = fminsearch(fun,pguess);
        K=p(1);
        
        xfit=linspace(0,max(xdata),50);
        yfit=K*xfit.^2;
        figure; plot(xfit,yfit,'-',xdata,ydata,'*r');
        xlabel('radius [pix]');
        ylabel('ke [eV]');
        title(['vmi energy calibration: ke[eV]=K*r[pix]^2;  K = ' num2str(K)]);
    case {'K'}
        K=data(1);
    otherwise
        disp('error parsing file');
end
        

return;

% %Function to calculate the sum of residuals for a given p1 and p2
% fun = @(p) sum((ydata - (p(1)*cos(p(2)*xdata)+p(2)*sin(p(1)*xdata))).^2);
% 
% %starting guess
% pguess = [1,0.2];
% 
% %optimise
% [p,fminres] = fminsearch(fun,pguess)