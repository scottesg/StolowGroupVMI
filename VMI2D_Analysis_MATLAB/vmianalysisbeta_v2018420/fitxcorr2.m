function tau = fitxcorr2(xdata, ydata, model, varargin)
%function tau = fitxcorr2(xdata, ydata, model)
%function tau = fitxcorr2(xdata, ydata, model, xrange)
%where "model" is 'erf', 'gauss', or 'erf+gauss'
%      "xrange"(optional) defines the range where the fit will be endeavoured [xfirst xlast]
%Example of usage: fitxcorr(delays,integrateTRPES(600,900,'tof12'),'erf')
%                  fitxcorr(delays,integrateTRPES(700,1000,'tof12'),'erf',[-1 0.3])
%                  fitxcorr(delays,-integrateTRPES(1600,1800,'tof12'),'gauss')   ("-" for hole-burned peaks)
%AB, Nov.8,2008;

%N.B.: This function depends on functions 'threshold1Darray.m', plus 'myerf.m', 'gaussian.m' or 'erfplusgauss.m' correspondingly

%last edited Nov.11,2008

%xdata=delays;
%ydata=integrateTRPES(600,900,'tof12');

xdatafull=xdata;
ydatafull=ydata;

if ~isempty(varargin)
    if length(varargin)==1 , first=varargin{1}(1); last=varargin{1}(2); end;
    if length(varargin)==2 , first=varargin{1}; last=varargin{2}; end;
    firsti=ceil(min(threshold1Darray(first,xdata),threshold1Darray(last,xdata)));
    lasti=floor(max(threshold1Darray(first,xdata),threshold1Darray(last,xdata)));
    xdata=xdata(firsti:lasti);
    ydata=ydata(firsti:lasti);
end

switch model
    case 'erf'
        
        %P0=[A tau xoffset yoffset]

        %starting values autoguess    
        A=max(ydata)-min(ydata);
        yoffset=min(ydata);
        xoffset=xdata( floor( threshold1Darray(yoffset+A/2,ydata) ) );
        tau=abs( xdata( floor( threshold1Darray(yoffset+A*3/4,ydata) ) ) - xdata( floor( threshold1Darray(yoffset+A/4,ydata) ) ) );
        P0=[A tau xoffset yoffset];

        % P0=[(max(ydata)-min(ydata)) 0.25 0 min(ydata)];
        P = lsqcurvefit(@myerf,P0,xdata,ydata);

        figure; plot(xdatafull, ydatafull, xdata, myerf(P,xdata));

    
    
    case 'gauss'   % fit gaussian, report FWHM of xcorr


        %starting values autoguess  (same algorithm as for "erf")
        A=max(ydata)-min(ydata);
        yoffset=min(ydata);
        xoffset=xdata( floor( threshold1Darray(yoffset+A/2,ydata) ) );
        tau=abs( xdata( floor( threshold1Darray(yoffset+A*3/4,ydata) ) ) - xdata( floor( threshold1Darray(yoffset+A/4,ydata) ) ) );
        P0=[A tau xoffset yoffset];

        %P0=[max(ydata)-min(ydata) 0.1 -0.15 min(ydata)];

        P = lsqcurvefit(@gaussian,P0,xdata,ydata);

        figure; plot(xdatafull, ydatafull, xdata, gaussian(P,xdata));


    case 'erf+gauss'   % fit gaussian, report FWHM of xcorr


        %starting values autoguess  (same algorithm as for "erf")
        A=max(ydata)-min(ydata);
        yoffset=min(ydata);
        xoffset=xdata( floor( threshold1Darray(yoffset+A/2,ydata) ) );
        tau=abs( xdata( floor( threshold1Darray(yoffset+A*3/4,ydata) ) ) - xdata( floor( threshold1Darray(yoffset+A/4,ydata) ) ) );
        P0=[A tau xoffset yoffset A];

        %P0=[max(ydata)-min(ydata) 0.1 -0.15 min(ydata)];

        P = lsqcurvefit(@erfplusgauss,P0,xdata,ydata);

        figure; plot(xdatafull, ydatafull, xdata, erfplusgauss(P,xdata));

        
end

%disp('      FWHM    t0');
%disp([2*P(2)*sqrt(log(2)) P(3)]*1000);

%title(['FWHM = ' num2str(2*P(2)*sqrt(log(2))*1e3) 'fs;       t0 =' num2str(P(3)*1e3) 'fs']);

annotation('textbox',...
    'String',['FWHM = ' num2str(2*P(2)*sqrt(log(2))*1e3) 'fs \newline t0 =' num2str(P(3)*1e3) 'fs'],...
    'LineStyle','none',...
    'Position',[0.4357 0.3183 0.2488 0.1175],...
    'FitHeightToText',...
    'on');

tau = 2*P(2)*sqrt(log(2))*1e3;