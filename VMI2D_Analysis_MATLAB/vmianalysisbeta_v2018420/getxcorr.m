function fwhm = getxcorr(file) 
%Quick fix for finding crosscorrelation width in vmi scan traces.
%usage:
%getxcorr
%getxcorr('e:\vmidata')
%getxcorr('e:\vmidata\rawdata\Xcorr_1_Xe_5w_3w.yield.csv')
%
if ~exist('file','var')||isempty(file),
    folderin='E:\vmidata\rawdata';%c:\data\vmi';%E:\vmidata\rawdata';
    filein='';
else
    [folderin, name, ext] = fileparts(file);
    if exist(file, 'file') == 2,
        filein=[name ext];
    else
        filein='';
    end;
end;

if isempty(filein),
    [filein, folderin]=uigetfile('*.yield.csv','mytitle',folderin);
end;
if folderin(end)~='\', folderin=[folderin '\']; end;

data=importdata([folderin filein]);
figure; plot(data(:,1), [data(:,2:end) data(:,2)-data(:,3)]);
title(filein,'interpreter','none');

y=data(:,2)-data(:,3);
x=data(:,1)*1e-3;

fwhm=fitxcorr2(x,y,'gauss');