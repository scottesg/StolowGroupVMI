function ydata = gaussian(A, xdata)
%function ydata = gaussian(A, xdata)
%y=y0+exp(-(x-x0)^2/tau^2)
%A(1)=ampl
%A(2)=tau
%A(3)=xoffset
%A(4)=yoffset
%AB, Nov.7,2008
ydata = A(4) + A(1)*exp(-((xdata-A(3))./A(2)).^2);
