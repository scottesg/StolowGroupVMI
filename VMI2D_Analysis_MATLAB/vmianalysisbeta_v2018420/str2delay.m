function psdelay=str2delay(str)
%% calculate ps delay from filename

t1=findstr(str,'.');
t2 = findstr(str,'ps');
s = str(t1(1)+1:t2-1);
psdelay = str2num(s);