%function [i0] = threshold1Darray(x0, X)
%Compares threshold x0 to the values in 1D array X starting at start index
%until it finds a pair of consecutive elements such that threshold x0 is
%greater than the value of the first element and less than or equal to the
%value of the second element. 
%Use this function only with sorted arrays.
%Returns fractional index i, which is zero when no thereshold was found.
%last edited ab20150512: small fix of the limit condition

%last edited August 12, 2008
%last edited September 29, 2007

function [i0] = index1Darray_v(x0, X)
i0=zeros(size(x0));
for j=1:length(x0),
    
n = length(X); %i0(j)=0;

% return zero if X was an empty array
if n==0, return; end; %added 120808

for i=1:(n-1)
    %if array sorted ascending
    if ( (X(i)<=x0(j)) && (X(i+1)>x0(j)) )
        i0(j)=i+(x0(j)-X(i))/(X(i+1)-X(i));
        break;
    end;
    %if array sorted descending
    if ( (X(i)>=x0(j)) && (X(i+1)<x0(j)) )
        i0(j)=i+(x0(j)-X(i))/(X(i+1)-X(i));
        break;
    end;

end;

%change the behaviour: now never return zero(except when X is empty). But limits instead;
%actually it could be a good idea to do it outside this function
if ~i0(j) 
    %if ( (X(1)>x0) & (X(n)>x0) ) i0=n; end;
    %if ( (X(1)<x0) & (X(n)<x0) ) i0=1; end;
    if (X(1)<X(n))  % ascending
        if (X(1)>x0(j)), i0(j)=1; else i0(j)=n; end;
    else            %descending
        %if (X(n)>x0), i0=n; else i0=1; end;
        if (X(n)>=x0(j)), i0(j)=n; else i0(j)=1; end; % to accomodate cases like X=[3,2,1,0]; and x0=0; %ab20150512        
    end;
    
end;
end;

return;
