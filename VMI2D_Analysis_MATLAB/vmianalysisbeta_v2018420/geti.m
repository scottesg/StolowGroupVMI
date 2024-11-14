function index=geti(val, arr)
%index=geti(val, arr)
%returns nearest INTEGER index for "val" in 1D-array "arr"
%val can be an array itself

%ab20161108

for i=1:length(val(:)),
    [c index(i)] = min(abs(arr-val(i)));
end
index=reshape(index,size(val));

