function yo = smoothimg(y,wid)

w = int32(wid);
if wid > 0
    
w=double(w);
w1 = 4*[w w];
w1 = w1 + mod(w1+1,2);
f = fspecial('gaussian',w1,w);
yo = filter2(f,y,'same');
else 
    yo = y;
end
