function roi = annroimask(y,y0,x0,r1,r2)
%% generate annular roi with origin x0,y0 bounding radii r1, r2

%dm2015
%ab201510 add an option to use dimension vector instead of inputing a matrix  
%ab20151028 fixed a bug with x and y being flipped
sz = size(y);
ny = sz(1);
nx = sz(2);
if sum(sz)==3, ny=y(1); nx=y(2); end; %if y is a dimension vector then use it
%roi = uint8(zeros(nx,ny));
roi = (zeros(ny,nx));

for i = 1:nx
    for j = 1:ny
        dst = sqrt((i - x0)*(i - x0)+(j - y0)*(j - y0));
        if (dst >= r1) && (dst <= r2) 
            roi(j,i) = 1;
        end
    end
end
        
        
