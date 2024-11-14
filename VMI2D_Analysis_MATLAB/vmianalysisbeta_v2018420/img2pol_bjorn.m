function polImg = img2pol_bjorn(in_Img,x0,y0,R,phi,method)
% function polImg = img2pol(in_Img,x0,y0,R,phi,method)
%
% IMG2POL resample an image on polar coordinates
% centered at pixel X0,Y0 
% with radial and angular resolution R resp PHI, 
% method is interpolation method to use: [{'linear'}|'cubic'|'nearest']. 
% Use 2-D arrays for R and PHI. [R,PHI] = meshgrid(r,phi);
 
% Version 0.01 Copyright B Gustavsson 20050607
% No argument checking et al at all.
 
polImg = interp2(in_Img,x0+R.*cos(phi),y0+R.*sin(phi),method);
