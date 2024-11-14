function [Ir, r, I]=iabel2D_ab(img, method)
% [Ir, r, I]=iabel2D(img, method)  4-fold symmetrizes and takes the top right quadrant of the
% input IMG and returns the radial intensity Ir as a function of spherical
% radius r as well as the 'slice' I(\rho,z).  The axis of cylindrical 
% symmetry is assumed to be vertical so that the Abel inversion is performed
% across rows. The pixel size is
% assumed to be equal to one in both directions and r is returned in steps
% of \sqrt(2).  It is assumed the the image is centered in between the
% middle pixels of the image.  Even sized images are assumed.
%
% Binning is acheved using a step function \Pi(r-dr/2,r+dr/2) so that the intensity is 
%
% I(r)=\int I(x,y,z) \Pi(r+dr)dxdyz =\int I(\rho,z,phi)  \Pi(r+dr) \rho d\rho dz d\phi
%                                   =\int I(\rho,z)  \Pi(r+dr) \rho d\rho dz pi
%
% The data from IMG is first inverted one line at a time to produced I(\rho,z)
% evenly sampled in \rho and z.
%
%ab20160623:fiddled with the Ben's abel inversion code and found it being correct :)
%           +packaged the 1D abel inversion function in - this gave a tiny speed improvement
%           +in an attempt to make sense of formulas for building area elements matrix (Sij) I 
%           used the math I derived off the paper myself  - this reads much cleaner, gives 
%           identical result
%           +replaced the Ir=4*Ir/dr; with Ir=4*Ir; so that sum(Ir)=sum(img)
%           N.B. reconstructed image contains electron emitted from the
%           central slice, not all the emitted electrons. This is why
%           sum(output_img)<>sum(input_img)
%
%ab20161026:tweaked rebinning part a bit by replacing r=(dr/2):dr:Rmax; which sometimes was
%           giving wrong number of points with r=(dr/2):dr:(Rmax+dr/2); 


persistent S  nn % Store S to save time if n hasn't changed.

img=imgsym(img,'full');  %4-fold symmetrize the image

opts.method=method;  %pass the inverion method to use for iabel

[m n]=size(img);
q1img=img(1:m/2,(n/2+1):end);  % take upper right quadrant.

Irho=zeros(m/2,n/2); %Allocate some memory
I=Irho;
R=I;

x=(0.5:(n/2));  %pixel position
rho=x;          % pixel polar cylindrical radius
for z=1:1:m/2  %Perform the Abel inversion for each row in the quadrant
    I(z,:)=iabel(q1img(z,:).',opts).'; % I is the 'slice' or intensity in cylindrical coordinates I(rho, theta, z)
    Irho(z,:)=I(z,:)*pi.*rho;     % this has the pi integrating over phi and rho for adding the jacobian
    R(z,:)=sqrt((m/2-z+0.5).^2+x.^2);   % this is the 3D radius
    %N.B. the input image contains all the electrons emitted;
    %the reconstructed image contains electron emitted from the central
    %slice;
    %in order to reconstruct the electron yield as a finction of the
    %radius; thus we need to take into account the fact that the volume of
    %the "onion shells" depends on the shell raduis - hence the total
    %electron yield per rho will scale as pi*rho.
end



Rmax=floor(sqrt((n/2)^2+(m/2)^2));  %upper bound of R
dr=sqrt(2);
%r=(dr/2):dr:Rmax;  %spherical radius
r=(dr/2):dr:(Rmax+dr/2);  %spherical radius 
nr=length(r);
Ir=zeros(1,nr);

% To perform the integral just sum the points of Irho that are in the bin
for i=1:(m*n/4)  %for each element in R
    for j=1:nr  %for each element of Ir
        if (R(i)>=r(j)-dr/2) & (R(i)<(r(j)+dr/2)) %if it's within r-dr and r
            Ir(j)=Ir(j)+Irho(i);
            break
        end
    end
end
%Ir=4*Ir/dr;  %make it such that sum(Ir.*dr)=sum(q1img) and then times 4  such that sum(Ir.*dr)=sum(img)
Ir=4*Ir;  %make it such that sum(Ir)=sum(q1img) and then times 4  such that sum(Ir)=sum(img)

I=[fliplr(I) I;flipud(fliplr(I)) flipud(I) ];  %recreate the four quadrants from one



    function out=iabel(in,opts)
        % out=iabel(in) returns the inverse abel transform of IN.  IN must be a vertical vector
        % Unit spacing of data is assumed and the retruned data is at the same spacing.
        % iabel(in,opts) uses different methods to invert: opts.method='inv',
        % 'lsqnonneg', 'lsqlin'
        %  The implied values of x,y,r are 0.5:(n-0.5)
        
        % See: My thesis and
        % Meas. Sci. Technol. 16 (2005) 878\u2013884 doi:10.1088/0957-0233/16/3/032
        % Application of Abel inversion in real-time
        % calculations for circularly and elliptically
        % symmetric radiation sources
        % Y T Cho and S-J Na
        %
        
        %persistent S  n % Store S to save time if n hasn't changed.
        if isrow(in) in = in';
        end
        if isempty(S)|| nn~=length(in)
            nn=length(in);
            S=(zeros(nn,nn));
            for jj=1:nn;
                for ii=1:jj
                    S(ii,jj)=Sij(ii,jj);
                end
            end
        end
        
        if nargin==1 %no options
            opts.method='inv';
        end
        %keyboard
        switch opts.method;
            case 'lsqnonneg'
                x0=(0.5*(S\in));
                x0=abs(x0-min(x0));
                out=0.5*lsqnonneg(S,in,x0);
            case 'lsqlin'
                out=0.5* lsqlin(S,in,[],[],[],[],zeros(n,1),[]);  %for use with optimization toolbox, force >0
            case 'inv'
                out=0.5*(S\in);
            otherwise
                error('Unknown opts.method.')
        end
        
        
        %%ben's
%         function out=Sij(i,j)  %area of Abel segment ij
%             if i==j
%                 out = -sqrt((-1 + 2 * i)) * i / 0.2e1 + sqrt((-1 + 2 * i)) / 0.2e1 - (i ^ 2) * asin((1 / i * (i - 1))) / 0.2e1 + (i ^ 2) * pi / 0.4e1;
%             else
%                 out  = j ^ 2 * asin(0.1e1 / j * i) / 0.2e1 - j ^ 2 * asin(0.1e1 / j * (i - 0.1e1)) / 0.2e1 + (j ^ 2 / 0.2e1 - j + 0.1e1 / 0.2e1) * asin(0.1e1 / (j - 0.1e1) * (i - 0.1e1)) + (-j ^ 2 / 0.2e1 + j - 0.1e1 / 0.2e1) * asin(0.1e1 / (j - 0.1e1) * i) + sqrt(j ^ 2 - 0.2e1 * j + 0.2e1 * i - i ^ 2) * i / 0.2e1 - sqrt(j ^ 2 + 0.2e1 * i - 0.1e1 - i ^ 2) * i / 0.2e1 - sqrt(j ^ 2 - 0.2e1 * j + 0.2e1 * i - i ^ 2) / 0.2e1 + sqrt(j ^ 2 + 0.2e1 * i - 0.1e1 - i ^ 2) / 0.2e1 - sqrt(j ^ 2 - 0.2e1 * j + 0.1e1 - i ^ 2) * i / 0.2e1 + sqrt(j ^ 2 - i ^ 2) * i / 0.2e1;
%             end
%         end


        %ab's: easier to read but 5x slower
%             function out=Sij(i,j) %area of Abel segment ij
%                 P=@(i,j)(j^2/2*acos((i-1)/j)-(i-1)/2*sqrt(j^2-(i-1)^2));
%                 if i==j,
%                     out=P(i,j);
%                 else
%                     out=P(i,j)-P(i+1,j)-P(i,j-1)+P(i+1,j-1);
%                 end
%             end
%             
        %ab's: still somewhat easier to read but now the same speed as Ben's. 
        %(plus, because of packaging the 1D iabel into the same file as iabel2D - overall it is even a tiny bit faster than original :)
            function out=Sij(i,j) %area of Abel segment ij
                %P=@(i,j)(j^2/2*acos((i-1)/j)-(i-1)/2*sqrt(j^2-(i-1)^2));
                if i==j,
                    %out=P(i,j);
                    out=(j^2/2*acos((i-1)/j)-(i-1)/2*sqrt(j^2-(i-1)^2));
                else
                    %out=P(i,j)-P(i+1,j)-P(i,j-1)+P(i+1,j-1);
                    out=(j^2/2*acos((i-1)/j)-(i-1)/2*sqrt(j^2-(i-1)^2))-...
                        (j^2/2*acos((i)/j)-(i)/2*sqrt(j^2-(i)^2))-...
                        ((j-1)^2/2*acos((i-1)/(j-1))-(i-1)/2*sqrt((j-1)^2-(i-1)^2))+...
                        ((j-1)^2/2*acos((i)/(j-1))-(i)/2*sqrt((j-1)^2-(i)^2));
                end
            end
            
        
        
    end


end

