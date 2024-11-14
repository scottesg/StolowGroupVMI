function img=imgsym(img,type)
%symmetrize IMG according to TYPE.  The image should be centred first.

[m n]=size(img);
switch type
    case 'full'

    case 'top' %only use the top half
    img((m/2+1:m),:)=0;
    img=img*2;
    case 'bottom' %only use the bottom half
    img((1:m/2),:)=0;
    img=img*2;
    case 'right' %only use the bottom half
    img(:,1:n/2)=0;
    img=img*2;
    case 'left' %only use the bottom half
    img(:,(n/2+1):n)=0;
    img=img*2;
    otherwise
        error('symmetrize type not identified')
 end
        img=((img+fliplr(img))+flipud(img+fliplr(img)))/4;
