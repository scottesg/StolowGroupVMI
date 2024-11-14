function vmiplotaverages(vmistats)

if length(vmistats.imavgs.imgp)~=1, imgp=true; else imgp=false; end;
if length(vmistats.imavgs.imge)~=1, imge=true; else imge=false; end;

figure; %(1)
l=1;
if imgp, l=l+1; end;
if imge, l=l+1; end;
subplot(l,3,1); imagesc(vmistats.imavgs.img); title('img'); axis image;
subplot(l,3,2); imagesc(vmistats.imavgs.bgb);title('bgb'); axis image;
subplot(l,3,3); imagesc(vmistats.imavgs.img-vmistats.imavgs.bgb);title('img-bgb'); axis image;
if imgp, 
subplot(l,3,4); imagesc(vmistats.imavgs.imgp); title('imgp'); axis image;
subplot(l,3,5); imagesc(vmistats.imavgs.bgbp);title('bgbp'); axis image;
subplot(l,3,6); imagesc(vmistats.imavgs.imgp-vmistats.imavgs.bgbp); title('imgp-bgbp'); axis image; 
end;
if imge, 
subplot(l,3,7); imagesc(vmistats.imavgs.imge); title('imge'); axis image;
subplot(l,3,8); imagesc(vmistats.imavgs.bgbe); title('bgbe'); axis image; 
subplot(l,3,9); imagesc(vmistats.imavgs.imge-vmistats.imavgs.bgbe); title('imge-bgbe'); axis image;
end;
set(gcf,'name',[vmistats.subfolder ':   average images (1)']);

%(2)
figure;        subplot(2,2,1); imagesc(vmistats.imavgs.img-vmistats.imavgs.bgb); title('img-bgb'); axis image;
if imgp,       subplot(2,2,2); imagesc(vmistats.imavgs.img-vmistats.imavgs.bgb-vmistats.imavgs.imgp+vmistats.imavgs.bgbp);title('(img-bgb)-(imgp-bgbp)'); axis image; end
if imge,       subplot(2,2,3); imagesc(vmistats.imavgs.img-vmistats.imavgs.bgb-vmistats.imavgs.imge+vmistats.imavgs.bgbe);title('(img-bgb)-(imge-bgbe)'); axis image; end
if imgp&&imge, subplot(2,2,4); imagesc(vmistats.imavgs.img-vmistats.imavgs.bgb-vmistats.imavgs.imgp+vmistats.imavgs.bgbp-vmistats.imavgs.imge+vmistats.imavgs.bgbe);title('(img-bgb)-(imgp-bgbp)-(imge-bgbe)'); axis image; end
set(gcf,'name',[vmistats.subfolder ':   average images (2)']);

figure; %(3)
imagesc(vmistats.imavgs.img-vmistats.imavgs.bgb-vmistats.imavgs.imgp+vmistats.imavgs.bgbp);title('(img-bgb)-(imgp-bgbp)'); axis image;
set(gcf,'name',[vmistats.subfolder ':   show image centre']);
hold on; 
circle(vmistats.vmicentre(2),vmistats.vmicentre(1),220); circle(vmistats.vmicentre(2),vmistats.vmicentre(1),150); circle(vmistats.vmicentre(2),vmistats.vmicentre(1),100); 
line([vmistats.vmicentre(2) vmistats.vmicentre(2)],[1 vmistats.imsize(1)],'linestyle',':','color','r');
line([1 vmistats.imsize(2)],[vmistats.vmicentre(1) vmistats.vmicentre(1)],'linestyle',':','color','r');
hold off; %axis image; title(['[x0,y0]=' num2str(x0) ', ' num2str(y0)]);
