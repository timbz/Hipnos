% the beam array is multiplied by a phase factor array 
% depending on the focal length (foc)
%

function output=lense(focallength,xcenter,ycenter)

global xelements xunits yelements yunits lambda

focx=focallength*xelements/xunits;
focy=focallength*yelements/yunits;
lmdx=lambda*xelements/xunits;
lmdy=lambda*yelements/yunits;
xc=xcenter*xelements/xunits;
yc=ycenter*yelements/yunits;

[j,k]=formatrix(yelements,xelements);
output=exp(i*pi*(((j-yelements/2-yc).^2)/(focy*lmdy) + ((k-xelements/2-xc).^2)/(focx*lmdx)));