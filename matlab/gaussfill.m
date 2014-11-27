function C=gaussfill(amplitude,beamwidth,phaseradius,xcenter,ycenter);

global xelements yelements xunits yunits lambda unitvectorx unitvectory;

% new units for equal sampling
xunits=beamwidth*sqrt(pi*xelements)
yunits=beamwidth*sqrt(pi*yelements)

unitvectorx = -xunits/2:xunits/xelements:xunits/2;
unitvectorx = unitvectorx(:,1:xelements);
unitvectory = -yunits/2:yunits/yelements:yunits/2;
unitvectory = unitvectory(:,1:yelements);

% conversion from metric units into discrete sampling points
bwx=beamwidth*xelements/xunits;
bwy=beamwidth*yelements/yunits;
prx=phaseradius*xelements/xunits;
pry=phaseradius*yelements/yunits;
lmdx=lambda*xelements/xunits;
lmdy=lambda*yelements/yunits;
xc=xcenter*xelements/xunits;
yc=ycenter*yelements/yunits;

% filling the matrix with 3D-Gaussian function
[j,k]=formatrix(yelements,xelements);
C = amplitude*exp(-((j-yelements/2-yc).^2)/bwy^2 - ((k-xelements/2-xc).^2)/bwx^2) .* exp(-i*pi*(((j-yelements/2-yc).^2)/(pry*lmdy) + ((k-xelements/2-xc).^2)/(prx*lmdx)));
