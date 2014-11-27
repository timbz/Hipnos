% aperture function for rectangular, circular and serrated apertures

% type: 	a) circ
%				par1 = x-radius
%				par2 = y-radius
%				par3 = angle
%				par4 = x-center
%				par5 = y-center
%
%			b) rect
%				par1 = x-width
%				par2 = y-witdh
%				par3 = angle
%				par4 = x-center
%				par5 = y-center
%
% 			c) serr
%				par1 = radius
%				par2 = modulation depth
%				par3 = angular frequency 1
%				par4 = angular frequency 2
%				par5 = ...

function C=aperture(type,par1,par2,par3,par4,par5)

global xunits xelements yunits yelements;

% precalculations
rdx=xelements*par1/xunits;
rdy=yelements*par2/yunits;
wdx=xelements*par1/(2*xunits);
wdy=yelements*par2/(2*yunits);
xc=par4*xelements/xunits;
yc=par5*yelements/yunits;
an=-par3/180*pi;

md = par2;
if (par2>1 & par2<0)
   md=0;
end
f1=floor(par3);
f2=floor(par4);

[j,k]=formatrix(yelements,xelements);

if type=='circ',
   C = (cos(an)*(j-yelements/2-yc)+sin(an)*(k-xelements/2-xc)).^2./rdy^2+(-sin(an)*(j-yelements/2-yc)+cos(an)*(k-xelements/2-xc)).^2./rdx^2 < 1;
end

if type=='rect',
   C = ~( (abs(cos(an)*(j-yelements/2-yc)+sin(an)*(k-xelements/2-xc))>wdy) | (abs(-sin(an)*(j-yelements/2-yc)+cos(an)*(k-xelements/2-xc))>wdx));
end

if type=='serr',
   alpha=atan((j-yelements/2)./(k-xelements/2));
	r=rdx*(1+md*cos(f1*alpha).*cos(f2*alpha));
	C = (j-yelements/2).^2+(k-xelements/2).^2 < r.^2;
end