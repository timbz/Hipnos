%	initialization routine

global xelements yelements zelements lambdaelements T 
global xunits yunits tunits lambda unitvectorx unitvectory lambdavector % obsolete but used

nm = 0.000000001; % in m

xunits 			= 0.01;		% spatial x-length of the source array in meter
yunits 			= 0.01;		% spatial y-length of the source array in meter
xelements 		= 256;		% number of array xelements (square)
yelements 		= 256;			% number of array yelements (square)
zelements		= 64;


unitvectorx = -xunits/2:xunits/xelements:xunits/2;	% obsolete but used
unitvectorx = unitvectorx(:,1:xelements);				% obsolete but used
unitvectory = -yunits/2:yunits/yelements:yunits/2;	% obsolete but used
unitvectory = unitvectory(:,1:yelements);				% obsolete but used

lambdamin 		= 1016 * nm;
lambdamax 		= 1046 * nm;
lambdaelements = 32;
lambda 			= 1030 * nm;	% wavelength in meter

lambdavector = lambdamax:-(lambdamax-lambdamin)/lambdaelements:lambdamin;
lambdavector = lambdavector(:,1:lambdaelements);

% defining the TRACE - MATRIX [[ T ]]

%u0(1:lambdaelements,:) 		= units;		% initial beam size
%r0(1:lambdaelements,:) 		= [0 0 0]; 	% initial beam location
%k0(1:lambdaelements,:) 		= [0 0 1];	% initial beam direction
%p0(1:lambdaelements,:) 		= [1 0 0];	% initial beam orientaion
%phi0(1:lambdaelements,:) 	= 0;			% initial phase

%T = [lambdavector',phi0,u0,r0,k0,p0]	% lambdaelements x 12 Matrix