% 2D-integration for intensity

function B=integration(input)

global xunits xelements yunits yelements

A=xunits*yunits/(xelements*yelements)*abs(input).^2;
B=sum(sum(A));

