function test

warning off
close all;

global unitvectorx unitvectory lambda xunits

initialize;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = gaussfill(1,0.002,10,0,0);
% Parameter: 
% Amplitude, Strahldurchmesser, Wellenfront-Krümmungs-Radius, x-, y- Postion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
surf(unitvectorx,unitvectory,abs(A))
axis tight;
shading interp
view(0,90);

figure;
surf(unitvectorx,unitvectory,angle(A))
axis tight;
shading interp
view(0,90);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = aperture('circ',0.002,0.002,0,0,0);
% Parameter: 
% Blenden-Typ, x-,y-Durchmesser, Drehwinkel, x-, y- Postion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
surf(unitvectorx,unitvectory,abs(B))
axis tight;
shading interp
view(0,90);

C=A.*B;

figure;
surf(unitvectorx,unitvectory,abs(C))
axis tight;
shading interp
view(0,90);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%D=propagation(C,100);
D=propagation2(C,0);

% propagation.m = Fernfeld-Propagation 
% propagation2.m = Nahfeld-Propagation 
% Parameter: Distanz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
surf(unitvectorx,unitvectory,abs(D))
axis tight;
shading interp
view(0,90);

figure
surf(unitvectorx,unitvectory,angle(D))
axis tight;
shading interp
view(0,90);


E=propTF(C,xunits,lambda, 0);
%E=propFF(C,xunits,lambda, 100);

figure
surf(unitvectorx,unitvectory,abs(E))
axis tight;
shading interp
view(0,90);


figure
surf(unitvectorx,unitvectory,angle(E))
axis tight;
shading interp
view(0,90);
