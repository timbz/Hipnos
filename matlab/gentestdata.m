function test

warning off
close all;

global unitvectorx unitvectory lambda xunits

initialize;

A = gaussfill(1,0.002,10,0,0);
dlmwrite('csv/gaussfill.csv', A, 'precision', 10);

B = aperture('circ',0.002,0.002,0,0,0);
dlmwrite('csv/aperture-trans.csv', B, 'precision', 10);

C=A.*B;
dlmwrite('csv/aperture.csv', C, 'precision', 10);

D=propTF(C,xunits,lambda, 100);
dlmwrite('csv/propagation.csv', D, 'precision', 10);

E=lense(25, 0, 0);
dlmwrite('csv/lense-trans.csv', E, 'precision', 10);

F=D.*E;
dlmwrite('csv/lense.csv', F, 'precision', 10);

G=propFF(F,xunits,lambda, 100);
dlmwrite('csv/propagation2.csv', G, 'precision', 10);
