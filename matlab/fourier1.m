function output=fourier1(input)

aux=size(input,2);

pk(1:2:aux)=-1;
pk(2:2:aux)=1;


% phase factor to move FT -0.5 at the end
pk3=exp(-i*pi*(1:aux)/aux+i*pi);

input=pk3.*input;

% phase correction factor for interval -T/2 ... +T/2
pk2=-pk.*exp(-i*pi*(1:aux)/aux)*exp(-i*pi);

% 1st dimension FFT
output=(fft((input.*pk)).*pk2);

