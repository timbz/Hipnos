function output=fourier(input)

aux1=size(input,1);
aux2=size(input,2);

pk(1:2:aux2)=-1;
pk(2:2:aux2)=1;

pq(1:2:aux1)=-1;
pq(2:2:aux1)=1;

% phase factor to move FT -0.5 at the end
[m,l]=formatrix(aux1,aux2);
pk3=exp(-i*pi*(l-m)/aux2-i*pi);

input=pk3.*input;

% phase correction factor for interval -T/2 ... +T/2
[m,l]=formatrix(aux2,aux1);
pk2=(-1).^(m+l).*exp(-i*pi*(l+m)/aux2)*exp(-i*pi);

% 1st dimension FFT
l=pk'; pk=l(:,ones(aux1,1),:);
input=(fft(((input(aux1:-1:1,:).').*pk)));

% 2nd dimension FFT
l=pq'; pq=l(:,ones(aux2,1),:);
output=((fft(((input(aux2:-1:1,:).').*pq)))).*pk2(aux2:-1:1,:).';
