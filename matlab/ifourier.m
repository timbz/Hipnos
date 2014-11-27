function output=ifourier(input)

aux1=size(input,1);
aux2=size(input,2);

pk(1:2:aux2)=-1;
pk(2:2:aux2)=1;

pq(1:2:aux1)=-1;
pq(2:2:aux1)=1;

% phase factor to move FT -0.5 at the end
[m,l]=formatrix(aux1,aux2);
pk3=exp(-i*pi*(l+m)/aux2-i*pi);

input=pk3.*input;

% phase correction factor for interval -F/2 ... 0 F/2
[m,l]=formatrix(aux2,aux1);
pk2=(-1).^(m+l).*exp(-i*pi*(m+l)/aux2)*exp(-i*pi);

% 1st dimension FFT
l=pk'; pk=l(:,ones(aux1,1),:);
input=1/aux2*(fft(((input(1:aux1,:).').*pk)));

% 2nd dimension FFT
l=pq'; pq=l(:,ones(aux2,1),:);
output=1/aux1*(((fft(((input(1:aux2,:).').*pq)))).*pk2(1:aux2,:).');
