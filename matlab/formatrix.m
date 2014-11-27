function [rows,columns]=formatrix(counter1,counter2)

l=(0:counter1-1)';
rows=l(:,ones(counter2,1),:);

l=(0:counter2-1)';
columns=l(:,ones(counter1,1),:)';

% generates a matrix:

% 1 1 1 1 ...
% 2 2 2 2 ...
% .
% .
% .
% rows rows ...