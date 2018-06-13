%MAIN() Executive script for running AdaBoost on BUPA liver disorder data
%   Data Link:
%   https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/
%
%   Author: Asher Hensley
%   $Revision: 1.0 $  $Date: 2015/03/19 19:23:54 $
%
%   MIT License
%   
%   Permission is hereby granted, free of charge, to any person obtaining a 
%   copy of this software and associated documentation files (the 
%   "Software"), to deal in the Software without restriction, including 
%   without limitation the rights to use, copy, modify, merge, publish, 
%   distribute, sublicense, and/or sell copies of the Software, and to 
%   permit persons to whom the Software is furnished to do so, subject to 
%   the following conditions:
% 
%   The above copyright notice and this permission notice shall be included 
%   in all copies or substantial portions of the Software.
% 
%   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
%   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
%   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
%   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
%   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
%   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
%   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

%Clean Up
clear
close all
clc

%Load Data
fid = fopen('bupa.data.txt');
bupa = textscan(fid,'%f','delimiter',',');
bupa = reshape(bupa{1},7,[])';
X = bupa(:,1:6);
y = bupa(:,7);
N = size(X,1);

%Format Labels
M = y==1;
y(M) = -1;
y(~M) = +1;

%Setup
ntrials = 50;
T = 100;
train_split = round(0.9*N);

%Run
result = adaboost(X,y,ntrials,T,train_split);

%Plot Results
figure,hold on,grid on
plot(1:T,mean(result.error_train),'b')
plot(1:T,mean(result.error_test),'r')
set(gca,'box','on','fontname','fixedWidth','fontsize',14)
xlabel('Iteration')
ylabel('Error')
legend('Train','Test')
title('Results')

