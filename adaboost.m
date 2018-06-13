function result = adaboost(X,y,ntrials,T,train_split)
%ADABOOST Adaptive Weak Learner Boosting
%   ADABOOST(X,Y,NTRIALS,T) runs the AdaBoost algorithm on data set X with
%   labels Y. The number of runs is controlled with NTRIALS and the number
%   of boosting iterations is controlled with T. It is assumed X is a
%   matrix with rows corresponding to observations and columns
%   corresponding to variables. It is assumed the training labels Y are
%   binary and from the set {-1,+1}. For each trial, the data is randomly 
%   split into training and test sets based on TRAIN_SPLIT being the number 
%   of training examples to assign to the training set.
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

%Setup
N = size(X,1);
result.error_train = zeros(ntrials,T);
result.error_test = zeros(ntrials,T);
result.feature_split = zeros(ntrials,T);
result.threshold = zeros(ntrials,T);
result.label_C1 = zeros(ntrials,T);
result.H_train = cell(1,ntrials);
result.H_test = cell(1,ntrials);
result.alpha_vec = cell(1,ntrials);
result.trials = 1:ntrials;
result.iterations = 1:T;

%Run
hw = waitbar(0,'Running AdaBoost');
for kk = 1:ntrials
    
    %Rand Split
    [~,idx] = sort(rand(N,1));
    train_mask = idx(1:train_split);
    test_mask = idx(train_split+1:end);
    data.train = X(train_mask,:);
    data.test = X(test_mask,:);
    label.train = y(train_mask);
    label.test = y(test_mask);
    
    %Algorithm Init
    D = ones(train_split,1)/train_split;
    alpha = zeros(T,1);
    h = cell(1,T);
    H.train = zeros(train_split,T);
    H.test = zeros(N-train_split,T);
    
    %Train
    for t = 1:T
        
        %Split
        S = decstump(data.train,label.train,D);
        e = S.error;
        h{t} = S.h;
        
        %Compute Classifier Weight
        alpha(t) = 0.5*log((1-e)/e);
        
        %Update H Matrices
        H.train(:,t) = h{t}(data.train);
        H.test(:,t) = h{t}(data.test);
        
        %Update Data Weights
        D = D.*exp(-alpha(t)*label.train.*H.train(:,t));
        D = D/sum(D);
        
        %Update Results Structure
        result.error_train(kk,t) = mean(sign(H.train*alpha)~=label.train);
        result.error_test(kk,t) = mean(sign(H.test*alpha)~=label.test);
        result.feature_split(kk,t) = S.feature;
        result.threshold(kk,t) = S.thr;
        result.label_C1(kk,t) = S.C1;
        
    end
    
    %Save Boosting Variables
    result.H_train{kk} = H.train;
    result.H_test{kk} = H.test;
    result.alpha_vec{kk} = alpha;
    
    %Update Waitbar
    waitbar(kk/ntrials,hw,['Running AdaBoost (Trial ' num2str(kk) ')']);
    
end
delete(hw)

function S = decstump(X,y,D)
%DECSTUMP Optimal Decision Stump for AdaBoost Algorithm
%   DECSTUMP(X,Y,D) finds the optimal dimension (i.e. column) of X to split
%   to predict binary label Y \in {-1,+1} based on the data weights D.
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

%Setup
Nf = size(X,2);
th = zeros(1,Nf);
emin = zeros(1,Nf);
C1 = ones(1,Nf);
C2 = ones(1,Nf);

%Loop Over Columns of X
for ii = 1:Nf
    
    cur = X(:,ii);
    vals = unique(cur);
    nvals = length(vals);
    e1 = zeros(nvals,1);
    e2 = zeros(nvals,1);
    
    %Loop Over Unique Values & Compute Error
    for jj = 1:nvals
        y1 = ones(size(cur));
        y2 = ones(size(cur));
        split = cur>=vals(jj); 
        y1(split) = -1;
        y2(~split) = -1;
        e1(jj) = sum(D(y1~=y));
        e2(jj) = sum(D(y2~=y));
    end
    
    %Find Decision Stump Labels C1 and C2
    [e1_min,e1_idx] = min(e1);
    [e2_min,e2_idx] = min(e2);
    if e1_min<e2_min
        th(ii) = vals(e1_idx);
        emin(ii) = e1_min;
        C1(ii) = -1;
    else 
        th(ii) = vals(e2_idx);
        emin(ii) = e2_min;
        C2(ii) = -1;
    end
    
end

%Configure Output Structure
[S.error,f] = min(emin);
S.feature = f;
S.thr = th(f);
S.C1 = C1(f);
S.C2 = C2(f);
S.h = @(Z)S.C1*(Z(:,f)>=S.thr)+S.C2*(Z(:,f)<S.thr);


