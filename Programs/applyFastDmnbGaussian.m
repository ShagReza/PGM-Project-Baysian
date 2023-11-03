function [predY,accuracy,perplexity,phi,gama]=applyFastDmnbGaussian(X,Y,alpha,mu,sigma,eta,maskX)
%
% k = number of clusteres
% d = number of features
% n = number of data points 
% c = number of classes, k=c if #clusters=#classes
%
% Input:
%   X:          n*d, data
%   Y:          M*(c-1); each row is the class label for one doc.  
%               The ith dimension with value 1 indicates the doc class is
%               i. If all dimensions are 0, the doc class is c.
%   alpha:      k*1,
%   mu,sigma:   k*d,
%   eta:        k*(c-1), parameter for regression for c-1 classes
%   
%
% Output:
%   Y:      n*1, the predicted labels for M test docs
%   post:   k*n, posterior
%   phi:        k*n;
%   gama:       k*n;
%-----------------------------------------------------------------

disp(['Applying Fast DMMNB...'])
n=size(X,1);
k=length(alpha);
c=size(Y,2)+1;

% get phi on test data
for u=1:n
    x_u=X(u,:);
    mask_u=maskX(u,:);
    [estimatedGama,estimatedPhi]=fastMnbGaussianEstep(alpha,mu,sigma,x_u,mask_u);
    phi(:,u)=estimatedPhi;
    gama(:,u)=estimatedGama;      
end

% perplexity on test data
[logProb,perplexity]=fastDmnbGaussianGetPerp(X,Y,alpha,mu,sigma,eta,phi,gama,maskX);

% get predicted class on test data
rawY=eta'*phi; %(c-1)*n
rawY=[rawY;zeros(1,n)];  %c*n
post=exp(rawY).*(ones(c,1)*(1./sum(exp(rawY),1)));
[C,predY]=max(rawY',[],2);

% get accuracy
[C,trueY]=max([Y,1-sum(Y,2)],[],2);
accuracy=sum(trueY==predY)/length(predY)
