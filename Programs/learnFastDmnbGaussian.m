function [alpha_t,mu_t,sigma_t,eta_t,phi_t,gama_t,logProb_time,perplexity_time]=learnFastDmnbGaussian(X,Y,alpha,mu,sigma,eta,maskX)
%
% k = number of clusteres
% c = number of classes
% d = number of features 
% n = number of data points
% 
% Input:
%   alpha:          k*1; Dirichlet distribution
%   mu,sigma:       k*d; Gaussian distribution parameter for d features and k
%                   cluters
%   eta:            k*(c-1), regression parameter for c-1 classes, if
%                   #class=#cluster, c=k;
%   X:              n*d; data matrix.
%   Y:              M*(c-1); each row is the class label for one doc.  
%                   The ith dimension with value 1 indicates the doc class is
%                   i. If all dimensions are 0, the doc class is c.
%   maskX:          n*d; 0-1 matrix with 1 indicating valide entries.
%
% Ouptput:
%   alpha_t:            k*1;
%   mu_t, sigma_t:      k*d;
%   eta_t:              k*(c-1);
%   phi_t,gama_t:       k*n;
%---------------------------------------------------


[n,d] = size(X);
[k,d]=size(mu);

alpha_t=alpha;
mu_t=mu;
sigma_t=sigma;
eta_t=eta;
phi_t=ones(k,n)/k;

clear alpha mu sigma eta;


perplexity_t=1;
phi_tt=[];
gama_tt=[];

logProb_time=[];
perplexity_time=[];


epsilon=0.001;
time=500;

e=100;
t=1;
disp(['Learning Fast DMMNB...'])
while e>epsilon && t<time
    % E-step
    for u=1:n
        x_u=X(u,:);
        y_u=Y(u,:);
        mask_u=maskX(u,:);
        [estimatedGama,estimatedPhi]=fastDmnbGaussianEstep(alpha_t,mu_t,sigma_t,eta_t,phi_t(:,u),x_u,y_u,mask_u);
        % phiAll gammaAll are the parameters for all samples
        phi_tt(:,u)=estimatedPhi;
        gama_tt(:,u)=estimatedGama;      
    end
    
    % compute perplexity
    [logProb_tt,perplexity_tt]=fastDmnbGaussianGetPerp(X,Y,alpha_t,mu_t,sigma_t,eta_t,phi_tt,gama_tt,maskX);
    logProb_time=[logProb_time,logProb_tt];
    perplexity_time=[perplexity_time,perplexity_tt];

    % M-step
    [alpha_tt,mu_tt,sigma_tt,eta_tt]=fastDmnbGaussianMstep(alpha_t,eta_t,phi_tt,gama_tt,X,Y,sigma_t,maskX);
    
    
    % difference from the previous iteration
    if perplexity_tt==Inf||perplexity_t==Inf
        e=100;
    else
        e=abs(perplexity_tt-perplexity_t)/perplexity_t;
    end
    disp(['t=',int2str(t),' error= ',num2str(e), ' perplexity=',num2str(perplexity_tt)]);
    logProb_t=logProb_tt;
    perplexity_t=perplexity_tt;

    
    alpha_t=alpha_tt;
    mu_t=mu_tt;
    sigma_t=sigma_tt;
    eta_t=eta_tt;
    phi_t=phi_tt;
    gama_t=gama_tt;
    
    t=t+1;
    
end