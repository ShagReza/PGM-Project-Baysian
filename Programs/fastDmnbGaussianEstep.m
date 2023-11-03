function [gamma,phi]=fastDmnbGaussianEstep(alpha,mu,sigma,eta,phi_last,x,y,maskx)
%
% variational E-step.
%
% k: number of components(clusters).
% c: number of classes.
% d: length of each feature vector.
% n: number of data points.
%
% Input:
%   alpha:      k*1, Dirichlet distribution paramter. 
%   mu,sigma:   k*d, parameters for d features and k clusters
%   eta:        k*(c-1), regression parameter for c-1 classes, if
%               #class=#cluster, c=k;
%   phi_last:   k*1, estimation of phi from the last iteration
%   x:          1*d, input data vector. 
%   y:          1*(c-1), 0-1 vector, the ith dimension with value 1
%               indicates the class i. If all dimensions are 0, the class is c
%   maskx:      1*d; 0-1 vector with 1 indicating valide entries.
% 
% Output:
%   gamma:      k*1, variational parameters for to generate pi - mixing weight. 
%   phi:        k*1, variational parameters to generate z--the class.
%-----------------------------------------------------------------------

%=================initilization==================
[k,d]=size(mu);
m=length(find(maskx>0));
nc=length(y)+1;

phi_t=ones(k,1)/k;
gamma_t=alpha+m*phi_t;


%=================iteration======================
epsilon=0.001;
time=500;

e=100;
t=1;

temp=-(ones(k,1)*x-mu).^2./(2*sigma)-log(sqrt(2*pi*sigma));
xi=1+sum(sum(phi_last*ones(1,nc-1).*exp(eta)));

while e>epsilon && t<time
        
    % phi
    phi_tt=exp(psi(gamma_t)-psi(sum(gamma_t))+sum(temp.*(ones(k,1)*maskx),2)/m+sum(eta.*(ones(k,1)*y),2)/m-sum(exp(eta),2)/(m*xi));
    phi_tt=(phi_tt+realmin)/sum(phi_tt+realmin); 
    
    % xi
    xi=1+sum(sum(phi_tt*ones(1,nc-1).*exp(eta)));
    
    % gamma
    gamma_tt=alpha+m*phi_tt;
    
    % error
    e1=sum(abs(phi_t-phi_tt))/sum(sum(phi_t));
    e2=sum(abs(gamma_t-gamma_tt))/sum(gamma_t);
    e=max(e1,e2);
    
    phi_t=phi_tt;
    gamma_t=gamma_tt;
    t=t+1;
end

gamma=gamma_t;
phi=phi_t;
