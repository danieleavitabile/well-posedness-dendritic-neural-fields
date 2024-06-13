%%%%%  This scirpt pots grahs  ol level lines for Example 1
%% Clean
clear all, close all, clc;

%% Model parameters
xi0 = 1; eps = 0.5; 
nu  = 0.1; xStar = 20; c = 0.5;
mu  = 1000; vth = 0.1; kappa = 1;

%% Numerical parameters
nx  = 2^12; Lx  = 24*pi; 
nxi = 2^10; Lxi = 3;
tau = 0.05; nt = 60; iplot = 20; ihist = iplot;

%% Spatial grid
hx  = 2*Lx/nx;  x  = -Lx +[0:nx-1]*hx;
hxi = 2*Lxi/(nxi-1); xi = -Lxi + [0:nxi-1]'*hxi;
[X,XI] = meshgrid(x,xi);

%% Function handles
wFun      = @(x) kappa*0.5*exp(-abs(x)); 
% alphaFun  = @(xi,eps) (eps*sqrt(pi))^-1 * exp( -(xi-xi0).^2/eps^2)...
              % .*(abs(xi - xi0) <= 2*eps);
alphaFun  = @(xi,eps) (eps*sqrt(pi))^-1 * exp( -(xi-xi0).^2/eps^2);
% alphaPFun  = @(xi,eps) (eps*sqrt(pi))^-1 * exp( -xi.^2/eps^2)...  
%                .*(abs(xi) <= 2*eps);
alphaPFun  = @(xi,eps) (eps*sqrt(pi))^-1 * exp( -xi.^2/eps^2);
SFun      = @(v) 1./(1+exp(-mu*(v-vth))); 

%% Precomputing vectors
wHat = fft(wFun(x));
alpha  = alphaFun(xi,eps);  J  = find(alpha  ~= 0); alpha  = alpha(J);
alphaP = alphaPFun(xi,eps); JP = find(alphaP ~= 0); alphaP = alphaP(JP);

%% Quadrature weights
sigma = ones(nxi,1); sigma([1 nxi]) = 0.5; sigma = hxi*sigma; sigma = sigma(JP);
%%%% change nu
for inu=0:1
    nu=inu*0.1
    
%% Differentiation matrix and linear operator
e = ones(nx,1); D2 = spdiags([e -2*e e], -1:1, nxi, nxi); 
D2(1,2) = 2; D2(nxi,nxi-1) = 2; D2 = D2/hxi^2;
A  = (1+tau*c)*speye(nxi) - tau*nu*D2;
dA = decomposition(A);
%% Initial condition
V = ((1 - 1./(1+exp(-5*(X-xStar)))).*(X>0) + (1 - 1./(1+exp(5*(X+xStar)))).*(X<=0));
V=V.*alphaPFun(xi,eps);
%V = 0.5*V;
%V=ones(nxi,nx);
% sol = load('waveProfile.mat'); V = 5*sol.V;
t = 0;


%% TimeStep 
for it = 1:nt
  % Compute F
  F = zeros(nxi,nx);
  S = (alphaP .* sigma)'*SFun(V(JP,:));
  conv = hx*ifftshift(real(ifft(fft(S) .* wHat)));
  F(J,:) = alpha * conv;

  % Update V
  V = dA\(V + tau*F);
   t = t + tau;
time(it)=t;
Vmax(it)=max(max(abs(V)));
  % Update plot
  if mod(it,iplot)== 0  
      ns=it/iplot
      subplot(3,2,2*(ns-1)+inu+1);
    imagesc(x,xi,V);
    caxis([0,0.9]);
colorbar
xlabel('x')
ylabel('xi')
title(['t=', num2str(ns), '  nu=', num2str(nu)])
   % xc=nx/2;
 % V1= V(1:nxi, xc);
  %plot(xi,V1);
 % pause
  end

  % Save
 %%   VHist = [VHist; t, V(:)'];
 % end

end

clear dA; save('savedData.mat');
end
