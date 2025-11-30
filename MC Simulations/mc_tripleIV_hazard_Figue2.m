function [BiasA, BiasB] = mc_tripleIV_hazard_Figue2()
% Monte Carlo: Triple-Interaction IV for Fund Flows -> Liquidation Risk
% Discrete-time hazard with complementary log-log link ('comploglog') and endogenous flows.
% Instrument: Z_it = Post_t × Exposure_i × Flex_i.
%
% Estimators compared:
%   (A) Naive comploglog (no IV)
%   (B) Control-function comploglog (IV) using first-stage residuals

rng(20250928,'twister');

%% -------------------- Simulation settings --------------------
M      = 100;     % Monte Carlo replications
N      = 1200;    % funds
T      = 60;      % months
t0     = 10;      % policy start month (index)
useRamp = true;   % FATCA-like ramp over [t0, t0+12] then 1

% True hazard parameters (comploglog scale)
beta0   = -6.50;   % intercept
b_flow  = -2.40;   % TRUE causal effect of Flow on liquidation hazard
b_ret   = -1.20;   % performance lowers hazard
b_vol   = +0.60;   % volatility raises hazard
b_lnAUM = -0.15;   % size lowers hazard

% First-stage Flow DGP
pi0     =  0.00;
piZ     =  0.80;   % instrument strength for Z = Post × Exposure × Flex
piX     = [0.50, -0.20, 0.10];  % [ret, vol, lnAUM] effects in first stage
rho_U   =  0:0.1:1;   % unobserved factor loading in Flow (endogeneity)
sigma_e =  0.60;   % Flow shock sd
gama_U  = 0:0.1:1;    % unobserved frailty loading in hazard (endogeneity)
% Exposure/Flex cross-sectional distributions
aExp = 2.0; bExp = 3.0;  % Exposure ~ Beta(2,3), mean ~ 0.40
aFlx = 4.0; bFlx = 2.5;  % Flex     ~ Beta(4,2.5), mean ~ 0.62

% Controls persistence
rho_ret = 0.60; rho_vol = 0.50; rho_aum = 0.90;

alpha  = 0.05;   % 95% CI

vec = @(X) X(:);

%% -------------------- Storage --------------------
estA = nan(M,1); seA = nan(M,1);   % Naive coefficient & its SE (Flow)
estB = nan(M,1); seB = nan(M,1);   % IV CF coefficient & its SE (Flow)
F1   = nan(M,1);                   % first-stage partial F for Z

%% -------------------- Monte Carlo loop --------------------
for i = 1:length(rho_U)
    i
    for j = 1:length(gama_U)
        j
        for m = 1:M
            % ---- Cross-section: Exposure & Flex (fixed by fund) ----
            Exposure = betarnd(aExp,bExp,[N,1]);   % e.g., offshore/US-LP share proxy
            Flex     = betarnd(aFlx,bFlx,[N,1]);   % contractual liquidity [0,1]
            
            % ---- Time: Post_t (FATCA-like ramp or step) ----
            Post = zeros(T,1);
            if useRamp
                rampLen = 12; % FATCA-like ramp over [t0, t0+12] then 1
                for t = 1:T
                    if t < t0
                        Post(t) = 0;
                    elseif t <= t0 + rampLen
                        Post(t) = (t - t0 + 1) / (rampLen + 1);
                    else
                        Post(t) = 1;
                    end
                end
            else
                Post(t0:end) = 1;
            end
        
            % ---- Instrument: Z_it = Post_t × Exposure_i × Flex_i (outer product) ----
            % N×1 times 1×T -> N×T
            Z = (Exposure .* Flex) * Post(:)';
        
            % ---- Controls: ret, vol, lnAUM (panel) then standardize ----
            [ret, vol, lnAUM] = simulate_controls(N,T,rho_ret,rho_vol,rho_aum);
        
            % ---- Unobserved factor driving both Flow and hazard (endogeneity) ----
            U = 0.7*randn(N,1)*ones(1,T) + 0.3*randn(N,T);
        
            % ---- First-stage DGP: Flow_it ----
            e    = sigma_e*randn(N,T);
            Flow = pi0 + piZ*Z + (piX(1)*ret + piX(2)*vol + piX(3)*lnAUM) + rho_U(i)*U + e;
        
            % ---- Hazard DGP (comploglog) ----
            theta = gama_U(j);                     % U enters hazard too
            FE_t  = 0.10*standardize((1:T));  % mild quarter FE
            eta = beta0 ...
                + b_flow*Flow ...
                + b_ret*ret + b_vol*vol + b_lnAUM*lnAUM ...
                + theta*U ...
                + (ones(N,1)*FE_t);
            p_liq   = 1 - exp(-exp(eta));     % inverse comploglog
            Liquid  = (rand(N,T) < p_liq);    % Bernoulli outcome
        
            % ---- Stack panel to vectors ----
            y    = vec(Liquid);
            Fv   = vec(Flow);
            Xret = vec(ret); Xvol = vec(vol); Xaum = vec(lnAUM);
            Ziv  = vec(Z);
        
            % ---- Quarter fixed effects (omit first) ----
            TFE  = eye(T);
            TFE  = TFE(:,2:end);                 % T × (T-1)
            Tbig = kron(TFE, ones(N,1));         % (N*T) × (T-1)
        
            % ================= First stage: OLS with time FE =================
            % Full model: Flow ~ Z + controls + time FE
            Xfs_full = [Ziv, ones(N*T,1), Xret, Xvol, Xaum, Tbig];
            b1  = Xfs_full \ Fv;
            r1  = Fv - Xfs_full*b1;
        
            % Restricted: Flow ~ controls + time FE (drop Z)
            Xfs_rest = [ones(N*T,1), Xret, Xvol, Xaum, Tbig];
            b1r = Xfs_rest \ Fv;
            rR  = Fv - Xfs_rest*b1r;
        
            % Partial F for Z | controls, FE
            SSR_full  = sum(r1.^2);
            SSR_rest  = sum(rR.^2);
            q = 1; n = size(Xfs_full,1); k = size(Xfs_full,2);
            F1(m) = ((SSR_rest - SSR_full)/q) / (SSR_full/(n - k));
        
            % ================= (A) Naive comploglog (no IV) =================
            XA = [Fv, Xret, Xvol, Xaum, Tbig];
            [bA_vec, seA_vec] = comploglog_glm(y, XA);
            estA(m) = bA_vec(1);      % Flow coefficient (scalar)
            seA(m)  = seA_vec(1);     % SE(Flow) (scalar)
        
            % ================= (B) CF comploglog (IV) =======================
            % Add first-stage residual as control
            res_within = r1;
            XB = [Fv, res_within, Xret, Xvol, Xaum, Tbig];
            [bB_vec, seB_vec] = comploglog_glm(y, XB);
            estB(m) = bB_vec(1);      % Flow coefficient (scalar)
            seB(m)  = seB_vec(1);     % SE(Flow) (scalar)
        end
        BiasA(i,j) = mean(estA)-b_flow;
        BiasB(i,j) = mean(estB)-b_flow;
    end
end

[X, Y] = meshgrid(rho_U,gama_U);
figure(1);
surf(X, Y, BiasA, 'FaceColor', 'blue', 'FaceAlpha', 0.7, 'EdgeColor', 'none');
hold on;
surf(X, Y,BiasB, 'FaceColor', 'green', 'FaceAlpha', 0.7, 'EdgeColor', 'none');
hold off
xlabel('Endogeneity Loading');
ylabel('Frailty Loading');
zlabel('Bias in $\beta_{flow}$', 'Interpreter', 'latex', 'FontSize', 12);
legend('Naive', 'IV');
title('Figure 2a: Comparison of Bias: Naive vs. IV Estimator');

figure(2)
surf(X, Y, BiasA - BiasB);
xlabel('Endogeneity Loading');
ylabel('Frailty Loading');
zlabel('Bias Difference in $\beta_{flow}$ (Naive - IV)', 'Interpreter', 'latex', 'FontSize', 12);
title('Figure 2b: Excess Bias of Naive Estimator over IV');
colorbar;
end

function [ret, vol, lnAUM] = simulate_controls(N,T,rho_ret,rho_vol,rho_aum)
% AR(1)-style panels, then standardize to mean 0 / sd 1
ret   = zeros(N,T); vol = zeros(N,T); lnAUM = zeros(N,T);
ret(:,1)   = 0.02 + 0.04*randn(N,1);
vol(:,1)   = 0.20 + 0.10*randn(N,1);
lnAUM(:,1) = 4.50 + 0.50*randn(N,1);
for t=2:T
    ret(:,t)   = 0.02 + rho_ret*ret(:,t-1)   + 0.05*randn(N,1);
    vol(:,t)   = 0.20 + rho_vol*vol(:,t-1)   + 0.10*randn(N,1);
    lnAUM(:,t) = 4.50 + rho_aum*lnAUM(:,t-1) + 0.20*randn(N,1);
end
% Clip / standardize
vol = max(0.02, vol);
ret   = standardize(ret);
vol   = standardize(vol);
lnAUM = standardize(lnAUM);
end

function Xs = standardize(X)
mu = mean(X(:)); sd = std(X(:));
Xs = (X - mu) / max(sd,1e-9);
end

function [b, se] = comploglog_glm(y, X)
% GLM (binomial) with complementary log-log link. Returns beta and SE vector.
[b, ~, stats] = glmfit(X, y, 'binomial', 'link','comploglog', 'constant','off');
se = stats.se;
end

