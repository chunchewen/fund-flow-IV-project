
%  The code conducts the discrete hazard regression with twoWay clust
%    • Cloglog with time FE 
%    • Two-way clustered SEs (fund, time)
%    • Average Marginal Effects (AME) for key explantory variables
%
%  Inputs 
%    performance_with_instrument.csv
%    Flow_spec_fitted_within.csv      
%  Outputs:
%    - hazard_merged.csv
%    - hazard_panel.csv
%    - hazard_coef_twoWay.csv
%    - hazard_ame.csv
%    - hazard_summary.txt

clear; clc;
perf_csv  = 'C:\Users\ias05106\OneDrive - University of Strathclyde\Desktop\LipperTASS2018\FundFlowWork\Revision2025\RevisedIVdummy\performance_with_instruments.csv';   % base panel
fitfile   = 'C:\Users\ias05106\OneDrive - University of Strathclyde\Desktop\LipperTASS2018\FundFlowWork\Revision2025\RevisedIVdummy\Flow_spec_CFTConly_fitted_within.csv';  % from IVpanelRegressionD2_FATCAonly.m script

use_quarter_FE = false;      % false = Month FE
link_choice    = 'cloglog';  % cloglog link function for discrete hazaard regression

winsor_p1  = 1; winsor_p99 = 99;

% Load performance
if exist(perf_csv, 'file')
    P = readtable(perf_csv);
elseif exist(perf_xlsx, 'file')
    P = readtable(perf_xlsx);
else
    error('Performance file not found (.csv or .xlsx).');
end
assert(all(ismember({'FundID','Year','Month'}, P.Properties.VariableNames)), ...
    'Performance data must include FundID, Year, Month.');

numLikely = intersect({'Return','AUM','FundFlow','Event'}, P.Properties.VariableNames);
for j=1:numel(numLikely)
    v = numLikely{j};
    if ~isnumeric(P.(v)), P.(v) = str2double(string(P.(v))); end
end

% Month index
ym = @(Y,M) Y*12+M;
P.m_id = ym(P.Year, P.Month);
P = sortrows(P, {'FundID','m_id'});

% Load first-stage fitted/residuals 
assert(exist(fitfile,'file')==2, 'First-stage fitted file not found: %s', fitfile);
F = readtable(fitfile);
needF = {'FundID','Year','Month','yhat_within','res_within'};
assert(all(ismember(needF, F.Properties.VariableNames)), ...
    'Fitted file must have: FundID, Year, Month, yhat_within, res_within');

% Merge
M = outerjoin(P, F(:,needF), 'Keys', {'FundID','Year','Month'}, 'MergeKeys', true, 'Type', 'left');
M = sortrows(M, {'FundID','Year','Month'});

% ----------------- CONTROLS & FLOWS -----------------
% Return lag
M.ret_L1 = lagWithinFund(M, M.Return, 1);

% lnAUM and its lag
M.lnAUM = nan(height(M),1);
ok = isfinite(M.AUM) & M.AUM>0;
M.lnAUM(ok) = log(M.AUM(ok));
M.lnAUM_L1 = lagWithinFund(M, M.lnAUM, 1);

% Volatility: 12-month rolling std of Return, lagged
M.vol_roll = rollingStdWithinFund(M, M.Return, 12);
M.vol_L1   = lagWithinFund(M, M.vol_roll, 1);

% Flows: use FundFlow if present, else construct
if ismember('FundFlow', M.Properties.VariableNames) && any(~isnan(M.FundFlow))
    M.Flow = double(M.FundFlow);
else
    AUM_L1  = lagWithinFund(M, M.AUM, 1);
    M.Flow  = (M.AUM - AUM_L1 .* (1 + M.Return)) ./ AUM_L1;
end

% Winsorize to control outliers
M.ret_L1   = winsorVec(M.ret_L1, winsor_p1, winsor_p99);
M.vol_L1   = winsorVec(M.vol_L1, winsor_p1, winsor_p99);
M.lnAUM_L1 = winsorVec(M.lnAUM_L1, winsor_p1, winsor_p99);
M.Flow     = winsorVec(M.Flow, winsor_p1, winsor_p99);

% Ensure Event is binary 0/1
if ~ismember('Event', M.Properties.VariableNames)
    error('Event variable missing (0/1 liquidation indicator).');
end
M.Event = double(M.Event>0);

% HAZARD PANEL 
% Keep only up to first failure per fund (drop post-failure months)
M = sortrows(M, {'FundID','m_id'});
uF = unique(M.FundID);
keep = true(height(M),1);
for f = uF.'
    ix = find(M.FundID==f);
    k  = find(M.Event(ix)==1, 1, 'first');
    if ~isempty(k), keep(ix(k+1:end)) = false; end
end
M = M(keep,:);

% Save merged pre-estimation panel
writetable(M, 'hazard_merged_CFTC.csv');

% Drop missing key regressors
keyvars = {'Flow','res_within','ret_L1','vol_L1','lnAUM_L1','Event'};
ok = true(height(M),1);
for j=1:numel(keyvars)
    if ~isnumeric(M.(keyvars{j})) || all(isnan(M.(keyvars{j})))
        M.(keyvars{j}) = NaN(size(M,1),1);
    end
    ok = ok & ~isnan(M.(keyvars{j}));
end
H = M(ok,:);
writetable(H, 'hazard_panel_CFTC.csv');

H.m_id = ym(H.Year, H.Month);
if use_quarter_FE
    H.t_id = H.Year*4 + ceil(H.Month/3);
    tlabel = 'quarter';
else
    H.t_id = H.m_id;
    tlabel = 'month';
end

[t_codes, t_list] = grp2idx(H.t_id);
Gt = max(t_codes);
Dt = sparse(1:numel(t_codes), t_codes, 1, numel(t_codes), Gt);
Dt(:,1) = [];   % drop base FE

% Core regressors
Xcore = [ones(height(H),1), H.Flow, H.res_within, H.ret_L1, H.vol_L1, H.lnAUM_L1];
core_names = {'Intercept','Flow','res_within','ret_L1','vol_L1','lnAUM_L1'};
X = [sparse(Xcore), Dt];
y = H.Event;

% ESTIMATION 
switch lower(link_choice)
    case 'logit'
        [beta, mu, gprime] = irls_binomial_sparse(X, y, @logit_link);
    case 'cloglog'
        [beta, mu, gprime] = irls_binomial_sparse(X, y, @cloglog_link);
    otherwise
        error('Unknown link_choice: %s', link_choice);
end

% Two-way cluster SEs (fund, time)
[f_codes, ~] = grp2idx(H.FundID);
Vf = cluster_sandwich_binomial(X, y, mu, f_codes);
Vt = cluster_sandwich_binomial(X, y, mu, t_codes);
Vp = cluster_sandwich_binomial(X, y, mu, grp2idx(string(H.FundID)+"_"+string(t_codes)));
V2 = Vf + Vt - Vp;

se = sqrt(diag(V2));
z  = beta ./ se;
p  = 2*normcdf(-abs(z));

% Names
fe_names = strcat("FE_", tlabel, "_", string(t_list(2:end)'));
names = [core_names, cellstr(fe_names)]';

% Coef table
Tcoef = table(names, beta, se, z, p, 'VariableNames', {'Variable','Beta','SE_twoWay','z','p'});
writetable(Tcoef, 'hazard_coef_twoWay_CFTC.csv');

% AVERAGE MARGINAL EFFECTS (AME) 
k_core = numel(core_names);
AME = zeros(k_core,1);
SE_AME = zeros(k_core,1);

gbar = mean(gprime);
for j=1:k_core
    b_j = beta(j);
    AME(j) = mean(gprime .* b_j);
    SE_AME(j) = abs(gbar) * se(j);   % delta approx
end

Tame = table(core_names', AME, SE_AME, 'VariableNames', {'Variable','AME','AME_SE_approx'});
writetable(Tame, 'hazard_ame_CFTC.csv');

%% ----------------- SUMMARY TXT -----------------
fid = fopen('hazard_summary.txt','w');
fprintf(fid, 'Discrete-time hazard GLM (%s link), %s FE, two-way clustered SEs (fund,time)\n', ...
    upper(link_choice), tlabel);
fprintf(fid, 'N obs = %d, N funds = %d, N %s bins = %d\n', numel(y), numel(unique(H.FundID)), tlabel, Gt);
fprintf(fid, '\nCore coefficients (two-way SEs):\n');
for j=1:k_core
    fprintf(fid, '  %-10s  % .6f   (SE=%.6f)   z=%.2f   p=%.4g\n', core_names{j}, beta(j), se(j), z(j), p(j));
end
fprintf(fid, '\nAverage Marginal Effects (AME):\n');
for j=1:k_core
    fprintf(fid, '  %-10s  % .6f   (SE≈%.6f)\n', core_names{j}, AME(j), SE_AME(j));
end
fclose(fid);

function vLag = lagWithinFund(T, v, L)
    vLag = NaN(height(T),1);
    uF = unique(T.FundID);
    for f = uF.'
        ix = find(T.FundID==f);
        Luse = min(L, numel(ix));
        vLag(ix) = [NaN(Luse,1); v(ix(1:end-Luse))];
    end
end

function vRoll = rollingStdWithinFund(T, v, W)
    vRoll = NaN(height(T),1);
    uF = unique(T.FundID);
    for f = uF.'
        ix = find(T.FundID==f);
        vv = v(ix);
        vRoll(ix) = movstd(vv, [W-1 0], 'omitnan');
    end
end

function xw = winsorVec(x, p1, p99)
    xw = x;
    ok = isfinite(x);
    if nnz(ok) < 5, return; end
    lo = prctile(x(ok), p1); hi = prctile(x(ok), p99);
    xw(ok & x < lo) = lo; xw(ok & x > hi) = hi;
end

function [beta, mu, gprime] = irls_binomial_sparse(X, y, linkfun)
    % IRLS for GLM with binomial family and generic link
    p = size(X,2);
    beta = zeros(p,1);
    tol = 1e-8; maxit = 200;
    for it=1:maxit
        [mu, eta, gprime] = linkfun(X*beta);
        V = max(mu .* (1-mu), eps);      % binomial variance function
        W = (gprime.^2) ./ V;            % GLM weights
        z = eta + (y - mu) ./ max(gprime, eps); % working response
        WX  = bsxfun(@times, X, W);
        XtWX = X' * WX;
        XtWz = X' * (W .* z);
        ridge = 1e-8 * speye(p);         % small ridge for stability
        beta_new = (XtWX + ridge) \ XtWz;
        if norm(beta_new - beta) < tol*(1+norm(beta)), beta = beta_new; break; end
        beta = beta_new;
    end
    [mu, eta, gprime] = linkfun(X*beta);
end

function [mu, eta, gprime] = logit_link(eta)
    mu = 1 ./ (1 + exp(-eta));
    gprime = mu .* (1 - mu);   % dμ/dη
end

function [mu, eta, gprime] = cloglog_link(eta)
    % mu = 1 - exp(-exp(eta))
    % dmu/deta = exp(eta) .* exp(-exp(eta))   (element-wise)
    mu = 1 - exp(-exp(eta));
    gprime = exp(eta) .* exp(-exp(eta));
end

function V = cluster_sandwich_binomial(X, y, mu, cluster_id)
    % Cluster-robust sandwich for GLM with binomial variance.
    % Use the quasi-score U_i = x_i * (y_i - mu_i).
    Vfun = mu .* (1 - mu);
    Bread = pinv(full( X' * (bsxfun(@times, X, Vfun)) ));
    r = (y - mu);
    clusters = unique(cluster_id);
    p = size(X,2);
    Meat = zeros(p,p);
    for g = clusters.'
        idx = (cluster_id==g);
        Xg  = X(idx,:);
        rg  = r(idx);
        Ug  = Xg' * rg;
        Meat = Meat + (Ug * Ug');
    end
    V = Bread * Meat * Bread;
end
