%% This code conduct the first-stage IV regression based on windowed policy period from 1997 to 2015

clear; clc;
run_Flow_t   = true;     
lead_horizons = [6,12];
ym = @(Y,M) Y*12 + M;  

T = readtable('C:\Users\ias05106\OneDrive - University of Strathclyde\Desktop\LipperTASS2018\FundFlowWork\Revision2025\RevisedIVdummy\performance_with_instruments.csv');
% [FundID	Year	Month	Day	Event	Return	NAV	AUM	FundFlow	Flow_YM	Flow_YM_SP	Flow_YM_DP	
% Z_CFTC_q	Z_FATCA_q	Z_TaxWin1997_A	Z_TaxWin2003_A	Z_TaxWin2013_A	Z_TaxWin1997_C	Z_TaxWin2003_C	Z_TaxWin2013_C	
% Z_TaxWin1997_A_q	Z_TaxWin2003_A_q	Z_TaxWin2013_A_q	Z_TaxWin1997_C_q	Z_TaxWin2003_C_q	Z_TaxWin2013_C_q]

T(T.Year <= 1996 | T.Year >= 2015, :)=[];  % This is windowed estimation

assert(all(ismember({'FundID','Year','Month'}, T.Properties.VariableNames)), ...
    'Input must contain FundID, Year, Month.');

% Keys & ordering
T.m_id = T.Year*12 + T.Month;
T = sortrows(T, {'FundID','m_id'});
numLikely = intersect({'Return','AUM','FundFlow'}, T.Properties.VariableNames);
for j=1:numel(numLikely)
    if ~isnumeric(T.(numLikely{j})), T.(numLikely{j}) = str2double(string(T.(numLikely{j}))); end
end

% Identify columns
ret_col = 'Return'; assert(ismember(ret_col, T.Properties.VariableNames), 'Return missing');
aum_col = 'AUM';    assert(ismember(aum_col, T.Properties.VariableNames), 'AUM missing');

% Build lnAUM, lags, vol,etc
T.lnAUM = nan(height(T),1);
ok = (T.(aum_col)>0) & isfinite(T.(aum_col));
T.lnAUM(ok) = log(T.(aum_col)(ok));
T.ret_L1    = lagWithinFund(T, T.(ret_col), 1);
T.lnAUM_L1  = lagWithinFund(T, T.lnAUM, 1);

W = 12;
T.vol_roll  = rollingStdWithinFund(T, T.(ret_col), W);
T.vol_L1    = lagWithinFund(T, T.vol_roll, 1);

% Flows
if ismember('FundFlow', T.Properties.VariableNames) && any(~isnan(T.FundFlow))
    T.Flow = double(T.FundFlow);
else
    AUM_L1  = lagWithinFund(T, T.(aum_col), 1);
    T.Flow  = (T.(aum_col) - AUM_L1 .* (1 + T.(ret_col))) ./ AUM_L1;
end
T.Flow_L1 = lagWithinFund(T, T.Flow, 1);

% Winsorize
T.ret_L1    = winsorVec(T.ret_L1, 1, 99);
T.vol_L1    = winsorVec(T.vol_L1, 1, 99);
T.lnAUM_L1  = winsorVec(T.lnAUM_L1, 1, 99);
T.Flow      = winsorVec(T.Flow, 1, 99);
T.Flow_L1   = winsorVec(T.Flow_L1, 1, 99);

% Fixed Effect
T.m_id_L1 = lagWithinFund(T, T.m_id, 1);
T.fe1 = grp2idx(string(T.FundID));
T.fe2 = grp2idx("m_" + string(T.m_id_L1));  % month-year FE at t-1

% IV candidates
zcols = T.Properties.VariableNames(startsWith(T.Properties.VariableNames,'Z_'));
assert(~isempty(zcols), 'No Z_* instruments found.');

% Preferred IV set
preferred_ivars = {'Z_CFTC_q','Z_FATCA_q','Z_TaxWin1997_A_q','Z_TaxWin2003_A_q','Z_TaxWin2013_A_q'};
fallback_tax_C  = {'Z_TaxWin1997_C_q','Z_TaxWin2003_C_q','Z_TaxWin2013_C_q'};

ivars = preferred_ivars(ismember(preferred_ivars, T.Properties.VariableNames));
if numel(ivars) < 5
    taxC = fallback_tax_C(ismember(fallback_tax_C, T.Properties.VariableNames));
    ivars = union(intersect({'Z_CFTC_q','Z_FATCA_q'}, zcols), taxC, 'stable');
end
if ~any(strcmp(ivars,'Z_CFTC_q')) && ismember('Z_CFTC_q', zcols), ivars = ['Z_CFTC_q', ivars]; end
if ~any(strcmp(ivars,'Z_FATCA_q')) && ismember('Z_FATCA_q', zcols), ivars = ['Z_FATCA_q', ivars]; end

% Controls 
controls_all = {'ret_L1','vol_L1','lnAUM_L1'};   
controls = {};
for c = 1:numel(controls_all)
    v = controls_all{c};
    if ismember(v, T.Properties.VariableNames) && nnz(~isnan(T.(v))) >= 100
        controls{end+1} = v; 
    end
end
if isempty(controls)
    controls = {'ret_L1','vol_L1','lnAUM_L1'};
end
fprintf('Controls used: %s\n', strjoin(controls, ', '));
fprintf('IVs used: %s\n', strjoin(ivars, ', '));

% -------- Window and Pre-period functions --------
winfun = struct();
winfun.CFTC  = @(m_id) m_id >= ym(2013,1);
winfun.FATCA = @(m_id) m_id >= ym(2014,7) & m_id <= ym(2015,12);
winfun.TAX97 = @(m_id) m_id >= ym(1997,5) & m_id <  ym(2003,5);
winfun.TAX03 = @(m_id) m_id >= ym(2003,5) & m_id <  ym(2013,1);
winfun.TAX13 = @(m_id) m_id >= ym(2013,1);

prefun = struct();
prefun.CFTC  = @(m_id) m_id < ym(2013,1);
prefun.FATCA = @(m_id) m_id < ym(2014,7);
prefun.TAX97 = @(m_id) m_id < ym(1997,5);
prefun.TAX03 = @(m_id) m_id < ym(2003,1);   % tightened to avoid early-2003 anticipation
prefun.TAX13 = @(m_id) m_id < ym(2013,1);

function run_spec(T, yvar, ivars, controls, tag, lead_horizons, winfun, prefun, ym)
    X = T;  % local copy
    % Align IV timing if y=Flow_L1
    iv_use = ivars;
    if strcmpi(yvar,'Flow_L1')
        Zall = X.Properties.VariableNames(startsWith(X.Properties.VariableNames,'Z_'));
        for ii=1:numel(Zall)
            z = Zall{ii};
            X.([z '_L1']) = lagWithinFund(X, X.(z), 1);
        end
        for j=1:numel(iv_use), iv_use{j} = [iv_use{j} '_L1']; end
        fprintf('[%s] Using lagged instruments (Z_{t-1}).\n', tag);
    else
        fprintf('[%s] Using contemporaneous instruments (Z_t).\n', tag);
    end

    % Keep complete rows
    Xnames = [iv_use(:); controls(:)];
    ok = ~isnan(X.(yvar)) & X.fe2>0 & ~isnan(X.m_id_L1);
    for j=1:numel(Xnames)
        if ismember(Xnames{j}, X.Properties.VariableNames)
            ok = ok & ~isnan(X.(Xnames{j}));
        else
            ok = ok & false;
        end
    end
    D = X(ok,:);
    if height(D) < 1000
        warning('[%s] Too few rows after filtering (%d). Skipped.', tag, height(D));
        return;
    end

    % Absorb FE
    y = D.(yvar);
    Xmat = D{:, Xnames};
    [y_til, X_til] = absorb_two_way(y, Xmat, D.fe1, D.fe2);

    % Drop near-constant & linearly dependent columns via QR
    [Q,R,E] = qr(X_til,0);
    diagR = abs(diag(R));
    tol_qr = max(size(X_til))*eps(max(diagR));
    r = sum(diagR > tol_qr);
    keep_cols = false(size(E)); if r>0, keep_cols(E(1:r))=true; end
    X_use = X_til(:, keep_cols);
    names_use = Xnames(keep_cols);
    b_use = R(1:r,1:r) \ (Q(:,1:r)' * y_til);
    e_til = y_til - X_use*b_use;

    % VCOVs (two-way cluster); use provided XtXinv from QR block
    n = size(X_use,1); k = size(X_use,2);
    s2 = (e_til'*e_til)/max(1,n-k);
    R11 = R(1:r,1:r); 
    if r==0
        warning('[%s] No rank in design.', tag);
        return;
    end
    R11inv = inv(R11);
    XtXinv = R11inv*R11inv';
    V_ols = s2*XtXinv;
    V_f = cluster_vcov(X_use, e_til, D.fe1, XtXinv);
    V_m = cluster_vcov(X_use, e_til, D.m_id_L1, XtXinv);
    V_p = cluster_vcov(X_use, e_til, grp2idx(string(D.fe1)+"_"+string(D.m_id_L1)), XtXinv);
    V_2 = V_f + V_m - V_p;

    Res = table(names_use(:), b_use, sqrt(diag(V_ols)), sqrt(diag(V_f)), sqrt(diag(V_m)), sqrt(diag(V_2)), ...
        'VariableNames', {'Variable','Beta','SE_noncl','SE_cluster_fund','SE_cluster_month','SE_twoWay'});
    writetable(Res, [tag '_coef.csv']);

    % -------- Block F-tests (stable Wald) --------
    blocks = struct('CFTC',[],'FATCA',[],'TAX97',[],'TAX03',[],'TAX13',[]);
    for i=1:numel(names_use)
        nm = names_use{i};
        if contains(nm,'CFTC'),  blocks.CFTC(end+1)=i; end 
        if contains(nm,'FATCA'), blocks.FATCA(end+1)=i; end 
        if contains(nm,'TaxWin1997'), blocks.TAX97(end+1)=i; end 
        if contains(nm,'TaxWin2003'), blocks.TAX03(end+1)=i; end 
        if contains(nm,'TaxWin2013'), blocks.TAX13(end+1)=i; end 
    end
    BT = table(); df2 = max(1,n-k);
    flds = fieldnames(blocks);
    for bb=1:numel(flds)
        bnm = flds{bb};
        idx = blocks.(bnm);
        if isempty(idx)
            BT = [BT; {string(bnm), 0, NaN, NaN, NaN, NaN, NaN}]; 
        else
            [Fstat, pF, chi2, pChi, q_eff] = wald_block_stable(b_use, V_2, idx, n, k);
            R2p = (Fstat*q_eff)/(Fstat*q_eff + df2);
            BT = [BT; {string(bnm), q_eff, Fstat, pF, chi2, pChi, R2p}]; 
        end
    end
    BT.Properties.VariableNames = {'block','q_params','F_twoWay','p_F','Chi2','p_Chi2','partial_R2_approx'};
    writetable(BT, [tag '_block_tests.csv']);

    % Windowed F-tests  
    WF = table('Size',[0 3],'VariableTypes',{'string','double','double'},'VariableNames',{'block','F_twoWay','p_F'});
    bnames = {'CFTC','FATCA','TAX97','TAX03','TAX13'};
    for bb=1:numel(bnames)
        bnm = bnames{bb};
        iv_match = find(contains(names_use, bnm, 'IgnoreCase', true), 1);
        if isempty(iv_match), continue; end
        ivb = names_use{iv_match};
        m = D.m_id;
        if      strcmp(bnm,'CFTC'),  maskW = (m >= ym(2013,1));
        elseif  strcmp(bnm,'FATCA'), maskW = (m >= ym(2014,7)) & (m <= ym(2015,12));
        elseif  strcmp(bnm,'TAX97'), maskW = (m >= ym(1997,5)) & (m < ym(2003,5));
        elseif  strcmp(bnm,'TAX03'), maskW = (m >= ym(2003,5)) & (m < ym(2013,1));
        elseif  strcmp(bnm,'TAX13'), maskW = (m >= ym(2013,1));
        else, maskW = false(size(m));
        end
        W = D(maskW, :);
        if height(W) < 200, continue; end
        y_w = W.(yvar); X_w = [W.(ivb), W{:, intersect(names_use, controls, 'stable')}];
        [y_wt, X_wt] = absorb_two_way(y_w, X_w, W.fe1, W.fe2);
        [Qw,Rw,~] = qr(X_wt,0);
        dw = abs(diag(Rw)); tol = max(size(X_wt))*eps(max(dw));
        rw = sum(dw>tol); if rw==0, continue; end
        Xw_use = X_wt(:,1:rw);
        bw_use = Rw(1:rw,1:rw) \ (Qw(:,1:rw)' * y_wt);
        ew     = y_wt - Xw_use*bw_use;
        Rw11 = Rw(1:rw,1:rw); Rw11inv = inv(Rw11);
        XtXinv_w = Rw11inv*Rw11inv';
        Vw_f = cluster_vcov(Xw_use, ew, W.fe1, XtXinv_w);
        Vw_m = cluster_vcov(Xw_use, ew, W.m_id_L1, XtXinv_w);
        Vw_p = cluster_vcov(Xw_use, ew, grp2idx(string(W.fe1)+"_"+string(W.m_id_L1)), XtXinv_w);
        Vw_2 = Vw_f + Vw_m - Vw_p;
        [Fstat, pF, ~, ~, ~] = wald_block_stable(bw_use, Vw_2, 1, size(Xw_use,1), size(Xw_use,2));
        WF = [WF; {string(bnm), Fstat, pF}]; 
    end
    writetable(WF, [tag '_windowed_F.csv']);

    % ALL_LEADS placebo test
    leadNames = {};
    for L = lead_horizons
        for v = 1:numel(iv_use)
            base = iv_use{v};
            nm = sprintf('%s_lead%d', base, L);
            D.(nm) = leadWithinFund(D, D.(base), L);
            leadNames{end+1,1} = nm;
        end
    end
    if ~isempty(leadNames)
        XBnames = [iv_use(:); leadNames(:); controls(:)];
        okb = true(height(D),1);
        for j=1:numel(XBnames), okb = okb & ismember(XBnames{j}, D.Properties.VariableNames) & ~isnan(D.(XBnames{j})); end
        B = D(okb,:);
        if height(B) >= 500
            yb = B.(yvar); Xb = B{:, XBnames};
            [yb_t, Xb_t] = absorb_two_way(yb, Xb, B.fe1, B.fe2);
            [Qb,Rb,Eb] = qr(Xb_t,0);
            dg = abs(diag(Rb)); tol = max(size(Xb_t))*eps(max(dg));
            rb = sum(dg>tol);
            keep_b = false(size(Eb)); if rb>0, keep_b(Eb(1:rb)) = true; end
            Xb_use = Xb_t(:, keep_b);
            bb_use = Rb(1:rb,1:rb) \ (Qb(:,1:rb)' * yb_t);
            eb     = yb_t - Xb_use*bb_use;
            Rb11 = Rb(1:rb,1:rb); Rb11inv = inv(Rb11);
            XtXinv_b = Rb11inv*Rb11inv';
            Vf = cluster_vcov(Xb_use, eb, B.fe1, XtXinv_b);
            Vm = cluster_vcov(Xb_use, eb, B.m_id_L1, XtXinv_b);
            Vp = cluster_vcov(Xb_use, eb, grp2idx(string(B.fe1)+"_"+string(B.m_id_L1)), XtXinv_b);
            V2 = Vf + Vm - Vp;
            kept_full = Eb(1:rb);
            full_to_kept = containers.Map('KeyType','double','ValueType','double');
            for kk=1:numel(kept_full), full_to_kept(kept_full(kk)) = kk; end
            lead_idx_full = find(ismember(XBnames, leadNames));
            idx_use_leads = [];
            for q=1:numel(lead_idx_full)
                if isKey(full_to_kept, lead_idx_full(q))
                    idx_use_leads(end+1) = full_to_kept(lead_idx_full(q)); 
                end
            end
            if ~isempty(idx_use_leads)
                [Fpl, pFpl, chi2pl, pChipl, qpl] = wald_block_stable(bb_use, V2, idx_use_leads, size(Xb_use,1), size(Xb_use,2));
                PT = table("ALL_LEADS", qpl, Fpl, pFpl, chi2pl, pChipl, 'VariableNames', ...
                    {'lead','q_params','F_twoWay','p_F','Chi2','p_Chi2'});
            else
                PT = table("ALL_LEADS", 0, NaN, NaN, NaN, NaN, 'VariableNames', ...
                    {'lead','q_params','F_twoWay','p_F','Chi2','p_Chi2'});
            end
        else
            PT = table("ALL_LEADS", 0, NaN, NaN, NaN, NaN, 'VariableNames', ...
                {'lead','q_params','F_twoWay','p_F','Chi2','p_Chi2'});
        end
        writetable(PT, [tag '_placebo_all.csv']);
    end

    % -------- Pre-period, block-specific placebos --------
    PP = table('Size',[0 6],'VariableTypes',{'string','double','double','double','double','double'}, ...
               'VariableNames',{'block','q_params','F_twoWay','p_F','Chi2','p_Chi2'});
    bnames2 = {'CFTC','FATCA','TAX97','TAX03','TAX13'};
    for bb=1:numel(bnames2)
        bnm = bnames2{bb};
        base_match = find(contains(ivars, bnm, 'IgnoreCase', true), 1);
        if isempty(base_match), continue; end
        baseZ = ivars{base_match};
        m = D.m_id;
        if      strcmp(bnm,'CFTC'),  maskPre = (m < ym(2013,1));
        elseif  strcmp(bnm,'FATCA'), maskPre = (m < ym(2014,7));
        elseif  strcmp(bnm,'TAX97'), maskPre = (m < ym(1997,5));
        elseif  strcmp(bnm,'TAX03'), maskPre = (m < ym(2003,1));  % tightened
        elseif  strcmp(bnm,'TAX13'), maskPre = (m < ym(2013,1));
        else, maskPre = false(size(m));
        end
        W = D(maskPre,:);
        if height(W) < 200 || ~ismember(baseZ, W.Properties.VariableNames), continue; end
        leadH = [6,12];
        leadNamesPre = cell(numel(leadH),1);
        for L=1:numel(leadH)
            nm = sprintf('%s_lead%d_pre', baseZ, leadH(L));
            W.(nm) = leadWithinFund(W, W.(baseZ), leadH(L));
            leadNamesPre{L} = nm;
        end
        okp = ~isnan(W.(yvar)) & W.fe2>0 & ~isnan(W.m_id_L1);
        for j=1:numel(leadNamesPre), okp = okp & ~isnan(W.(leadNamesPre{j})); end
        for j=1:numel(controls), if ismember(controls{j}, W.Properties.VariableNames), okp = okp & ~isnan(W.(controls{j})); end, end
        W = W(okp,:); if height(W) < 200, continue; end
        yb = W.(yvar); Xb = [W{:, leadNamesPre'}, W{:, controls}];
        [yb_t, Xb_t] = absorb_two_way(yb, Xb, W.fe1, W.fe2);
        [Qb,Rb,~] = qr(Xb_t,0);
        dg = abs(diag(Rb)); tol = max(size(Xb_t))*eps(max(dg));
        rb = sum(dg>tol); if rb==0, continue; end
        Xb_use = Xb_t(:,1:rb);
        bb_use = Rb(1:rb,1:rb) \ (Qb(:,1:rb)' * yb_t);
        eb     = yb_t - Xb_use*bb_use;
        Rb11 = Rb(1:rb,1:rb); Rb11inv = inv(Rb11);
        XtXinv_b = Rb11inv*Rb11inv';
        Vf = cluster_vcov(Xb_use, eb, W.fe1, XtXinv_b);
        Vm = cluster_vcov(Xb_use, eb, W.m_id_L1, XtXinv_b);
        Vp = cluster_vcov(Xb_use, eb, grp2idx(string(W.fe1)+"_"+string(W.m_id_L1)), XtXinv_b);
        V2 = Vf + Vm - Vp;
        [Fpl, pFpl, chi2pl, pChipl, qpl] = wald_block_stable(bb_use, V2, 1:numel(leadNamesPre), size(Xb_use,1), size(Xb_use,2));
        PP = [PP; {string(bnm), qpl, Fpl, pFpl, chi2pl, pChipl}];
    end
    writetable(PP, [tag '_placebo_preperiod.csv']);

    %  Fitted within & residuals 
    yhat_within = X_use*b_use;
    res_within  = e_til;
    OUT = table(D.FundID, D.Year, D.Month, y_til, yhat_within, res_within, ...
        'VariableNames', {'FundID','Year','Month','y_within','yhat_within','res_within'});
    writetable(OUT, [tag '_fitted_within.csv']);
end

% Conduct full specifications
if run_Flow_t
    run_spec(T, 'Flow', ivars, controls, 'Flow_spec', lead_horizons, winfun, prefun, ym);
end

function vLag = lagWithinFund(T, v, L)
    vLag = NaN(height(T),1);
    uF = unique(T.FundID);
    for f = uF.'
        ix = find(T.FundID==f);
        Luse = min(L, numel(ix));
        vLag(ix) = [NaN(Luse,1); v(ix(1:end-Luse))];
    end
end

function vLead = leadWithinFund(T, v, L)
    vLead = NaN(height(T),1);
    uF = unique(T.FundID);
    for f = uF.'
        ix = find(T.FundID==f);
        Luse = min(L, numel(ix));
        vLead(ix) = [v(ix(1+Luse:end)); NaN(Luse,1)];
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
    lo = prctile(x(ok), p1);
    hi = prctile(x(ok), p99);
    xw(ok & x < lo) = lo;
    xw(ok & x > hi) = hi;
end

function [y_out, X_out] = absorb_two_way(y, X, g1, g2)
    y_out = y; X_out = X;
    maxIter = 1000; tol = 1e-10;
    for it=1:maxIter
        y_prev = y_out; X_prev = X_out;
        y_out = y_out - group_mean(y_out, g1);
        X_out = X_out - group_mean(X_out, g1);
        y_out = y_out - group_mean(y_out, g2);
        X_out = X_out - group_mean(X_out, g2);
        if max(norm(y_out - y_prev)/max(1,norm(y_prev)), norm(X_out - X_prev,'fro')/max(1,norm(X_prev,'fro'))) < tol
            break;
        end
    end
end

function M = group_mean(A, g, return_group_means)
    if nargin<3, return_group_means=false; end
    g = double(g(:)); n = size(A,1);
    if any(g<1 | isnan(g)), error('group_mean: bad ids'); end
    G = max(g);
    S = sparse(g, (1:n)', 1, G, n);
    sums = S * A;
    cnts = S * ones(n,1);
    cnts(cnts==0) = 1;
    mu = sums ./ cnts;
    if return_group_means, M = mu; else, M = mu(g,:); end
end

function V = cluster_vcov(X, e, cluster_id, XtXinv)
    [n,k] = size(X);
    if nargin<4 || isempty(XtXinv)
        XtX = X'*X;
        XtXinv = pinv(XtX);
    end
    G = max(cluster_id);
    S = zeros(k,k);
    for g=1:G
        idx = (cluster_id==g);
        if ~any(idx), continue; end
        Xg = X(idx,:); eg = e(idx);
        S = S + (Xg' * eg) * (Xg' * eg)';
    end
    df_c = 1;
    if G>1, df_c = (G/(G-1)) * ((n-1)/max(1,n-k)); end
    V = df_c * (XtXinv * S * XtXinv);
end

function [Fstat, pF, chi2, pChi, q_eff] = wald_block_stable(b, V, idx, n, k)
    q  = numel(idx);
    if q==0, Fstat=NaN; pF=NaN; chi2=NaN; pChi=NaN; q_eff=0; return; end
    R = zeros(q, numel(b)); 
    for i=1:q, R(i, idx(i)) = 1; end
    rb  = R*b;
    RVRT = R*V*R';
    [U,S,~] = svd((RVRT+RVRT')/2, 'econ');
    s = diag(S);
    if isempty(s), Fstat=NaN; pF=NaN; chi2=NaN; pChi=NaN; q_eff=0; return; end
    tol = max(size(RVRT))*eps(max(s));
    keep = s > tol;
    r = sum(keep);
    if r==0
        ridge = max(1e-10, 1e-12*trace(RVRT)/max(1,q));
        K = RVRT + ridge*eye(q);
        chi2 = rb' * (K \ rb);
        q_eff = q;
    else
        U1 = U(:, keep);
        s1 = s(keep);
        z  = U1' * rb;
        chi2 = sum( (z.^2) ./ max(s1, eps) );
        q_eff = r;
    end
    pChi = 1 - chi2cdf(chi2, q_eff);
    Fstat = chi2 / max(1,q_eff);
    pF    = 1 - fcdf(Fstat, max(1,q_eff), max(1,n-k));
end
