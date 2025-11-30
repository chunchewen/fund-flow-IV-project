% The difference between this code with "c:...\Revision2025\RevisedIVdummy\IVdummy.m is
% equally weights are applied in this robust tests.
clear; clc;
% Monthly calendar for instruments 
startDate = datetime(1993,1,1);
endDate   = datetime(2022,12,31);
monthGrid = (dateshift(startDate,'start','month'):calmonths(1):dateshift(endDate,'start','month')).';

%(1) LOAD DATA & CLEAN -999 
T = readtable('C:\Users\ias05106\OneDrive - University of Strathclyde\Desktop\Updated Hedge Fund Project\HedgeFundCharacters.xlsx','sheet', 'FundCharacteristics');
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

% Replace -999 with NaN in numeric columns
isNum = varfun(@isnumeric, T, 'OutputFormat','uniform');
for j = find(isNum)
    v = T.(j); v(v == -999) = NaN; T.(j) = v;
end

col = @(name) getColCaseInsensitive(T, name);

% FundID fallback
if any(strcmpi(T.Properties.VariableNames,'FundID'))
    FundID = T.FundID;
else
    FundID = (1:height(T)).';
end

%% ---------- 1) FLEX (continuous pass-through) ----------
lockup_m  = col('LockUPPeriod');            % months
notice_d  = col('RedemptionNoticePeriod');  % days
sub_freq  = col('SubscriptionFrequency');   % per year
red_freq  = col('RedemptionFrequency');     % per year

Lc = clamp01( (3 - lockup_m) ./ 3 );        % 1 if lockup<=3m, taper to 0
Nc = clamp01( (30 - notice_d) ./ 30 );      % 1 if notice<=30d, taper to 0
Sc = clamp01(  sub_freq ./ 12 );            % cap at monthly
Rc = clamp01(    red_freq ./ 12 );          % cap at monthly

Flex = mean([0.25*Lc, 0.25*Nc, 0.25*Rc, 0.25*Sc], 2, 'omitnan');   % [0,1]

%(2) DERIVATIVES INTENSITY 
Futures   = to01(col('Futures'));
Derivs    = to01(col('Derivatives'));
Margin    = to01(col('Margin'));
FXCredit  = to01(col('FXCredit'));

lev_max   = col('MaxLeverage');
lev_avg   = col('AvgLeverage');
LevScaled = mean([scale01(lev_max), scale01(lev_avg)], 2, 'omitnan');  % [0,1]

% Trading Style 
% 1=ConvArb, 2=ShortBias, 3=EM, 4=EqMktNeutral, 5=EventDriven, 6=FIArb,
% 7=FoF, 8=GlobalMacro, 9=L/S Equity, 10=ManagedFutures, 11=Multi-Strategy,
% 12=Options, 13=Other, 14=Undefined
styleNum = col('Style');
StyleBoost = zeros(height(T),1);
derivCodes = [10 8 6 1 12];                 % MF, Macro, FI Arb, Conv Arb, Options
StyleBoost(ismember(styleNum, derivCodes)) = 0.33;

flagBlock = mean([0.25*Futures, 0.25*Derivs, 0.25*Margin, 0.25*FXCredit], 2, 'omitnan');
DerivativesIntensity = clamp01(0.33*flagBlock + 0.33*LevScaled + StyleBoost);

%(3) HNW SCORE 
minInv    = col('MinimumInvestment');       % USD
HNW_score = NaN(height(T),1);
valid = minInv>0 & ~isnan(minInv);
if any(valid)
    x = log(minInv(valid));
    x = winsor(x,1,99);
    HNW_score(valid) = scale01(x);        
end

RIA = to01(col('RegisteredInvestmentAdviser'));

dc = col('DomicileCountry');                % numeric; 66 = US
US_domicile = NaN(height(T),1);
if ~all(isnan(dc))
    US_domicile = double(dc == 66);
    US_domicile(isnan(dc)) = NaN;
end
Offshore = double(US_domicile == 0);
Offshore(isnan(US_domicile)) = NaN;

OpenToPublic      = to01(col('OpenToPublic'));
AccptsManagedAcc  = to01(col('AccptsManagedAccounts'));

% (4) POLICY TIME SHOCKS 
Post = @(d) double(monthGrid >= d);
rampStart = datetime(2013,7,1); rampEnd = datetime(2013,12,31);
PostCFTC = zeros(numel(monthGrid),1);
for t = 1:numel(monthGrid)
    if monthGrid(t) < rampStart
        PostCFTC(t) = 0;
    elseif monthGrid(t) > rampEnd
       PostCFTC(t) = 1;
    else
        PostCFTC(t) = days(monthGrid(t) - rampStart) / days(rampEnd - rampStart);
    end
end
rampStart = [];   rampEnd = [];

% FATCA ramp: 2014-07-01 to 2015-12-31, then 1
rampStart = datetime(2014,7,1); rampEnd = datetime(2015,12,31);
FATCA_ramp = zeros(numel(monthGrid),1);
for t = 1:numel(monthGrid)
    if monthGrid(t) < rampStart
        FATCA_ramp(t) = 0;
    elseif monthGrid(t) > rampEnd
        FATCA_ramp(t) = 1;
    else
        FATCA_ramp(t) = days(monthGrid(t) - rampStart) / days(rampEnd - rampStart);
    end
end

% --- Disjoint US tax windows ---
Inc1997 = (monthGrid >= datetime(1997,5,7)) & (monthGrid < datetime(2003,5,6));
Inc2003 = (monthGrid >= datetime(2003,5,6)) & (monthGrid < datetime(2013,1,1));
Inc2013 = (monthGrid >= datetime(2013,1,1));
Inc1997 = double(Inc1997); Inc2003 = double(Inc2003); Inc2013 = double(Inc2013);

%(5) INSTRUMENTS 
nF = height(T);  nT = numel(monthGrid);
toPanel = @(timeVec, fundVec) timeVec .* (ones(nT,1) * fundVec.');  % outer product

% Base exposures
fund_CFTC   = DerivativesIntensity .* Flex;                 % [0,1]
fund_FATCA  = Offshore .* Flex;                             % [0,1]

% TAX exposures (patched)
E_tax_A = HNW_score;                                        % [0,1]
E_tax_C = clamp01(0.33*HNW_score + 0.33*OpenToPublic + 0.33*AccptsManagedAcc);

% Quantile ranks (0..1) to spread mass
q_CFTC     = quantileRank(fund_CFTC, 10);
q_FATCA    = quantileRank(fund_FATCA, 10);
q_Etax_A   = quantileRank(E_tax_A, 10);
q_Etax_C   = quantileRank(E_tax_C, 10);

% --- Core instruments ---
Z_CFTC_q         = toPanel(PostCFTC,   q_CFTC);
Z_FATCA_q  = toPanel(FATCA_ramp, q_FATCA);

% --- Disjoint Tax window instruments (A & C exposures; raw + _q) ---
Z_TaxWin1997_A   = toPanel(Inc1997, E_tax_A);
Z_TaxWin2003_A   = toPanel(Inc2003, E_tax_A);
Z_TaxWin2013_A   = toPanel(Inc2013, E_tax_A);

Z_TaxWin1997_C   = toPanel(Inc1997, E_tax_C);
Z_TaxWin2003_C   = toPanel(Inc2003, E_tax_C);
Z_TaxWin2013_C   = toPanel(Inc2013, E_tax_C);

Z_TaxWin1997_A_q = toPanel(Inc1997, q_Etax_A);
Z_TaxWin2003_A_q = toPanel(Inc2003, q_Etax_A);
Z_TaxWin2013_A_q = toPanel(Inc2013, q_Etax_A);

Z_TaxWin1997_C_q = toPanel(Inc1997, q_Etax_C);
Z_TaxWin2003_C_q = toPanel(Inc2003, q_Etax_C);
Z_TaxWin2013_C_q = toPanel(Inc2013, q_Etax_C);

%(6) DIAGNOSTICS
FundID_rep = repmat(FundID.', nT, 1);
Date_rep   = repmat(monthGrid, 1, nF);

InstrumentsLong = table;
InstrumentsLong.Date   = Date_rep(:);
InstrumentsLong.FundID = FundID_rep(:);

% Add all Z_* columns explicitly
addZ = struct( ...
 'Z_CFTC_q',Z_CFTC_q,  ...
 'Z_FATCA_q',Z_FATCA_q,'Z_TaxWin1997_A',Z_TaxWin1997_A,'Z_TaxWin2003_A',Z_TaxWin2003_A,'Z_TaxWin2013_A',Z_TaxWin2013_A, ...
 'Z_TaxWin1997_C',Z_TaxWin1997_C,'Z_TaxWin2003_C',Z_TaxWin2003_C,'Z_TaxWin2013_C',Z_TaxWin2013_C, ...
 'Z_TaxWin1997_A_q',Z_TaxWin1997_A_q,'Z_TaxWin2003_A_q',Z_TaxWin2003_A_q,'Z_TaxWin2013_A_q',Z_TaxWin2013_A_q, ...
 'Z_TaxWin1997_C_q',Z_TaxWin1997_C_q,'Z_TaxWin2003_C_q',Z_TaxWin2003_C_q,'Z_TaxWin2013_C_q',Z_TaxWin2013_C_q );

fns = fieldnames(addZ);
for k=1:numel(fns)
    zname = fns{k}; zmat = addZ.(zname);
    InstrumentsLong.(zname) = zmat(:);
end

% Fund-level exposures 
FundExposures = table(FundID, Flex, DerivativesIntensity, HNW_score, RIA, US_domicile, Offshore, ...
    'VariableNames', {'FundID','Flex','DerivativesIntensity','HNW_score','RIA','US_domicile','Offshore'});

% Diagnostics
ivar = InstrumentsLong.Properties.VariableNames;
ivar = ivar( cellfun(@(s) strncmp(s,'Z_',2), ivar) );
nI = numel(ivar);
nonzero_share = zeros(nI,1);
stdev         = zeros(nI,1);
for k = 1:nI
    x  = InstrumentsLong.(ivar{k});
    ok = ~isnan(x);
    nonzero_share(k) = mean(x(ok) ~= 0);
    stdev(k)         = std(x(ok));
    fprintf('%-20s  nz=%.3f  sd=%.4f\n', ivar{k}, nonzero_share(k), stdev(k));
end
Diag = table(string(ivar(:)), nonzero_share, stdev, ...
    'VariableNames', {'instrument','nonzero_share','stddev'});
Diag.low_variation = (nonzero_share < 0.05) | (stdev < 1e-6);

writetable(InstrumentsLong, 'starter_instruments_long.csv');
writetable(FundExposures,   'fund_exposures.csv');
writetable(Diag,            'instrument_diagnostics.csv');

function v = getColCaseInsensitive(T, name)
    vn = T.Properties.VariableNames;
    ix = find(strcmpi(vn, name), 1);
    if isempty(ix)
        ix = find(contains(lower(vn), lower(name)), 1);
    end
    if isempty(ix)
        v = NaN(height(T),1);
        fprintf(2,'[WARN] Column "%s" not found; filling NaN.\n', name);
    else
        v = T.(ix);
    end
end

function y = clamp01(x)
    y = x; y(y<0) = 0; y(y>1) = 1;
end

function y = scale01(x)
    y = NaN(size(x));
    ok = isfinite(x);
    if any(ok)
        a = min(x(ok)); b = max(x(ok));
        if b>a, y(ok) = (x(ok)-a) / (b-a); else, y(ok) = 0.5; end
    end
end

function xw = winsor(x,p_lo,p_hi)
    lo = prctile(x, p_lo); hi = prctile(x, p_hi);
    xw = min(max(x,lo),hi);
end

function z = to01(x)
    if isnumeric(x) || islogical(x)
        z = double(x==1); z(isnan(x)) = NaN;
    else
        s = string(x); s = lower(strtrim(s));
        z = double(ismember(s, ["y","yes","true","1"]));
        z(ismissing(s)) = NaN;
    end
end

function q = quantileRank(x, nbin)
    q = NaN(size(x));
    ok = isfinite(x);
    if any(ok)
        % use prctile for compatibility
        pct = linspace(0,100,nbin+1);
        edges = unique(prctile(x(ok), pct));
        if numel(edges) < 3
            q(ok) = 0.5;
        else
            [~,~,bin] = histcounts(x(ok), edges);
            q(ok) = (bin - 0.5) ./ max(bin);  % (0,1]
        end
    end
end

