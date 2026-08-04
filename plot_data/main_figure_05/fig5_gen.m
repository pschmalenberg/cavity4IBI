close all
clear
clc

alpha = 0.75;
diffScale = 0.15;

% Read files
a = readmatrix('a.csv');
b = readmatrix('b.csv');
c = readmatrix('c.csv');

d = readmatrix('d.csv');
e = readmatrix('e.csv');
f = readmatrix('f.csv');

aa = readmatrix('aa.csv');
bb = readmatrix('bb.csv');
cc = readmatrix('cc.csv');

dd = readmatrix('dd.csv');
ee = readmatrix('ee.csv');
ff = readmatrix('ff.csv');

% NEW THIRD SET
aaa = readmatrix('aaa.csv');
bbb = readmatrix('bbb.csv');
ccc = readmatrix('ccc.csv');

ddd = readmatrix('ddd.csv');
eee = readmatrix('eee.csv');
fff = readmatrix('fff.csv');

% Force column vectors
a=a(:); b=b(:); c=c(:);
d=d(:); e=e(:); f=f(:);

aa=aa(:); bb=bb(:); cc=cc(:);
dd=dd(:); ee=ee(:); ff=ff(:);

aaa=aaa(:); bbb=bbb(:); ccc=ccc(:);
ddd=ddd(:); eee=eee(:); fff=fff(:);

% Concatenate signals
meas  = [a; b; c];
pred  = [d; e; f];

meas1 = [aa; bb; cc];
pred1 = [dd; ee; ff];

% NEW THIRD CONCATENATION
meas2 = [aaa; bbb; ccc];
pred2 = [ddd; eee; fff];

% Normalize
norm01 = @(x) (x-min(x))/(max(x)-min(x));

meas_n  = norm01(meas);
pred_n  = norm01(pred);

meas_n1 = norm01(meas1);
pred_n1 = norm01(pred1);

meas_n2 = norm01(meas2);
pred_n2 = norm01(pred2);

% Differences
diff_sig  = diffScale*(pred_n-meas_n);
diff_sig1 = diffScale*(pred_n1-meas_n1);
diff_sig2 = diffScale*(pred_n2-meas_n2);

% Time axis
t = linspace(0,36,length(meas_n));

figure(1)
clf
hold on

% Row 1
plot(t, meas_n,'Color',[0.4660 0.6740 0.1880 alpha],'LineWidth',2)
plot(t, pred_n,':','Color',[0.8500 0.3250 0.0980 alpha],'LineWidth',2)
plot(t, diff_sig+0.5,'k','LineWidth',2)

% Row 2
plot(t, meas_n1-1,'Color',[0.4660 0.6740 0.1880 alpha],'LineWidth',2)
plot(t, pred_n1-1,':','Color',[0.8500 0.3250 0.0980 alpha],'LineWidth',2)
plot(t, diff_sig1-0.5,'k','LineWidth',2)

% Row 3 (NEW corrected third dataset)
plot(t, meas_n2-2,'Color',[0.4660 0.6740 0.1880 alpha],'LineWidth',2)
plot(t, pred_n2-2,':','Color',[0.8500 0.3250 0.0980 alpha],'LineWidth',2)
plot(t, diff_sig2-1.5,'k','LineWidth',2)

yticks([])

ax = gca;
ax.XGrid='off';
ax.YGrid='on';
ax.GridColor=[0 0 0];
ax.GridLineStyle='--';
ax.GridAlpha=0.75;
box on 
legend('ECG Measurement','ECG Prediction','Difference')

xlabel('Seconds [s]')
ylabel('Normalized Value [arb. unit]')
xlim([0 36])