
OC_range = 0:1:20;
OM_range = OC_range/0.58;
w = 0:0.02:0.7;

% Use bulk densities and organic carbon (OC) obtained from laboratory soil
% analysis to estimate the linear equation parameters for bulk density vs.
% OC
bulk_density = [1.0430    0.9660    0.8930    0.8890    0.7920    0.7180    0.6010];
OC = [1.54 3.03 4.25 4.90 14.14 18.97 31.58];
figure(2);clf;
h1 = scatter(OC, bulk_density);
hold on;
p = polyfit(OC,bulk_density,1);
b = p(1); 
a = p(2);
h2 = refline(p);
% set(h1,'LineWidth',2);
set(h2,'LineWidth',2,'LineStyle', '--', 'Color', 'k');

xlabel('Organic carbon (%)');
ylabel('Bulk density (g/cm^3)');
grid on;
legend({'Data points',sprintf('Fit:  BD = %.3f*OC + %.3f',b,a)}, 'Location','northeast')
set(gca,'FontSize',24);


%%
% Sand compost mixtures
v_clay = 0;
v_silt = 0;
v_sand = 1;


e_sand = 3 + 0.078j;
e_silt = 5 + 0.078j;
e_clay = 5 + 0.078j;
e_soil = v_sand * e_sand + v_clay * e_clay + v_silt * e_clay;

e_air = 1;

e_min = 4.9;
freq = 2.4e9;
omega = 2 * pi * freq;
tau_bound = 1e-11;

e_bound_max = -36 * v_clay + 44;
e_bound_real = e_min + (e_bound_max - e_min) / (1 + (omega * tau_bound).^2);
e_bound_imag = omega * tau_bound * (e_bound_max - e_min) / (1 + (omega * tau_bound).^2);
e_bound = e_bound_real + 1j * e_bound_imag;

% fprintf('Bould water permitivity: ');
% disp(e_bound);

T = 20;
tau_free = (1.1109e-10 + 3.824e-12 * T + 6.938e-14 * T^2 - 5.096e-16 * T^3) / 2 / pi;
e_free_max = 88.045 - 0.4147 * T + 6.295e-4 * T^2 + 1.075e-5 * T^3;
e_free_real = e_min + (e_free_max - e_min) / (1 + (omega * tau_free).^2);
e_free_imag = omega * tau_free * (e_free_max - e_min) / (1 + (omega * tau_free).^2);
e_free = e_free_real + 1j * e_free_imag;
% fprintf('Free water permitivity: ');
% disp(e_free);


%%
bulk_density = b * OC_range + a;
p = 0.6819 - 0.0648 ./ (OC_range + 1) - 0.119 * bulk_density.^2 - 0.02668 ...
    + 0.1489 * v_clay + 0.08031 * v_silt + 0.02321 ./ ((OC_range + 1) .* (bulk_density.^2)) ...
    - 0.01908 * bulk_density.^2 - 0.1109 * v_clay - 0.2315 * v_clay * v_silt ...
    - 0.01197 * v_silt* bulk_density.^2 - 0.01068 * v_clay * bulk_density.^2;


wilting_point = 0.02982 + 0.089 * v_clay + 0.00786 * OM_range;


%%
f=figure(3);clf;
plot(OC_range, p, 'LineWidth', 2);
hold on;
plot(OC_range, wilting_point, '--', 'LineWidth', 2);
f.Position = [10 10 550 300];

grid on;

xlabel('Organic carbon (%)');
ylabel('m^3/m^3');
legend('Porosity p', 'Wilting point w_{wp}', 'Location', 'best');
set(gca,'FontSize',24);


