% Bulk density and organic matter (OM) are obtained from laboratory soil
% analysis
bulk_density = [1.49 1.42 1.32 1.07];
OM = [2.83 3.62 5.46 7.79];

v_clay = 0.4;
v_silt = 0.4;
v_sand = 0.2;
OC = OM * 0.58;

%% 
% Mathematical modeling of permittivity
wilting_point = 0.02982 + 0.089 * v_clay + 0.00786 * OM;
p = 0.6819 - 0.0648 ./ (OC + 1) - 0.119 * bulk_density.^2 - 0.02668 ...
    + 0.1489 * v_clay + 0.08031 * v_silt + 0.02321 ./ ((OC + 1) .* (bulk_density.^2)) ...
    - 0.01908 * bulk_density.^2 - 0.1109 * v_clay - 0.2315 * v_clay * v_silt ...
    - 0.01197 * v_silt* bulk_density.^2 - 0.01068 * v_clay * bulk_density.^2;
e_sand = 3 + 0.078j;
e_silt = 5 + 0.078j;
e_clay = 5 + 0.078j;
e_soil = v_sand * e_sand + v_clay * e_clay + v_silt * e_clay;

% VWC
w = 0:0.01:0.7; 

e_min = 4.9;
freq = 2.4e9;
omega = 2 * pi * freq;
tau_bound = 1e-11;

e_bound_max = -36 * v_clay + 44;
e_bound_real = e_min + (e_bound_max - e_min) / (1 + (omega * tau_bound).^2);
e_bound_imag = omega * tau_bound * (e_bound_max - e_min) / (1 + (omega * tau_bound).^2);
e_bound = e_bound_real + 1j * e_bound_imag;

T = 20;
tau_free = (1.1109e-10 + 3.824e-12 * T + 6.938e-14 * T^2 - 5.096e-16 * T^3) / 2 / pi;
e_free_max = 88.045 - 0.4147 * T + 6.295e-4 * T^2 + 1.075e-5 * T^3;
e_free_real = e_min + (e_free_max - e_min) / (1 + (omega * tau_free).^2);
e_free_imag = omega * tau_free * (e_free_max - e_min) / (1 + (omega * tau_free).^2);
e_free = e_free_real + 1j * e_free_imag;

e_air = 1;
e_eff = [];
for i = 1:length(p)
    w1 = w(w <= wilting_point(i));
    w2 = w(w > wilting_point(i) & w <= p(i));
    w3 = w(w > p(i));
    e_eff(i, 1:length(w1)) = (1 - p(i)) * e_soil + w1 .* e_bound + (p(i) - w1) * e_air;
    e_eff(i, length(w1) + 1 : length(w1) + length(w2)) = (1 - p(i)) * e_soil + ...
                 w2 .* ((p(i) - w2) / (p(i) - wilting_point(i))* e_bound + (w2 - wilting_point(i)) / (p(i) - wilting_point(i)) * e_free) + ...
                 (p(i) - w2) * e_air;
    e_eff(i, length(w1) + length(w2) + 1 : length(w)) = (1 - w3) * e_soil + w3 .* e_free;
end
e_a = real(e_eff) / 2 .* (sqrt(1 + (imag(e_eff) ./ real(e_eff)) .^ 2) + 1);

%% 
f=figure(1);clf;
f.Position = [10 10 550 300];
for i = 1:length(p)
    plot(w*100, e_a(i, :), 'LineWidth', 2);
    hold on;
end
xlabel('Volumetric water content (%)');

ylabel('Permittivity');
grid off;
legend('1% oc', '2% oc', '3% oc', '4% oc', 'Location', 'best');
set(gca,'FontSize',24);
ylim([0 40]);



