load(fullfile('data', 'scarf_permittivity.mat'))
load(fullfile('data', 'lightness_cali.mat'))
load(fullfile('data', 'oven_vwc_data.mat'))
addpath('utils/')
OC_range = 0:0.1:35;
OM_range = OC_range/0.58;


% Add a constant offset
e_measure_all = scarf_result + 2;
l_measure_all = lightness_cali;
moisture_levels = oven_vwc_sand;

% Use bulk densities and organic carbon (OC) obtained from laboratory soil
% analysis to estimate the linear equation parameters for bulk density vs.
% OC
bulk_density = [1.0430    0.9660    0.8930    0.8890    0.7920    0.7180    0.6010];
OC = [1.54 3.03 4.25 4.90 14.14 18.97 31.58];
OC_measured_2d = reshape(repmat(OC, 1, 6), [], 6);

p = polyfit(OC,bulk_density,1);
b = p(1); 
a = p(2);



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

%         fprintf('Bould water permitivity: ');
%         disp(e_bound);

T = 20;
tau_free = (1.1109e-10 + 3.824e-12 * T + 6.938e-14 * T^2 - 5.096e-16 * T^3) / 2 / pi;
e_free_max = 88.045 - 0.4147 * T + 6.295e-4 * T^2 + 1.075e-5 * T^3;
e_free_real = e_min + (e_free_max - e_min) / (1 + (omega * tau_free).^2);
e_free_imag = omega * tau_free * (e_free_max - e_min) / (1 + (omega * tau_free).^2);
e_free = e_free_real + 1j * e_free_imag;
%         fprintf('Free water permitivity: ');
%         disp(e_free);

bulk_density = b * OC_range + a;
p = 0.6819 - 0.0648 ./ (OC_range + 1) - 0.119 * bulk_density.^2 - 0.02668 ...
    + 0.1489 * v_clay + 0.08031 * v_silt + 0.02321 ./ ((OC_range + 1) .* (bulk_density.^2)) ...
    - 0.01908 * bulk_density.^2 - 0.1109 * v_clay - 0.2315 * v_clay * v_silt ...
    - 0.01197 * v_silt* bulk_density.^2 - 0.01068 * v_clay * bulk_density.^2;

wilting_point = 0.02982 + 0.089 * v_clay + 0.00786 * OM_range;



%%

estimated_oc = [];
estimated_w = [];
for box_id = 1:7
    for moisture_id = 1:6
        l_measure = l_measure_all(box_id, moisture_id);
        e_measure = e_measure_all(box_id, moisture_id);
        w = (-20.8084*log(OC_range+1) + 92.0114 - l_measure)/0.5565/ 100; % calibrated lightness  
        e_a = [];
        for i = 1:length(w)
            e_a(i) = compute_e(OC_range(i), w(i), e_soil, e_bound, e_air, e_free); 
        end
        obj_fun = abs(e_a - e_measure);
        [val, id] = min(obj_fun);
        estimated_oc(box_id, moisture_id) = OC_range(id);
        estimated_w(box_id, moisture_id) = w(id);
    end
end

%%

correlation_matrix = corrcoef(estimated_oc(:), OC_measured_2d(:));
correlation_oc = correlation_matrix(2, 1);
s_res = sum((OC_measured_2d(:) - estimated_oc(:)).^2);
y_mean = mean(OC_measured_2d(:));
s_tot = sum((OC_measured_2d(:)-y_mean).^2);
r_squared_oc = 1-s_res/s_tot;
mse_oc = immse(estimated_oc(:), OC_measured_2d(:));
fprintf('OC Correlation: %.4f, R^2: %.4f, MSE: %.3f\n', correlation_oc, r_squared_oc, mse_oc);

f=figure(3);clf;
f.Position = [10 10 400 350];
scatter(OC_measured_2d(:), estimated_oc(:),30, 'filled');
hold on;
x = linspace(0, 35, 20);
plot(x, x, '--k', 'LineWidth', 2);
xlabel('Ground truth carbon (%)');
ylabel('Estimated carbon (%)');
txt2 = sprintf('R^2: %.3f', r_squared_oc);
txt1 = sprintf('Corr: %.3f', correlation_oc);
txt3 = sprintf('MSE: %.3f', mse_oc);
txt = {txt1, txt2, txt3};
text(0.3,31,txt, 'FontSize', 20)
set(gca,'FontSize',24);
legend({'Data points','y=x'}, 'Location','southeast');

estimated_vwc = estimated_w * 100;
correlation_matrix = corrcoef(estimated_w(:)*100, moisture_levels(:));
correlation_w = correlation_matrix(2, 1);
s_res = sum((moisture_levels(:) - 100*estimated_w(:)).^2);
y_mean = mean(moisture_levels(:));
s_tot = sum((moisture_levels(:)-y_mean).^2);
r_squared_w = 1-s_res/s_tot;
mse_w = immse(estimated_vwc(:), moisture_levels(:));
fprintf('w Correlation: %.4f, R^2: %.4f, MSE: %.3f\n', correlation_w, r_squared_w, mse_w);


f=figure(4);clf;
f.Position = [10 10 400 350];
scatter(moisture_levels(:), estimated_w(:)*100,30, 'filled');
hold on;
x = linspace(0, 45, 20);
plot(x, x, '--k', 'LineWidth', 2);
xlabel('Ground truth VWC (%)');
ylabel('Estimated VWC (%)');
txt2 = sprintf('R^2: %.3f', r_squared_w);
txt1 = sprintf('Corr: %.3f', correlation_w);
txt3 = sprintf('MSE: %.3f', mse_w);
txt = {txt1, txt2, txt3};
text(0.5,35,txt, 'FontSize', 20)
set(gca,'FontSize',24);
legend({'Data points','y=x'}, 'Location','southeast');


