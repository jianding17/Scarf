function e = compute_e_soil(oc, w, e_soil, e_bound, e_air, e_free)
    v_clay = 0.4;
    v_silt = 0.4;
    v_sand = 0.2;
%     v_clay = 0.4+0.1;
%     v_silt = 0.4-0.1;
%     v_sand = 0.2;
%     v_clay = 0.4+0.3;
%     v_silt = 0.4-0.3;
%     v_sand = 0.2-0;
    wilting_point = 0.02982 + 0.089 * v_clay + 0.00786 * oc / 0.58;
%     bulk_density = -0.0131 * oc + 0.9904;
    bulk_density = -0.0197 * oc + 1.0667;
    p = 0.6819 - 0.0648 ./ (oc + 1) - 0.119 * bulk_density.^2 - 0.02668 ...
        + 0.1489 * v_clay + 0.08031 * v_silt + 0.02321 ./ ((oc + 1) .* (bulk_density.^2)) ...
        - 0.01908 * bulk_density.^2 - 0.1109 * v_clay - 0.2315 * v_clay * v_silt ...
        - 0.01197 * v_silt* bulk_density.^2 - 0.01068 * v_clay * bulk_density.^2;
    if w < wilting_point
        e_eff = (1 - p) * e_soil + w .* e_bound + (p - w) * e_air;
    elseif w > wilting_point && w <= p
        e_eff = (1 - p) * e_soil + ...
                 w .* ((p - w) / (p - wilting_point)* e_bound + (w - wilting_point) / (p - wilting_point) * e_free) + ...
                 (p - w) * e_air;   
    else
        e_eff = (1 - w) * e_soil + w .* e_free;
    end
    
    e = real(e_eff) / 2 .* (sqrt(1 + (imag(e_eff) ./ real(e_eff)) .^ 2) + 1);
end