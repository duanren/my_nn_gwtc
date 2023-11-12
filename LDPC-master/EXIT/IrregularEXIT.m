clear

% vn_degree = [1 2 3 9];%dv - 1.
% vn_portion = [0.267 0.176 0.127 0.430];%edge portion
% cn_degree = [4 7];%dc - 1.
% cn_portion = [0.113 0.887];%edge portion

vn_degree = [1 2 3 4 5]; %dv - 1
vn_edge_portion = [0.5133 0.4505 0 0 0.0361];
cn_degree = [1 2]; %dc - 1
cn_edge_portion = [0.3533 0.6467];

I_AV = 0.001 : 0.001 : 0.999;
I_EV = zeros(length(I_AV), 1);
EbN0 = 0.5;%dB
R = 0.5;
sigma_ch = sqrt(8 * R * 10^(EbN0/10));%sigma_ch^2 = 4 sigma_w^2 = 8 R Eb/N0.
for i = 1 : length(I_AV)
    for i_vn = 1 : length(vn_degree)
        I_EV(i) = I_EV(i) + vn_edge_portion(i_vn) * J(sqrt(vn_degree(i_vn) * Jinv(I_AV(i))^2 + sigma_ch^2));
    end
end
plot(I_AV, I_EV);
hold on;

I_AC = 0.001 : 0.001 : 0.999;
I_EC = zeros(length(I_AC), 1);
for i = 1 : length(I_AC)
    for i_cn = 1 : length(cn_degree)
        I_EC(i) = I_EC(i) + cn_edge_portion(i_cn) * (1 - J(sqrt(cn_degree(i_cn) * Jinv(1 - I_AC(i))^2)));
    end
end
plot(I_EC, I_AC);
hold on;
