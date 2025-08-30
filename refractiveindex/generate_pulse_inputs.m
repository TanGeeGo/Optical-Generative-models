% clear all
load('RefIndexMeasurements.mat')

load('TD_1.mat');
time_signal = R_fixed(:,3701:8000);

load('TD_2.mat');
time_signal = [time_signal; R_fixed(:,3701:8000)];

load('TD_3.mat');
R_fixed(:,end-12:end)=[];
time_signal = [time_signal; R_fixed(:,3701:8000)];

number_of_wavelenghts =29;
% broadband_f = 1e12*linspace(0.35,1.65,number_of_wavelenghts);
% f_new = 1e12*linspace(0.35,1.65,number_of_wavelenghts*801);

broadband_f = 1e12*[0.35:(1.65-0.35)/(number_of_wavelenghts-1):1.65];
f_new = 1e12*[0.35:(1.65-0.35)/(801*(number_of_wavelenghts-1)):1.65];

fs=1/(time(2)-time(1));
r_f = 0:fs/4300:fs;
f_max = size(time_signal,2)/2;
r_f = r_f(1:f_max);

n_plastic_new = interp1(f,n_plastic,f_new);


for i=1:size(time_signal,1)
%     size(time_signal,1)
    fft_signal = fft(time_signal(i,:));
    
    r_fM = fft_signal(1:f_max);
    
    r_fM_new = interp1(r_f,abs(r_fM),f_new,'nearest');
    r_fPhase_new = interp1(r_f,unwrap(angle(r_fM)),f_new,'linear');

    %corr_factor = zeros(1,number_of_wavelenghts);
    for j=1:number_of_wavelenghts
       relative_magnitude(i,j) = r_fM_new(f_new == broadband_f(j));
       relative_phase(i,j) =  r_fPhase_new(f_new == broadband_f(j));
       n_wavelengths(j) = n_plastic_new(f_new == broadband_f(j));
    end

end




correction_factor = 2.*pi.*(n_wavelengths-1).*broadband_f./(299792458*10^6);

corr_factor = '[';
broadband_lamda_str = '[';
for j=1:number_of_wavelenghts
    corr_factor = [corr_factor ',' num2str(correction_factor(j))];
    broadband_lamda_str = [broadband_lamda_str ',' num2str((10^9)*299792458/broadband_f(j))];
end
corr_factor = [corr_factor ']'];
broadband_lamda_str = [broadband_lamda_str ']'];

save('relative_magnitude_v2.mat','relative_magnitude');
save('relative_phase_v2.mat','relative_phase');

signal_power=sum(time_signal(1,:).*time_signal(1,:));
output_signal = interp(time_signal(1,:),10);
interp_signal_power=sum(output_signal.*output_signal);
output_signal = sqrt(signal_power/interp_signal_power).* output_signal;
%figure;plot(output_signal);figure;plot(time_signal(1,:))

r_f = 0:fs/size(output_signal,2):fs;
f_max = size(output_signal,2)/2;
r_f = r_f(1:f_max);

fft_signal_o = fft(output_signal);
r_fM = fft_signal_o(1:f_max);

figure;plot(abs(r_fM))
%figure;plot(r_fPhase)

r_fM_new = interp1(r_f,abs(r_fM),f_new,'nearest');
r_fPhase_new = interp1(r_f,unwrap(angle(r_fM)),f_new,'nearest');

figure;plot(r_fM_new)
figure;plot(r_fPhase_new)

[~,index]=min(abs(r_f-min(f_new)));
rr=[r_f(333:1561)];
r_fM_new = interp1(f_new,abs(r_fM_new),rr,'linear');
r_fPhase_new = interp1(f_new,(r_fPhase_new),rr,'linear');

r_fM_new = [zeros(1,332) r_fPhase_new zeros(1,length(r_f)-1561)];
r_fPhase_new = [zeros(1,332) r_fPhase_new zeros(1,length(r_f)-1561)];

r_fM_new2 = [r_fM_new fliplr(r_fM_new)];
r_fPhase_new2 = [r_fPhase_new (-1)*fliplr(r_fPhase_new)];

figure;plot(r_fM_new2)
figure;plot(r_fPhase_new2)

dummy = r_fM_new2.*exp(1i*r_fPhase_new2);
dum = ifft(dummy);
figure;plot(real(dum));



figure;plot(r_f,unwrap(angle(r_fM)))
figure;plot(f_new,r_fPhase_new)

for j=1:number_of_wavelenghts
       relative_magnitude_o(1,j) = r_fM_new(f_new == broadband_f(j));
       relative_phase_o(1,j) =  r_fPhase_new(f_new == broadband_f(j));
       n_wavelengths(j) = n_plastic_new(f_new == broadband_f(j));
end
output(1,:) = relative_magnitude_o(1,:).*exp(1i.*relative_phase_o(1,:));

% time_o = time(3701:7500);
interp_rel_mag = spline(broadband_f,relative_magnitude_o,f_new);
interp_phase = spline(broadband_f,relative_phase_o,f_new);

%figure;plot(interp_rel_mag);hold on;plot(abs(r_fM))

%output(1,:) =rel.*exp(1i.*relative_phase_o(end,:));
save('output_v2.mat','output')




