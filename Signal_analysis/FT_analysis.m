clear all, close all, clc
filename = '20181202_25sec.mp3';
[y, fs] = audioread(filename);
%sound(y, fs)

%%% Denoising techniques for Underwater Ambient noise

%% Fourier Transform
% Basic spectral analysis
n = length(y);
newn = pow2(nextpow2(n));
dfty = fft(y,newn);
newf = (0:newn-1)*(fs/newn)/10;
power = abs(dfty).^2/newn;
figure
subplot(2,1,1)
plot(newf(1:floor(newn/2))./1000,10*log10(power(1:floor(newn/2))))
title('Welch’s power spectral density estimate, with input resized')
xlabel('Frequency (kHz)')
ylabel('PSD log scale (dB/kHz)')
axis tight
subplot(2,1,2)
plot(newf(1:floor(newn/2))./1000,power(1:floor(newn/2)))
xlabel('Frequency (kHz)')
ylabel('PSD (dB/kHz)')
axis tight
%Power spectrum density
[pxx, f] = pwelch(y, fs);
figure
subplot(2,1,1)
plot(f, 10*log10(pxx))
xlabel('Frequency (kHz)')
ylabel('PSD log scale(dB/kHz)')
title('Welch’s power spectral density estimate')
subplot(2,1,2)
plot(f,pxx)
xlabel('Frequency (kHz)')
ylabel('PSD (dB/kHz)')
%From the graphs you can see a mirrored step function 
% with tiny spikes,we want to get rid of these tiny 
% spikes
% Cannot denoise by thresholding the fourier 
% coefficients and then ifft

%% Moving average filter
% step = 100;
% coeff = ones(1, step)/step;
% avgpw = filter(coeff,1,10*log10(power(1:floor(newn/2))));
% fDelay = (length(coeff)-1)/2./1000;
% figure
% subplot(2,1,1)
% plot(newf(1:floor(newn/2))./1000, [10*log10(power(1:floor(newn/2))) avgpw])
% legend('PSD (dB/kHz)','100kHz Average')
% xlabel('Frequency (kHz)')
% ylabel('PSD log scale(dB/kHz)')
% title('Welch’s power spectral density estimate (No delay)')
% axis tight
% % Accounting for half step delay
% subplot(2,1,2)
% plot(newf(1:floor(newn/2))./1000, 10*log10(power(1:floor(newn/2))), ...
%     newf(1:floor(newn/2))./1000-fDelay,avgpw)
% legend('PSD (dB/kHz)','100kHz Average')
% xlabel('Frequency (kHz)')
% ylabel('PSD log scale(dB/kHz)')
% title('Welch’s power spectral density estimate (with delay)')
% axis tight

%on frequency
%dfty = fft(y,newn);
%newf = (0:newn-1)*(fs/newn)/10;
step = 100;
coeff = ones(1, step)/step;
avgdfty = filter(coeff,1,dfty);
fDelay = (length(coeff)-1)/2./1000;
%power = abs(dfty).^2/newn;
avgpw = abs(avgdfty).^2/newn;
avgpw = 10*log10(avgpw(1:floor(newn/2)));
figure
subplot(2,1,1)
plot(newf(1:floor(newn/2))./1000, [10*log10(power(1:floor(newn/2))) avgpw])
legend('PSD (dB/kHz)','100kHz Average')
xlabel('Frequency (kHz)')
ylabel('PSD log scale(dB/kHz)')
title('Welch’s power spectral density estimate (No delay)')
axis tight
% Accounting for half step delay
subplot(2,1,2)
plot(newf(1:floor(newn/2))./1000, 10*log10(power(1:floor(newn/2))), ...
    newf(1:floor(newn/2))./1000-fDelay,avgpw)
legend('PSD (dB/kHz)','100kHz Average')
xlabel('Frequency (kHz)')
ylabel('PSD log scale(dB/kHz)')
title('Welch’s power spectral density estimate (with delay)')
axis tight
%% Weighted moving average filter and gaussian expansion filter
% h = [1/2 1/2];
% binomialCoeff = conv(h,h);
% for n = 1:4
%     binomialCoeff = conv(binomialCoeff,h);
% end
% bDelay = (length(binomialCoeff)-1)/2;
% binomialMA = filter(binomialCoeff, 1, 10*log10(power(1:floor(newn/2))));
% alpha = 0.45;
% exponentialMA = filter(alpha, [1 alpha-1], 10*log10(power(1:floor(newn/2))));
% plot(newf(1:floor(newn/2))./1000, 10*log10(power(1:floor(newn/2))), ...
%     newf(1:floor(newn/2))./1000-bDelay./1000,binomialMA, ...
%     newf(1:floor(newn/2))./1000-(1/step)./1000,exponentialMA)
% legend('PSD (dB/kHz)','Binomial weighted Average', 'Exponential Weighted Average')
% xlabel('Frequency (kHz)')
% ylabel('PSD log scale(dB/kHz)')
% title('Welch’s power spectral density estimate')
% axis tight

%on frequency
h = [1/2 1/2];
binomialCoeff = conv(h,h);
for n = 1:4
    binomialCoeff = conv(binomialCoeff,h);
end
bDelay = (length(binomialCoeff)-1)/2;
binomialMA = filter(binomialCoeff, 1, dfty);
cubicpw = abs(binomialMA).^2/newn;
cubicpw = 10*log10(cubicpw(1:floor(newn/2)));

alpha = 0.45;
exponentialMA = filter(alpha, [1 alpha-1], dfty);
exponentialpw = abs(exponentialMA).^2/newn;
exponentialpw = 10*log10(exponentialpw(1:floor(newn/2)));

plot(newf(1:floor(newn/2))./1000, 10*log10(power(1:floor(newn/2))), ...
    newf(1:floor(newn/2))./1000-bDelay./1000,cubicpw, ...
    newf(1:floor(newn/2))./1000-(1/step)./1000,exponentialpw)
legend('PSD (dB/kHz)','Binomial weighted Average', 'Exponential Weighted Average')
xlabel('Frequency (kHz)')
ylabel('PSD log scale(dB/kHz)')
title('Welch’s power spectral density estimate')
axis tight
%% Savitzky-Golay finite impulse response (FIR) smoothing filter
% y = sgolayfilt(x,order,framelen)
% cubicMA   = sgolayfilt(10*log10(power(1:floor(newn/2))), 3, 7);
% quarticMA = sgolayfilt(10*log10(power(1:floor(newn/2))), 4, 7);
% quinticMA = sgolayfilt(10*log10(power(1:floor(newn/2))), 5, 9);
% plot(newf(1:floor(newn/2))./1000,[10*log10(power(1:floor(newn/2))) cubicMA quarticMA quinticMA])
% legend('PSD (dB/kHz)','Cubic-Weighted MA', 'Quartic-Weighted MA', ...
%        'Quintic-Weighted MA')
% xlabel('Frequency (kHz)')
% ylabel('PSD log scale(dB/kHz)')
% title('Welch’s power spectral density estimate')
% axis tight

%on frequency
cubicMA   = sgolayfilt(dfty, 3, 7);
cubicpw = abs(cubicMA).^2/newn;
cubicpw = 10*log10(cubicpw(1:floor(newn/2)));

quarticMA = sgolayfilt(dfty, 4, 7);
quarticpw = abs(quarticMA).^2/newn;
quarticpw = 10*log10(quarticpw(1:floor(newn/2)));

quinticMA = sgolayfilt(dfty, 5, 9);
quinticpw = abs(quarticMA).^2/newn;
quinticpw = 10*log10(quinticpw(1:floor(newn/2)));

plot(newf(1:floor(newn/2))./1000,[10*log10(power(1:floor(newn/2))) cubicpw quarticpw quinticpw])
legend('PSD (dB/kHz)','Cubic-Weighted MA', 'Quartic-Weighted MA', ...
       'Quintic-Weighted MA')
xlabel('Frequency (kHz)')
ylabel('PSD log scale(dB/kHz)')
title('Welch’s power spectral density estimate')
axis tight
%% Median average
% yMedFilt2 = medfilt1(10*log10(power(1:floor(newn/2))),12,'truncate');
% yMedFilt10 = medfilt1(10*log10(power(1:floor(newn/2))),10,'truncate');
% yMedFilt12 = medfilt1(10*log10(power(1:floor(newn/2))),12,'truncate');
% figure
% plot(newf(1:floor(newn/2))./1000,[10*log10(power(1:floor(newn/2))) yMedFilt2 ...
%     yMedFilt10 yMedFilt12])
% legend('PSD (dB/kHz)','Median filter 2nd degree', 'Median filter 10th degree', ...
%     'Median filter 12th degree')
% xlabel('Frequency (kHz)')
% ylabel('PSD log scale(dB/kHz)')
% title('Welch’s power spectral density estimate')
% axis tight

%on frequency
yMedFilt2 = medfilt1(abs(dfty),2,'truncate');
yMedFilt2 = yMedFilt2.^2/newn;
yMedFilt2pw = 10*log10(yMedFilt2(1:floor(newn/2)));

yMedFilt10 = medfilt1(abs(dfty),10,'truncate');
yMedFilt10 = yMedFilt10.^2/newn;
yMedFilt10pw = 10*log10(yMedFilt10(1:floor(newn/2)));

yMedFilt12 = medfilt1(abs(dfty),12,'truncate');
yMedFilt12 = yMedFilt12.^2/newn;
yMedFilt12pw = 10*log10(yMedFilt12(1:floor(newn/2)));

% yMedFilt2 = medfilt1(abs(dfty).^2/newn,12,'truncate');
% yMedFilt2pw = 10*log10(yMedFilt2(1:floor(newn/2)));
% 
% yMedFilt10 = medfilt1(abs(dfty).^2/newn,10,'truncate');
% yMedFilt10pw = 10*log10(yMedFilt10(1:floor(newn/2)));
% 
% yMedFilt12 = medfilt1(abs(dfty).^2/newn,12,'truncate');
% yMedFilt12pw = 10*log10(yMedFilt12(1:floor(newn/2)));

figure
plot(newf(1:floor(newn/2))./1000,[10*log10(power(1:floor(newn/2))) yMedFilt2pw ...
    yMedFilt10pw yMedFilt12pw])
legend('PSD (dB/kHz)','Median filter 2nd degree', 'Median filter 10th degree', ...
    'Median filter 12th degree')
xlabel('Frequency (kHz)')
ylabel('PSD log scale(dB/kHz)')
title('Welch’s power spectral density estimate')
axis tight
%% Hampel signal(time domain only)
[yhampel1,i,xmedian,xsigma] = hampel(abs(dfty));
yhampel = yhampel1.^2/newn;
xmedian = xmedian.^2/newn;
xsigma = xsigma.^2/newn;
ypow = 10*log10(yhampel(1:floor(newn/2)));
medpow = 10*log10(xmedian(1:floor(newn/2)));
sigpow = 10*log10(xsigma(1:floor(newn/2)));

figure
plot(newf(1:floor(newn/2))./1000, [10*log10(power(1:floor(newn/2))) ypow])
legend('PSD (dB/kHz)','Hampel filter')
xlabel('Frequency (kHz)')
ylabel('PSD log scale(dB/kHz)')
title('Welch’s power spectral density estimate')
axis tight

% hold on
% plot(newf(1:floor(newn/2))./1000, medpow-3*sigpow, newf(1:floor(newn/2))./1000, medpow+3*sigpow)
% x = newf(1:floor(newn/2))./1000;
% plot(find(i),x(i),'sk')
% hold off
% legend('Original signal','Lower limit','Upper limit','Outliers')

% hampel(10*log10(power(1:floor(newn/2))), 13)
% legend('PSD (dB/kHz)','Hampel filtered PSD')
% xlabel('Frequency (kHz)')
% ylabel('PSD log scale(dB/kHz)')
% title('Welch’s power spectral density estimate')
% axis tight

%% To listen to the audio back and save it
original = ifft(dfty, newn);
sound(original,fs)
% try ifft(filtered frequencey, newn, 'symmetric')
% sound(iffted file,fs)
% Moving average 
moving_final = ifft(avgdfty,'symmetric');
sound(moving_final,fs)
audiowrite('Moving_average.wav', moving_final,fs);
% Binomial weighted average filtered audio (better than the expo one)
bin_final = ifft(binomialMA,'symmetric');
sound(bin_final,fs)
audiowrite('Binomial_weighted_average.wav', bin_final,fs);
% Exponential weighted average filtered audio
expo_final = ifft(exponentialMA,'symmetric');
sound(expo_final,fs)
audiowrite('Exponential_weighted_average.wav', expo_final,fs);
% Savitzky-Golay filtered signal (worse than exponential)
quartic_final = ifft(quarticMA,'symmetric');
sound(quartic_final,fs) %quintic>quartic~cubic
audiowrite('Quartic_SG.wav', quartic_final,fs);
quintic_final = ifft(quinticMA,'symmetric');
sound(quintic_final,fs)
audiowrite('Quintic_SG.wav', quintic_final,fs);
cubic_final = ifft(cubicMA,'symmetric');
sound(cubic_final,fs)
audiowrite('Cubic_SG.wav', cubic_final,fs);
% Median average 
yMedFilt2_final = ifft(medfilt1(abs(dfty),2,'truncate'),'symmetric');
sound(yMedFilt2_final,fs) %too much boat
audiowrite('MedianAVG2.wav', yMedFilt2_final,fs); 
yMedFilt10_final = ifft(medfilt1(abs(dfty),10,'truncate'),'symmetric');
sound(yMedFilt10_final,fs) %boat noise quiting down in the second half
audiowrite('MedianAVG10.wav', yMedFilt10_final,fs); 
yMedFilt12_final = ifft(medfilt1(abs(dfty),12,'truncate'),'symmetric');
sound(yMedFilt12_final,fs) %too much soundwaves
audiowrite('MedianAVG12.wav', yMedFilt12_final,fs); %only noise
%hampel 
hampel_final = ifft(yhampel1,'symmetric');
sound(hampel_final,fs) %almost no orcas
audiowrite('Hampel.wav', hampel_final,fs);