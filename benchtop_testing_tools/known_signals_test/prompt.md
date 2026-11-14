Write python solution (numpy, scipy, pandas, matplotlib only) that evaluates the "known signal" test of my 8-channel ADS1299 biosignal system. A function generator fed a sine wave to all 8 channels at once through a breakout board. I want to show that the recorded waveforms match the known input across different gain, amplitude and frequency. Ignore crosstalk. Keep the code simple and readable, with no unnecessary abstractions.

## Inputs
- Root folder: /home/karl/Documents/Master_Thesis/Testing/benchtop_testing/known_signal_test
- Test folders are named ALL_<gain>G_Sine_<amp>mVpp_<freq>Hz (e.g. ALL_1G_Sine_0.1mVpp_250Hz). Parse gain, nominal Vpp and frequency from the name.
- Data inside each test folder: 
    - File format and column layout of the PC recordings:  
    sample,status1_ok,status2_ok,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12,ch13,ch14,ch15,ch16,roll,pitch,yaw,accel_x,accel_y,accel_z
    80992,1,1,484,-897,-911,-1006,-919,-896,-929,-972,-767,-903,-912,-893,-769,-901,-881,-896,,,,,,
    80993,1,1,364,-899,-908,-1012,-914,-896,-926,-964,-770,-901,-916,-894,-766,-902,-883,-896,,,,,,
    80994,1,1,243,-899,-903,-1014,-914,-893,-923,-969,-769,-897,-922,-888,-761,-905,-885,-894,,,,,,
    80995,1,1,125,-899,-904,-1007,-919,-899,-925,-971,-769,-899,-921,-891,-756,-906,-884,-894,,,,,,
    
    only emg channels populated, imu is off.
    
    (describe, or I will paste the first lines)
- Filename convention that encodes driven channel, frequency, and amplitude: CH1_1G_Sine_2mVpp_100Hz_PC.csv - only use PC.csv endings. CH1: channel 1, 1G: gain 1, Sine: sine waveform, 2mVpp: amplitude, 100Hz: frequency, PC: recorded on PC and not on SD card.
- scripts should be stored here: /home/karl/repos/Sokosti_tools/benchtop_testing_tools/known_signals_test


- Only process folders matching the naming pattern above. Ignore everything else in the root folder (the "Pictures" folder and any .log files).
- Some conditions may be missing (e.g. ALL_8G_Sine_2mVpp_100Hz). Skip missing ones and list them in the output.

## System parameters
- Sample rate: 1000 SPS
- Vref: 4.5 V
- Number of channels: 16
- Gain is taken from the folder name, applied per file
- Code to volts: V = code * 2*Vref / (gain * 2^24), signed 24-bit two's complement (skip if the data is already in volts)
- Discard the first: check for consistency in sample numbers, and take 3 seconds of continuous data. 
- Analysis window: 3 s of uniterrupted data
- compare with analyze.py from crosstalk test. /home/karl/repos/Sokosti_tools/benchtop_testing_tools/crosstalk_test/analyze.py

## Per-file, per-channel analysis
1. Convert to volts and remove the transient.
2. Fit an ideal sine at the KNOWN nominal frequency with a 3-parameter linear least-squares fit (IEEE 1057): y ≈ a*sin(2πft) + b*cos(2πft) + c, using n/fs as the time base (or the data timestamps if present).
3. Compute:
   - fitted amplitude A = sqrt(a²+b²), measured Vpp = 2A
   - amplitude error % = (measured Vpp − nominal Vpp) / nominal Vpp * 100
   - residual = data − fit
   - SINAD (dB) = 20*log10(RMS(fit sine part) / RMS(residual)), where the sine part is the fit with the DC term removed. This equals 10*log10(P_signal / P_noise+distortion). RMS_sine is A/√2; the residual is noise + distortion + any fit mismatch. SINAD includes the signal generator's own noise and distortion.
   - ENOB = (SINAD − 1.76) / 6.02 (IEEE 1241). No full-scale normalization: it describes how well *this* signal is digitized, not the extrapolated full-scale performance.
4. Aggregate across the 8 channels: mean, std, min, max of measured Vpp, amplitude error %, SINAD, and ENOB.

## Methods note
The method follows the sine-fit approach of IEEE 1057 and the ADC characterization conventions of IEEE 1241. SINAD includes the signal generator's own noise and distortion (it characterizes generator + input chain + ADC, not the ADC alone). ENOB = (SINAD − 1.76) / 6.02 with no full-scale normalization.

## Outputs (save to <OUTPUT_PATH>)
1. results.csv: one row per condition with gain, nominal Vpp, freq, measured Vpp (mean, std), amplitude error % (mean, min, max), SINAD (mean, min, max), ENOB (mean, min, max).
2. results_table: a compact matrix, grouped by gain then nominal Vpp, with frequency across the columns, showing mean amplitude error %, mean SINAD (dB) as the main quality column, and ENOB (bits) as the secondary column. Also export it as LaTeX (booktabs).
3. Figure 1, data vs. fitted ideal sine: 3 representative conditions (8G/10mVpp/10Hz, 1G/2mVpp/100Hz, 1G/0.1mVpp/250Hz). For each, plot channel <CH> over ~3–5 periods with the fitted sine overlaid, and the residual in a thin subplot below. Shared style, readable axes in µV or mV as appropriate.
4. Figure 2, amplitude ratio (measured/nominal) vs. frequency (log x-axis): one line per (gain, amplitude) combination, markers at the tested frequencies, a horizontal reference line at 1.0, and error bars showing the spread across channels.
Save figures as PNG (300 dpi) and PDF.

## Notes to handle in code and comments
- Warn if the test frequency is at or above Nyquist for the given sample rate.
- Droop near Nyquist is expected from the ADS1299 sinc filter. Don't treat it as a bug.
- Print a short summary at the end: number of conditions processed, missing conditions, and the best and worst cases by SINAD.

## practical tips
just like in crosstalk test, make an analyze and seperate plot python script. so the analysys doesnt have to be rerun every time i just want to change the plots. compare /home/karl/repos/Sokosti_tools/benchtop_testing_tools/crosstalk_test. make an outputs folder as well.
for baseline, if necessary, use the baseline value from the crosstalk test, which is breakout baseline. compare with /home/karl/repos/Sokosti_tools/benchtop_testing_tools/crosstalk_test/prompt.md

Make minimal assumptions. If something in the data format is ambiguous, say so and ask before guessing. make no mistakes.