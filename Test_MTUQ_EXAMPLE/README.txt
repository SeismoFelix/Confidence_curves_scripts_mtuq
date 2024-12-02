1. Run the MTUQ grid search only for surface waves example by typing in the terminal:

python syngine_GFs_GridSearch.DC_SW_Confidence_Curve.py -event 20090407201255351 -evla 61.454 -evlo -149.742 -evdp 33033.59 -mw 4.5 -time 2009-04-07T20:12:55.00000Z -np 40 -fb 16-40 -wl 150

The parameters are setup as they are in the original MTUQ example. Still, if you wish feel free to explore other parameters. 

After (1) the following files will be created in the directory OUTPUT_20090407201255351DC:


-20090407201255351DC_beachball.png
-20090407201255351DC_data_stats.txt
-20090407201255351DC_header_info_sw.txt
-20090407201255351DC_likelihood.png
-20090407201255351DC_likelihoods_angles.nc
-20090407201255351DC_misfit.nc
-20090407201255351DC_misfit.png
-20090407201255351DC_solution.json
-20090407201255351DC_waveforms.png


The file 20090407201255351DC_likelihoods_angles.nc is the most relevant for calculating the confidence curve since contain the information of the likelihood of other solutions sorted by their angular distance to the reference moment tensor. 


2. Calculate confidence curve by typing in the terminal: 

python confidence_curve_calculation.py

After (2) the followingg files will be created: 

angle_vs_cumulative_likelihood_and_derivative.pdf
confidence_curve.pdf
derivative_V_and_V.pdf