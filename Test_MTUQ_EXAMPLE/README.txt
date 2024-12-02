This example demonstrates how to calculate confidence curves in MTUQ using only surface waves, based on the default GridSearch.DoubleCouple.py example:

https://github.com/uafgeotools/mtuq/blob/master/examples/GridSearch.DoubleCouple.py

1. Run the MTUQ grid search

Open your terminal and execute the following command:

python syngine_GFs_GridSearch.DC_SW_Confidence_Curve.py -event 20090407201255351 -evla 61.454 -evlo -149.742 -evdp 33033.59 -mw 4.5 -time 2009-04-07T20:12:55.00000Z -np 40 -fb 16-40 -wl 150

This command runs the grid search with parameters similar to the original MTUQ example, but focuses on surface waves. Feel free to modify parameters to explore other solutions. 

After the grid search completes, you'll find the following output files in the OUTPUT_20090407201255351DC directory:

-20090407201255351DC_beachball.png
-20090407201255351DC_data_stats.txt
-20090407201255351DC_header_info_sw.txt
-20090407201255351DC_likelihood.png
-20090407201255351DC_likelihoods_angles.nc
-20090407201255351DC_misfit.nc
-20090407201255351DC_misfit.png
-20090407201255351DC_solution.json
-20090407201255351DC_waveforms.png


The 20090407201255351DC_likelihoods_angles.nc file is crucial for calculating the confidence curve, as it contains the likelihoods of different moment tensor solutions, organized by their angular distance from the best-fitting solution.

2. Calculate the confidence curve

Run the following command in your terminal:

python confidence_curve_calculation.py

After running the script, you'll obtain these PDF files in OUTPUT_20090407201255351DC:

angle_vs_cumulative_likelihood_and_derivative.pdf
confidence_curve.pdf
derivative_V_and_V.pdf