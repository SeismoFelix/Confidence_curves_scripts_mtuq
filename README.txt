
1. FK_GFs_GridSearch.DoubleCouple_SW_BW_options.py : Regular double couple grid-search using locally calculated GFs with FK. 

For running this scripts you have to provide some input parameters. Type:

Python  FK_GFs_GridSearch.DoubleCouple_SW_BW_options.py -h, for further information. 

For this event, type the following for launching the grid-search:

python FK_GFs_GridSearch.DoubleCouple_SW_BW_options.py -event 20171201023244 -evla 30.734 -evlo 57.39 -evdp 6000.0 -mw 6.0 -time 2017-12-01T02:32:44.000000Z -np 30 -fb 15-33

2. GridSearch.DoubleCouple_confidence.py : Random double couple grid-search using locally calculated GFs with FK. It also includes uncertainty curves (cdc, pdf, omega)

It runs similarly as (1):

Python GridSearch.DoubleCouple_confidence.py -event 20171201023244 -evla 30.734 -evlo 57.39 -evdp 6000.0 -mw 6.0 -time 2017-12-01T02:32:44.000000Z -np 30 -fb 15-33

3. GridSearch.DoubleCouple_random.py : Random double couple grid-search using streamed GFs with ak135 global velocity model. It also includes uncertainty curves (cdc, pdf, omega)

It runs just typing:

Python GridSearch.DoubleCouple_random.py

NOTE: for testing purposes, before running GridSearch.DoubleCouple_random.py use the weights_short.dat file that contains an small sample of all the stations available (cp weights_short.dat weights.dat)
