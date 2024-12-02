#!/usr/bin/env python

import os
import numpy as np
import argparse
from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens1, plot_beachball, plot_misfit_dc
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit  
from mtuq.misfit.waveform import estimate_sigma, calculate_norm_data
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.graphics import plot_likelihood_dc
from mtuq.util import dataarray_idxmin,dataarray_idxmax
from mtuq.util.math import to_mij
from mtuq.graphics.uq.double_couple import _likelihoods_dc_regular

class Header:
    def __init__(self,station,component,time_shift,cc):
        self.station = station
        self.component = component
        self.time_shift = time_shift
        self.cc = cc

def parse_args():

    parser = argparse.ArgumentParser(
    description="Input event info run MTUQ",
    formatter_class=argparse.RawTextHelpFormatter,
                                     )

    #parser.add_argument("-mdir",type=str,help="Main dir: -mdir /Users/felix/Documents/INVESTIGACION/2_FEB_JUL_2022/IRIS_WORKSHOP/MTUQ_INVERSIONS/FK_VS_1D_CUBIC_MESH/gs_mtuq")
    parser.add_argument("-event",type=str,help="event (event directory must be in main dir): -event 20090407201255351")
    parser.add_argument("-evla",type=str,help="Event latitude: -evla 61.454")
    parser.add_argument("-evlo",type=str,help="Event longitude: -evlo -149.742")
    parser.add_argument("-evdp",type=str,help="Event depth in m: -evdp 33033.59")
    parser.add_argument("-mw",type=float,help="Event magnitude: -mw 4.5")
    parser.add_argument("-time", type=str,help="Earthquake origin time: -time 2009-04-07T20:12:55.00000Z")
    parser.add_argument("-np", type=int,help="Number of points per axis: -np 40")
    parser.add_argument("-fb",type=str,help="Frequency band for filtering data in seconds: -fb 16-40")
    parser.add_argument("-wl",type=str,help="Window length in seconds: -wl 150")
   
    return parser.parse_args()

def _getattr(trace, name, *args):
    if len(args)==1:
        if not hasattr(trace, 'attrs'):
            return args[0]
        else:
            return getattr(trace.attrs, name, args[0])
    elif len(args)==0:
        return getattr(trace.attrs, name)
    else:
        raise TypeError("Wrong number of arguments")
    
def get_headerinfo(data,greens,misfit,stations,origin,source):
    #data_bw,data_sw,greens_bw,greens_sw,misfit_bw,misfit_sw,stations,origin,best_mt

    synthetics = misfit.collect_synthetics(data, greens.select(origin), source)

    header_info = []

    for _i in range(len(stations)):
        stream_dat = data[_i]
        stream_syn = synthetics[_i] 

        for dat in stream_dat:
            component = dat.stats.channel[-1].upper()
            try:
                syn = stream_syn.select(component=component)[0]
            except:
                warn('Missing component, skipping...')
                continue

            time_shift = 0.
            time_shift += _getattr(syn, 'time_shift', np.nan)
            time_shift += _getattr(dat, 'static_time_shift', 0)

            s = syn.data
            d = dat.data
            # display maximum cross-correlation coefficient
            Ns = np.dot(s,s)**0.5
            Nd = np.dot(d,d)**0.5

            if Ns*Nd > 0.:
                max_cc = np.correlate(s, d, 'valid').max()
                max_cc /= (Ns*Nd)
            else:
                max_cc = np.nan
                
            header_info.append(Header(stations[_i]['station'],component,np.round(time_shift,2),np.round(max_cc,2)))
            print('{},{}: {} {}'.format(stations[_i]['station'],component,np.round(time_shift,2),np.round(max_cc,2)))

    return(header_info)

def wrap_up(ts_list,cc_list,station):
    total_ts = np.round(np.nansum(np.abs(ts_list))/(ts_list.size - np.count_nonzero(np.isnan(ts_list))),2)
    total_cc = np.round(np.nansum(cc_list)/(cc_list.size - np.count_nonzero(np.isnan(cc_list))),2)
            
    line = '{} {} {} {} {} {} {} {} {}'.format(station,ts_list[0],ts_list[1],ts_list[2],total_ts,cc_list[0],cc_list[1],cc_list[2],total_cc)
    return(line)

def write_headers(header_info,event_id,type):
    open_header_file=open('{}DC_header_info_{}.txt'.format(event_id,type),'w')

    open_header_file.write('STATIONS  ts_Z ts_R ts_T abs_av_shift cc_Z cc_R cc_T av_cc\n')

    init_stat = header_info[0].station
    ts_list = np.array([np.nan,np.nan,np.nan])
    cc_list = np.array([np.nan,np.nan,np.nan])

    for i in range(len(header_info)):

        if header_info[i].station != init_stat:

            line = wrap_up(ts_list,cc_list,init_stat)
            open_header_file.write(line+'\n')
            print(line)

            ts_list = np.array([np.nan,np.nan,np.nan])
            cc_list = np.array([np.nan,np.nan,np.nan])

            init_stat = header_info[i].station

        if header_info[i].station == init_stat:
            if header_info[i].component == 'Z':
                ts_list[0] = header_info[i].time_shift
                cc_list[0] = header_info[i].cc

            if header_info[i].component == 'R':
                ts_list[1] = header_info[i].time_shift
                cc_list[1] = header_info[i].cc

            if header_info[i].component == 'T':
                ts_list[2] = header_info[i].time_shift
                cc_list[2] = header_info[i].cc

            if i == len(header_info)-1:
                line = wrap_up(ts_list,cc_list,header_info[i].station)
                open_header_file.write(line)
                print(line)

def calculate_variance(best_mt,data_sw,greens_sw,misfit_sw):
    
    print('Data variance estimation...\n')

    sigma_sw = estimate_sigma(data_sw,greens_sw,
        best_mt, misfit_sw.norm, ['Z', 'R', 'T'],
        misfit_sw.time_shift_min, misfit_sw.time_shift_max)

    stats = {'sigma_sw': sigma_sw,
            'variance_sw': sigma_sw**2}
    
    print('Sigma = {}'.format(sigma_sw))
    print('Variance = {}'.format(sigma_sw**2))

    return(stats)

def run_mt_estimation(param,mdir):
    
    path_data=    fullpath('{}/{}/*.[zrt]'.format(mdir,param.event))
    path_weights= fullpath('{}/{}/weights.dat'.format(mdir,param.event))
    event_id=     '{}'.format(param.event)
    #model='prem_i'
    model='ak135'

    wl_sw = float(param.wl)

    #
    # Body and surface wave measurements will be made separately
    #

    freqs = param.fb.split('-')
    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=1/float(freqs[1]),
        freq_max=1/float(freqs[0]),
        pick_type='taup',
        #taup_model='prem',
        taup_model='ak135',
        window_type='surface_wave',
        window_length=wl_sw,
        capuaf_file=path_weights,
        )

    #
    # For our objective function, we will use surface waves contributions
    # 

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        )

    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #
    station_id_list = parse_station_codes(path_weights)
    #
    # Next, we specify the moment tensor grid and source-time function
    #
    magnitudes=np.arange(param.mw-0.2,param.mw+0.2,0.1)
    grid = DoubleCoupleGridRegular(
        npts_per_axis=param.np,
        magnitudes= param.mw)
        #magnitudes= magnitudes.tolist())

    wavelet = Trapezoid(
        magnitude=param.mw)


    origin = Origin({
        'time': '{}'.format(param.time),
        'latitude': param.evla,
        'longitude': param.evlo,
        'depth_in_m': param.evdp,
        })

    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac', 
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:m', 'type:velocity']) 

        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_sw = data.map(process_sw)

        print('Reading Greens functions...\n')
        #Stream GFs from syngine: https://ds.iris.edu/ds/products/syngine/
        greens = download_greens_tensors(stations, origin, model)
        #greens_tensors = db.get_greens_tensors(stations, origin)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_sw = greens.map(process_sw)
        
    else:
        stations = None
        data_sw = None
        greens_sw = None

    stations = comm.bcast(stations, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)

    if comm.rank==0:
        print('Evaluating surface wave misfit...\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, grid)
    

    if comm.rank==0:

        results = results_sw

        #
        # Collect information about best-fitting source
        #

        source_idx = results.source_idxmin()
        best_mt = grid.get(source_idx)

        # dictionary of lune parameters
        lune_dict = grid.get_dict(source_idx)

        # dictionary of Mij parameters
        mt_dict = best_mt.as_dict()

        merged_dict = merge_dicts(
            mt_dict,origin,lune_dict, {'M0': best_mt.moment()},
            {'Mw': best_mt.magnitude()})
        
        data_stats = save_results(event_id,results,data_sw,greens_sw,misfit_sw,stations,origin,best_mt,merged_dict)
        uncertainty_analysis(event_id,results,data_stats)
        make_figures(event_id,results,data_sw, greens_sw, process_sw,misfit_sw, stations, origin, best_mt, lune_dict, data_stats)

        os.system('mkdir OUTPUT_{}DC'.format(param.event))
        mv_files = 'mv {}DC_* OUTPUT_{}DC/'.format(param.event,param.event)
        os.system(mv_files)
    
        
        print('\nFinished MT estimation\n')
    
def save_results(event_id,results,data_sw, greens_sw,misfit_sw, stations, origin, best_mt,merged_dict):

    print('Saving results...\n')

    # Save Time-shifts and cross-correlation values
    header_info_sw = get_headerinfo(data_sw,greens_sw,misfit_sw,stations,origin,best_mt)
    type = 'sw'
    write_headers(header_info_sw,event_id,type)   

    # save best-fitting source
    save_json(event_id+'DC_solution.json', merged_dict)

    # save misfit surface
    results.save(event_id+'DC_misfit.nc')   

    # Calculating and saving Sigma and Variance
    data_stats = calculate_variance(best_mt,data_sw,greens_sw,misfit_sw)
    print(data_stats['variance_sw'])

    name_file = '{}DC_data_stats.txt'.format(event_id)

    open_file = open(name_file,'w')
    open_file.write('Sigma = {}\n'.format(data_stats['sigma_sw']))
    open_file.write('Variance = {}'.format(data_stats['variance_sw']))
    open_file.close()
    
    return(data_stats)

def make_figures(event_id,results,data_sw, greens_sw, process_sw,misfit_sw, stations, origin, best_mt, lune_dict, data_stats):
    #
    # Generate figures and save results
    #

    print('Generating figures...\n')

    plot_data_greens1(event_id+'DC_waveforms.png',
        data_sw, greens_sw, process_sw, 
        misfit_sw, stations, origin, best_mt, lune_dict)

    plot_beachball(event_id+'DC_beachball.png',
        best_mt, stations, origin)

    plot_misfit_dc(event_id+'DC_misfit.png', results)

    name_file_dc = '{}DC_likelihood.png'.format(event_id)
    plot_likelihood_dc(name_file_dc,results,data_stats['variance_sw'])

def uncertainty_analysis(event_id,results,data_stats):
    print('********')
    print('Results')
    print(results)

    likelihoods = _likelihoods_dc_regular(results,data_stats['variance_sw'])
    print('Likelihoods')
    print(likelihoods)

    #Extract the reference moment tensor:
    min_misfit = results.min()
    min_misfit_coords = results.where(results == min_misfit, drop=True).coords
    rho_min = min_misfit_coords['rho'].values # magnitude
    v_min =  min_misfit_coords['v'].values #gamma, clvd
    w_min = min_misfit_coords['w'].values #delta, iso
    kappa_min = min_misfit_coords['kappa'].values #strike
    sigma_min = min_misfit_coords['sigma'].values #slip
    h_min = min_misfit_coords['h'].values #cos(dip)
    origin_idx_min = min_misfit_coords['origin_idx'].values #depth
    #moment tensor normalized
    mt_ref = norm_mt(to_mij(rho_min,v_min,w_min,kappa_min,sigma_min,h_min))
    print('Ref coordinates (kappa_ref, sigma_ref, h_ref):')
    print(kappa_min,sigma_min,h_min)
    print('ref MT')
    print(mt_ref)
    #ref_mt = []

    #Conver the dataarray likelihoods into a dataframe adding a two new dimensions: moment tensor, and moment tensor angle.
    rho = 1
    likelihoods_angles_df = likelihoods.to_dataframe(name='likelihoods').reset_index()
    likelihoods_angles_df['mt'] = likelihoods_angles_df.apply(lambda row: norm_mt(to_mij(rho, 0, 0, row['kappa'], row['sigma'], row['h'])), axis=1)
    likelihoods_angles_df['mt_angle'] = likelihoods_angles_df.apply(lambda row: MT_angle(row['mt'], mt_ref[0]), axis=1)
    
    print('likelihoods_angles_df')
    print(likelihoods_angles_df)
    print('Likelihoods dataframe header')
    print(likelihoods_angles_df.head())

    #This part is to double check that the MT angle at the reference MT coordinates is zero.
    #double_check_dataframe(likelihoods_angles_df)

    #Save the likelihood dataframe
    #Since one of the variables is the moment tensor, for saving the dataframe we need to drop the MT column
    likelihoods_angles_df_save = likelihoods_angles_df.drop(columns=['mt'], errors='ignore')
    # Convert the DataFrame to an xarray Dataset
    likelihoods_angles_df_save= likelihoods_angles_df_save.set_index(['kappa', 'sigma', 'h']).to_xarray()
    # Save the Dataset as a NetCDF file
    likelihoods_angles_df_save.to_netcdf('{}DC_likelihoods_angles.nc'.format(event_id))

    print('********')

def norm_mt(mt):
    magnitude = np.sqrt(np.sum(np.square(mt)))
    norm_mt = mt/magnitude
    return(norm_mt)

def MT_angle(T1,T2):
    
    num = np.dot(T1,T2)
    den1 = (np.dot(T1,T1))**(0.5)
    den2 = (np.dot(T2,T2))**(0.5)
    den = den1*den2
    cos_an = np.round(num/den,3)
    angle = np.arccos(cos_an)
    angle = np.degrees(angle)

    return(angle)

def double_check_dataframe(likelihoods_angles_df):
    # Find the row with the minimum 'mt_angle' value
    min_angle_idx = likelihoods_angles_df['mt_angle'].idxmin()

    # Retrieve the coordinates of the row with the minimum 'mt_angle'
    min_angle_row = likelihoods_angles_df.loc[min_angle_idx]

    kappa_min = min_angle_row['kappa']
    sigma_min = min_angle_row['sigma']
    h_min = min_angle_row['h']
    min_mt_angle = min_angle_row['mt_angle']

    # Print the coordinates and the minimum angle
    print(f"Coordinates with minimum mt_angle: kappa={kappa_min}, sigma={sigma_min}, h={h_min}")
    print(f"Minimum mt_angle: {min_mt_angle}")

if __name__=='__main__':

    param = parse_args()
    print(param.event)
    mdir = os.getcwd()
    run_mt_estimation(param,mdir)