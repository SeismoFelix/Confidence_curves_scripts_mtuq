
#For Running this you need the 20171201023244DC_likelihoods_angles.nc, which is created after running FK_GFs_GridSearch.DC_SW_Confidence_Curve.py
import xarray as xr
from mtuq.graphics.uq.double_couple import _likelihoods_dc_regular
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def read_likelihoods_angles(netcdf_file):

    #da = xr.open_dataarray(netcdf_file)
    ds = xr.open_dataset(netcdf_file)

    return(ds)

def double_check_xarray(ds):
    #This verification double-checks that the coordinates corresponding to the minimum value of the array in the dimension mt_angle 
    #are the same coordinates corresponding the the maximum value of the array in the dimension likelihood

    # Retrieve the minimum value of 'mt_angle'
    min_value = ds['mt_angle'].min()
    min_angle_coords = ds['mt_angle'].where(ds['mt_angle']== min_value.values, drop=True).coords
    kappa_min_angle = min_angle_coords['kappa'].values #strike
    sigma_min_angle = min_angle_coords['sigma'].values #slip
    h_min_angle = min_angle_coords['h'].values #cos(dip)
    print("Minimum value of mt_angle (kappa:{},sigma:{},h_min:{})\n:".format(kappa_min_angle,sigma_min_angle,h_min_angle), min_value.values)  
    

    # Retrieve the maximum value of 'mt_angle'
    max_value_likelihood = ds['likelihoods'].max()
    max_likelihood_coords = ds['likelihoods'].where(ds['likelihoods']== max_value_likelihood.values, drop=True).coords
    kappa_max_likelihood = max_likelihood_coords['kappa'].values #strike
    sigma_max_likelihood = max_likelihood_coords['sigma'].values #slip
    h_max_likelihood = max_likelihood_coords['h'].values #cos(dip)
    print("Maximum value of likelihoods (kappa:{},sigma:{},h_min:{})\n:".format(kappa_max_likelihood,sigma_max_likelihood,h_max_likelihood),max_value_likelihood.values)  

def make_v_derivative_and_v(ds,nbins,constrain,critical_values,path,curve=True):
    # Access and clean the 'mt_angle' DataArray by dropping NaNs
    mt_angle_clean = ds['mt_angle'].dropna(dim='kappa').dropna(dim='sigma').dropna(dim='h')
    mt_angle_values = mt_angle_clean.values.flatten()

    # Compute the histogram
    bin_counts, bin_edges = np.histogram(np.radians(mt_angle_values), bins=nbins, density=True)

    # Compute the bin middle points
    bin_widths = np.diff(bin_edges)
    bin_middles = bin_edges[:-1] + bin_widths / 2

    #bin_counts are the actual values of v_derivative as a function of the angular distance bin_middles
    #Compute V integrating bin_counts
    cumulative_integral = np.cumsum(bin_counts * bin_widths)

    # Create subplots: one for the histogram and curves, and one for the cumulative integral
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Plot the histogram in the upper subplot
    ax1.hist(np.radians(mt_angle_values), bins=nbins, edgecolor='black', density=True)
    ax1.plot(bin_middles,bin_counts,marker='.',color='green')
    ax1.set_title('Histogram of Moment Tensor Angle')
    ax1.set_ylabel(r"$V'(\omega)$")
    ax1.set_xlabel(r"$\omega$")

    # Plot the theoretical curve or critical lines based on constraints
    if constrain == 'FMT' and curve:
        # Theoretical FMT curve
        angle_array = np.arange(0, 181, 1)
        angle_array_rad = np.radians(angle_array)
        v_prime = (8 / (3 * np.pi)) * (np.sin(angle_array_rad) ** 4)
        ax1.plot(angle_array_rad, v_prime, marker='.', color='red')

    if constrain == 'DC':
        if curve:
            # Example: `omegafun` should be defined elsewhere in your code
            t = np.linspace(0, np.pi, 1000)
            p = omegafun(t)
            ax1.plot(t, p, marker='.', color='red')

        # Plot critical vertical lines if critical_values are provided
        if critical_values is not None:
            for critical_value in critical_values:
                ax1.axvline(x=critical_value, color='red', linestyle='--', linewidth=2)

    ax1.grid(True)

    # Plot the cumulative integral in the lower subplot
    ax2.plot(bin_middles, cumulative_integral, marker='o', color='blue')
    ax2.set_title('Cumulative Integral of the Histogram')
    ax2.set_ylabel(r"$V(\omega)$")
    ax2.set_xlabel(r"$\omega$")
    ax2.grid(True)


    # Save the figure in PDF format
    output_filename= '{}/derivative_V_and_V.pdf'.format(path,type)
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    plt.close()

    # Show the plot
    #plt.show()

    return(bin_middles,bin_counts,cumulative_integral)

def omegafun(t):

    # Define constants
    t0 = 0
    t1 = np.pi / 3
    t2 = 2 * np.pi / 3
    t3 = np.pi

    # Define the ranges
    i1 = np.where((t >= t0) & (t <= t1))[0]
    i2 = np.where((t > t1) & (t <= t2))[0]
    i3 = np.where((t > t2) & (t <= t3))[0]

    # Define P1omega and P1pts
    P1omega = np.arange(0, 60.5, 0.5) * np.pi / 180
    P1pts = np.array([
        0., 0.000024241, 0.0000969677, 0.000218191, 0.00038793, 0.00060621,
        0.000873064, 0.00118853, 0.00155267, 0.00196552, 0.00242716, 0.00293765,
        0.00349707, 0.00410552, 0.00476308, 0.00546987, 0.00622598, 0.00703155,
        0.00788669, 0.00879156, 0.00974629, 0.010751, 0.011806, 0.0129113,
        0.0140671, 0.0152736, 0.0165312, 0.0178398, 0.0191998, 0.0206115,
        0.022075, 0.0235906, 0.0251587, 0.0267794, 0.0284531, 0.03018,
        0.0319606, 0.0337951, 0.035684, 0.0376275, 0.039626, 0.04168,
        0.0437899, 0.0459561, 0.0481791, 0.0504593, 0.0527972, 0.0551934,
        0.0576484, 0.0601627, 0.0627369, 0.0653717, 0.0680677, 0.0708256,
        0.073646, 0.0765297, 0.0794773, 0.0824898, 0.0855679, 0.0887126,
        0.0919247, 0.0952051, 0.0985549, 0.101975, 0.105467, 0.109031,
        0.112669, 0.116382, 0.120172, 0.124039, 0.127986, 0.132013,
        0.136123, 0.140316, 0.144596, 0.148964, 0.153421, 0.15797,
        0.162614, 0.167354, 0.172193, 0.177134, 0.18218, 0.187334,
        0.192599, 0.197979, 0.203476, 0.209097, 0.214844, 0.220723,
        0.226738, 0.232896, 0.239201, 0.245661, 0.252283, 0.259075,
        0.266044, 0.273201, 0.280556, 0.28812, 0.295907, 0.303932,
        0.31221, 0.320761, 0.329606, 0.33877, 0.348282, 0.358177,
        0.368495, 0.379286, 0.39061, 0.402544, 0.415183, 0.428658,
        0.443143, 0.458889, 0.476281, 0.495959, 0.519169, 0.549105,
        0.619061
    ])

    # Define the interpolation type
    itype = 'linear'

    # Initialize the result array
    p = np.zeros_like(t)

    # Interpolate values for the first segment
    interp_func1 = interp1d(P1omega, P1pts, kind=itype, fill_value="extrapolate")
    p[i1] = interp_func1(t[i1])

    # Interpolate values for the third segment
    interp_func3 = interp1d(-P1omega + np.pi, P1pts, kind=itype, fill_value="extrapolate")
    p[i3] = interp_func3(t[i3])

    # Define the polynomial for the middle segment
    P2poly = [-0.28276, 0, 0.696581]
    pshift2 = np.pi / 2

    # Compute the polynomial values for the second segment
    p[i2] = np.polyval(P2poly, t[i2] - pshift2)

    return (p)

def make_P_derivative_and_P(ds,angular_distance,v,path):

    angle_array = np.degrees(angular_distance)
    likelihood_sum = []

    for angle in angle_array:
        ds_filter = ds.where(ds['mt_angle'] <= angle)
        likelihood_sum.append(ds_filter['likelihoods'].sum().item())
        print('angle = {}, sum_likelihood = {}'.format(angle,likelihood_sum[-1]))

    derivative_likelihood = np.gradient(likelihood_sum,angle_array)

    # Create a single canvas with two subplots
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Plot the curve angle vs. P
    ax1.plot(angle_array, likelihood_sum, marker='.',color='blue')
    ax1.plot(angle_array, v, marker='.',color='red')
    ax1.set_ylabel(r"$P(\omega)$")
    ax1.set_xlabel(r"$\omega$")
    ax1.set_title('Angle vs. Cumulative likelihood')
    ax1.grid(True) 

    # Plot the derivative 
    ax2.plot(angle_array, derivative_likelihood, marker='.', color='r')
    ax2.set_ylabel(r"$P'(\omega)$")
    ax2.set_xlabel(r"$\omega$")
    ax2.set_title('Angle vs. Derivative Cumulative likelihood')
    ax2.grid(True)
    ax2.set_xticks(np.arange(0, 185, 5))

    # Show the plots
    plt.tight_layout()
    # Save the figure
    fig.savefig('{}/angle_vs_cumulative_likelihood_and_derivative.pdf'.format(path))
    plt.close(fig)
    #plt.show()

    return(derivative_likelihood,likelihood_sum)
    
def  make_confidence_curve(v,p,path):

    plt.plot(v,p, marker='.',color='red')
    # Filled with gray below the curve
    plt.fill_between(v,p,alpha=0.3,color='gray')
    # Add a dashed diagonal line from (0, 0) to (1, 1)
    plt.plot([0, 1], [0, 1],linestyle='--',color='black',)
    #Integrate for calculating the average value
    average = np.trapz(p,v)
    plt.xlabel('Fractional Volume (V)')
    plt.ylabel(r'$\mathcal{P}(V)$')

    # Display the average value in the upper-right corner of the plot
    plt.text(0.95, 0.1, f'average = {average:.2f}', 
             horizontalalignment='right', 
             verticalalignment='top', 
             transform=plt.gca().transAxes,
             fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.6))
    
    plt.title('Confidence Curve')
    plt.grid(True)
    plt.savefig('{}/confidence_curve.pdf'.format(path))
    plt.close()
    #plt.show()

if __name__=='__main__':

    direct_run = True

    if direct_run:
        #Read misfit file
        path = 'OUTPUT_20090407201255351DC'
 
        type = 'DC'

        netcdf_file = '{}/20090407201255351{}_likelihoods_angles.nc'.format(path,type)
        
        ds= read_likelihoods_angles(netcdf_file)
        print(ds)

        #Check the dataarray is consistent
        double_check_xarray(ds)

        if type == 'DC':

            #Plot fractional volume and its derivative
            nbins = 180
            constrain = 'DC'
            curve = True
            critical_values = np.radians(np.array([0,60,120,180]))
            angular_distance,v_derivative,v = make_v_derivative_and_v(ds, nbins, constrain,critical_values,path,curve=True)
    
            #Plot cumulative likelihood and its derivative
            p_derivative,p = make_P_derivative_and_P(ds,angular_distance,v,path)

        if type == 'FMT':

            #Plot fractional volume and its derivative
            nbins = 180
            constrain = 'DC'
            curve = False
            critical_values = None
            angular_distance,v_derivative,v = make_v_derivative_and_v(ds, nbins, constrain,critical_values,path,curve)

            #Plot cumulative likelihood and its derivative
            p_derivative,p =  make_P_derivative_and_P(ds,angular_distance,v,path)

        make_confidence_curve(v,p,path)
        print(path)

    
        


    

