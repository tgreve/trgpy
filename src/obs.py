import numpy as np
import pylab as plt
import importlib


import trgpy.dictionary_transitions
importlib.reload(trgpy.dictionary_transitions)
from trgpy.dictionary_constants import cnsts
from trgpy.dictionary_transitions import freq
from trgpy.dictionary_observatories import band

#from pylab import *
#from trg.plt import *
#from collections import OrderedDict
#from astroquery.splatalogue import Splatalogue
#from astropy import units as u
#from astropy.table import Table



##
##
#def atmospheric_transmission(pwv=2.00):
#    """
#       Return atmospheric transmission (frequency vs transmission percentage)
#       for given input pwv (default pwv=1.00).
#
#       Uses models from http://almascience.eso.org/about-alma/weather/atmosphere-model
#       of the transmission for the ALMA site (Alt = 5040m) for pwv=0.25, 0.50, 1.0, 2.0,
#       3.0, 5.0. The frequency ranges is 10GHz to 1010GHz.
#    """
#
#    #Read in atmospheric model
#    f=open('/Users/tgreve/Desktop/Dark/local/python/trg/atm-models/plot.data','r')
#    lines=f.readlines()[5:]
#    f.close()
#
#    atm_frq=[]
#    atm_T1=[]
#    atm_T2=[]
#    atm_T3=[]
#    atm_T4=[]
#    atm_T5=[]
#    atm_T6=[]
#    for line in lines:
#        p=line.strip()
#        p=p.split()
#        if p != []:
#            atm_frq.append(float(p[0]))
#            atm_T1.append(float(p[1]))
#            atm_T2.append(float(p[2]))
#            atm_T3.append(float(p[3]))
#            atm_T4.append(float(p[4]))
#            atm_T5.append(float(p[5]))
#            atm_T6.append(float(p[6]))
#
#    # Select model for input pwv
#    if pwv==0.22:
#        atm_trans = atm_T1
#    elif pwv==0.5:
#        atm_trans = atm_T2
#    elif pwv==1.00:
#        atm_trans = atm_T3
#    elif pwv==2.00:
#        atm_trans = atm_T4
#    elif pwv==3.00:
#        atm_trans = atm_T5
#    elif pwv==5.00:
#        atm_trans = atm_T6
#    else:
#       print("No atmospheric model for chosen pwv... return -1")
#       return -1
#
#    return atm_frq,atm_trans


#
#
def transition_visibility_band(z,transition,*args,**kwargs):
    """
       For given redshift, transition(s), and facility(ies) output which
       receiver(s) the transition falls in.

       Also generates a plot showing the receivers bands and the
       observed frequency of the transitions along with a plot
       of the atmospheric transmission at Chajnantor. It is possible
       to set the frequency range of the plot (frq_1 and frq_2).

       Ex.:
           transition_visibility_band(0.2,'12CO(3-2)','ALMA','SMA',plot=True)
    """


    # Get keywords: frequency window and plot
    frq_1 = 1.0      # [GHz]
    frq_2 = 1000.0   # [GHz]
    plot=False
    for key in kwargs:
        if key == 'frq_1':
            frq_1 = kwargs['frq_1']
        if key == 'frq_2':
            frq_2 = kwargs['frq_2']
        if key == 'plot':
            plot = kwargs['plot']

    # Get transitions IDs and frequencies
    line_id = freq.keys()
    line_frq = freq.values()
    if transition != 'all':
        line_indices=[i for i,x in enumerate(line_id) if transition in x]
        line_frq=[line_frq[i] for i in line_indices]
        line_id=[line_id[i] for i in line_indices]
    else:
        line_indices=[i for i,x in enumerate(line_frq) if x >= frq_1 and x <= frq_2]
        line_frq=[line_frq[i] for i in line_indices]
        line_id=[line_id[i] for i in line_indices]

    if plot:
        # Get atmospheric transmission curve
        atm_frq, atm_trans = atmospheric_transmission(pwv=2.00)

        # Set up plot, plot atmospheric transmission
        setup_graph(title=r'Transmission curves, ALMA, Llano de Chajnantor, alt. 5040m',
                    x_label=r'frequency [GHz]',y_label=r'Transmission [%]', fig_size=(27,8))
        plt.xlim(frq_1,frq_2)
        plt.ylim(-0.1,1.1)
        plt.plot(atm_frq, atm_trans,'b-',label='pwv=1.00')
        plt.legend(loc="upper right",fontsize=22)

    # Loop through facilities and select receivers
    for facility in args:
        print(' ')
        print(' ')
        if 'ALMA' in facility:
            rx=band['ALMA']
            print('                                                                  ALMA')
        if 'SMA' in facility:
            rx=band['SMA']
            print('                                                                  SMA')
        if 'IRAM NOEMA' in facility:
            rx=band['IRAM NOEMA']
            print('                                                                  IRAM NOEMA')
        if 'IRAM-30m' in facility:
            rx=band['IRAM-30m']
            print('                                                                  IRAM-30m')
        if 'JVLA' in facility:
            rx=band['JVLA']
            print('                                                                  JVLA')

        # Prepare output string
        str_out='{0:22} {1:16}'.format('Line','Rest frequency')
        for i in np.arange(0,len(rx),1):
            str_out=str_out + ' {0:15}'.format(rx[i][0])
        print(str_out)
        print('{0:22} {1:16}'.format('','[GHz]'))

        j=0
        for line in line_id:
            output_string='{0:8.2f}'.format(line_frq[j])
            output_string='{0:20} {1:9}'.format(line_id[j],output_string)
            frq_obs = line_frq[j]/(1+z)
            for i in np.arange(0,len(rx),1):
                if frq_obs <= rx[i][2] and frq_obs >= rx[i][1]:
                    output_string = output_string+'{0:16.2f}'.format(line_frq[j]/(1+z))
                else:
                    output_string = output_string+'{0:8}'.format('          --    ')
            print(output_string)
            j=j+1

        if plot:
            # Plot lines and line IDs
            for i in range(0,len(line_id)):
                plt.plot([line_frq[i]/(1.+z),line_frq[i]/(1.+z)],[-100,100],'b-.')
                plt.text(line_frq[i]/(1.+z)+5,1.0,line_id[i],rotation=90,fontsize=20)

            # Plot receiver bands
            for i in np.arange(0,len(rx),1):
                plt.plot([rx[i][1],rx[i][2]],[0.4-i*0.05,0.4-i*0.05],'k-')
                if frq_1 <= rx[i][1] and frq_2 >= rx[i][1]:
                    plt.text(rx[i][1],0.41-i*0.05,rx[i][0],fontsize=20)

    if plot:
        plt.show()


#
#
def frequency_vs_redshift(transitions=['12CO(1-0)'], observatory='JCMT',
                          plot=False):
    """
       Started:
       Updated: 05.02.17
    """

    # Output observatory header to terminal and define rx-list.
    if observatory == 'JVLA':
        print('{0:>96}'.format('JVLA'))
        rx=band['JVLA']
    elif observatory == 'SMA':
        print('{0:>52}'.format('SMA'))
        rx=band['SMA']
    elif observatory == 'ALMA':
        rx=band['ALMA']
    elif observatory == 'IRAM NOEMA':
        print('{0:>59}'.format('IRAM NOEMA'))
        rx=band['IRAM NOEMA']
    elif observatory == 'IRAM 30m':
        print('{0:>81}'.format('IRAM 30m (EMIR)'))
        rx=band['IRAM-30m']
    elif observatory == 'JCMT':
        print('{0:>41}'.format('JCMT'))
        rx=band['JCMT']
    elif observatory == 'FIRI':
        print('{0:>33}'.format('FIRI (25-400um)'))
        rx=band['FIRI']

    # Returned array containing z-range for given transition and rx combination
    rx_z_range = [[[0 for i in range(2)] for j in range(len(rx))] for k in range(len(transitions))]
    foo = np.array(rx_z_range)

    # Prepare output string.
    str_out='{0:22} {1:16}'.format('Line','Rest frequency')
    for i in np.arange(0,len(rx),1):
        str_out=str_out + ' {0:15}'.format(rx[i][0])
    print(str_out)
    print('{0:22} {1:16}'.format('','[GHz]'))

    # Calculate redshift ranges for trans+rx combinations and output to
    # terminal. Also determine the min/max frequency of rx.
    frq_min = 1.E6
    frq_max = 0.
    i = 0
    foo = []
    for trans in transitions:
        output_string='{0:8.2f}'.format(freq[trans])
        output_string='{0:20} {1:19}'.format(trans,output_string)
        for j in range(0,len(rx),1):
            z_range = np.array([freq[trans],freq[trans]])/rx[j][1:] - 1

            if z_range[0] > 0 and z_range[0] <= 25:
                if z_range[1] < 0: z_range[1]=0
                dz_string = '{0:4.2f} {1:1} {2:4.2f}'.format(z_range[1],
                                                             '-',z_range[0])
                output_string = output_string+'{0:16}'.format(dz_string)
                rx_z_range[i][j][:] = [trans, rx[j][0], [z_range[0],z_range[1]]]
            else:
                output_string = output_string+'{0:16}'.format('--')

            # Determine min/max frequency range used for plot
            if frq_min > rx[j][1:][0]:
                frq_min = rx[j][1:][0]
            if frq_max < rx[j][1:][1]:
                frq_max = rx[j][1:][1]
        print(output_string)
        i = i + 1

    # Plot
    if plot:
        # Setup plot.
        pl.setup_graph(x_label=r'$z$', y_label=r'$\nu_{\rm obs}\,{\rm [GHz]}$',
                       fig_size=(8,8))
        plt.xlim(0., 10)
        ylim1=frq_min*0.5
        ylim2=frq_max*1.1
        plt.ylim(ylim1, ylim2)

        # Plot the observing bands.
        for i in range(0,len(rx),1):
            rectangle = plt.Rectangle((0, rx[i][1]), 10, rx[i][2]-rx[i][1],
                                      facecolor='green', edgecolor='black')
            plt.gca().add_patch(rectangle)
            plt.annotate(rx[i][0], xy=(0.1,rx[i][1]), fontsize=14)

        # Plot line frequency vs. redshift.
        z=np.linspace(0.001,10,100000)
        for var in transitions:
            if var in freq:
                frq_observed=freq[var]/(1.+z)
                plt.plot(z, frq_observed, 'k-', linewidth=2.0)
                y_label = freq[var]
                x_label=max(0.001, freq[var] / y_label - 1.)
                plt.annotate(var, xy=(x_label, y_label), fontsize=14)
        plt.show()


##
##
#def convert_dvelocity_to_dfrequency(frequency,dinterval,conversion):
#    """
#       Take input velocity interval (in km/s) and convert to a frequency
#       interval (in MHz) at a given frequency (in GHz). Or vice versa
#    """
#
#    c = cnsts['c']/1000.  #speed of light [km/s]
#    if conversion == 'kms2MHz':
#        dinterval = (dinterval/c)*(frequency*1000.)
#    if conversion == 'MHz2kms':
#        dinterval = (dinterval/(frequency*1000.))*c
#
#    return dinterval
#
#
#
##
##
#def jvla_sensitivity(rest_requency,IntegrationTime,VelocityBin,*Receiver,**Season):
#
#    for var in Season:
#        if Season[var] == 'Winter':
#    	    #1hr ON-source time/10km/s bin size/Winter/C-config/Natural Weighting/Dual Polarization/8bit Sampling
#    	    for var in Receiver:
#    	        if var == 'K':
#    	            Rxfrequency=np.array([18.0    ,18.5    ,19.0    ,19.5    ,20.0    ,20.5    ,21.0    ,21.5    ,22.0    ,22.5    ,23.0    ,23.5    ,24.0    ,24.5    ,25.0    ,25.5    ,26.0    ,26.5    ])  #[GHz]
#    	            rmsNoise   =np.array([580.1076,436.6316,329.7046,321.1845,316.4144,328.2933,341.9055,362.6436,383.0082,379.5607,371.3909,352.9678,334.5153,324.1508,314.9779,331.6622,349.2577,375.0435])  #[uJy]
#    	        if var == 'Ka':
#    	            Rxfrequency=np.array([26.0    ,26.5    ,27.0    ,27.5    ,28.0    ,28.5    ,29.0    ,29.5    ,30.0    ,30.5    ,31.0    ,31.5    ,32.0    ,32.5    ,33.0    ,33.5    ,34.0    ,34.5    ,35.0    ,35.5    ,36.0    ,36.5    ,37.0    ,37.5    ,38.0    ,38.5    ,39.0    ,39.5    ,40.0    ])  #[GHz]
#    	            rmsNoise   =np.array([440.8766,435.9359,431.4248,385.7326,341.0352,350.2211,359.7372,358.4757,357.7010,358.3324,359.1467,365.6811,372.4142,376.1260,379.9595,389.2686,398.6888,406.4558,414.5035,430.7185,446.9232,467.7548,489.9590,517.4141,545.5985,619.6471,695.1821,692.8362,691.2075])  #[uJy]
#    	        if var == 'Q':
#    	            Rxfrequency =np.array([40.0    ,41.0    ,42.0    ,43.0    ,44.0    ,45.0    ,46.0    ,47.0    ,48.0  ,49.0  ,50.0 ])  #[GHz]
#    	            rmsNoise   =np.array([448.8815,447.8544,477.1429,532.8013,580.5356,618.5305,728.5230,894.4382,1100.5,1480.7,2747.3])  #[uJy]
#
#    #Scaling to desired VelocityBin size
#    rmsNoise = rmsNoise*(10./VelocityBin)**0.5
#
#    #Scaling to desired ON-source time
#    rmsNoise = rmsNoise*(1./IntegrationTime)**0.5
#
#    #if Observedfrequency == -99, calculare for entire band
#    if Observedfrequency == -99:
#        Observedfrequency = np.arange(min(RxFrequency),max(RxFrequency),0.01)
#
#    #Interpolate rmsNoise to requested Observedfrequency
#    rmsNoise=np.interp(Observedfrequency,RxFrequency,rmsNoise)
#
#    return rmsNoise,Observedfrequency
