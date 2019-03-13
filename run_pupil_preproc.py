import numpy as np
import scipy as sp
import seaborn as sn
import matplotlib.pylab as pl
from IPython import embed as shell 
import glob
import os 
import shutil
from hedfpy import *

sn.set(style="ticks")

dataDir =  '/Users/stijnnuiten/Documents/UvA/Data/perception/'
task = 'loc'
subs = [25]
overwrite = True

analysis_params = {
                'sample_rate' : 1000.0,
                'lp' : 6.0,
                'hp' : 0.01,
                'normalization' : 'zscore',
                'regress_blinks' : True,
                'regress_sacs' : True,
                'regress_xy' : False,
                'use_standard_blinksac_kernels' : False,
                }

def preproc(subs,analysis_params):
	for s in subs:
		files = glob.glob(os.path.join(dataDir, task,str(s),'*.edf'))
		for f in files:
			print 'now preprocessing file ' + f
			alias = f.split('/')[-1][:4]
			preprocDir = os.path.join(dataDir, task,str(s), 'preproc/')

			try:
			    os.makedirs(os.path.join(preprocDir, 'raw/'))
			except OSError:
			    pass				
			# initiate HDFEyeoperator
			h5file = preprocDir + alias + '.h5'
			if os.path.isfile(h5file) and overwrite:
				os.remove(h5file)
			ho = HDFEyeOperator(os.path.expanduser(h5file))

			# extract data from EDF and run preprocessing (blink detection, filtering)
			shutil.copy(f, os.path.join(preprocDir, 'raw', '{}.edf'.format(alias)))
			ho.add_edf_file(os.path.join(preprocDir, 'raw', '{}.edf'.format(alias)))
			ho.edf_message_data_to_hdf(alias = alias)
			ho.edf_gaze_data_to_hdf(alias=alias,
	                                    sample_rate=analysis_params['sample_rate'],
	                                    pupil_lp=analysis_params['lp'],
	                                    pupil_hp=analysis_params['hp'],
	                                    normalization=analysis_params['normalization'],
	                                    regress_blinks=analysis_params['regress_blinks'],
	                                    regress_sacs=analysis_params['regress_sacs'],
	                                    use_standard_blinksac_kernels=analysis_params['use_standard_blinksac_kernels'],)

		# # downsample for plotting
		# downsample_rate = 10

		# # load times per session:
		# trial_times = ho.read_session_data(alias, 'trials')
		# trial_phase_times = ho.read_session_data(alias, 'trial_phases')

		# # check at what timestamps the recording started:
		# session_start_EL_time = np.array( trial_phase_times[np.array(trial_phase_times['trial_phase_index'] == 1) * np.array(trial_phase_times['trial_phase_trial'] == 0)]['trial_phase_EL_timestamp'] )[0]
		# session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]

		# # and, find some aspects of the recording such as sample rate and recorded eye
		# sample_rate = ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], alias)
		# eye = ho.eye_during_period([session_start_EL_time, session_stop_EL_time], alias)
		# if len(eye) > 0:
		#     eye = ['L','R'][0]

def main():
    preproc(subs=subs, analysis_params=analysis_params)

if __name__ == '__main__':
    main()