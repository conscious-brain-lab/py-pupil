import numpy as np
import scipy as sp
import seaborn as sn
import matplotlib.pylab as plt
from IPython import embed as shell 
import glob
import os, sys
import shutil
from hedfpy import *
import scipy.stats as stats 

sys.path.append(os.environ['ANALYSIS_HOME'])


from fir import FIRDeconvolution
import pandas as pd
from functions.funcs import SDT, label_diff

sn.set(style="ticks")

dataDir =  '/Users/stijnnuiten/Documents/UvA/Data/perception/'
task = 'loc'
subs = [25]
overwrite = True

analysis_params = {
                'sample_rate' : 500,
                'lp' : 6.0,
                'hp' : 0.01,
                'normalization' : 'psc',
                'regress_blinks' : True,
                'regress_sacs' : True,
                'regress_xy' : False,
                'use_standard_blinksac_kernels' : True,
                }

class pupilSession(object):
	def __init__(self, subject, index, task, analysis_params ):
		self.subject = subject
		self.index = index
		self.task = task
		self.analysis_params = analysis_params
		self.alias = str(subject) + '_' + str(index) 
		self.pupilDir = os.path.join(dataDir,self.task, str(self.subject),'preproc/') 
		self.load_h5()

	def load_h5(self ):
		self.hdf5_filename = self.alias + '.h5'
		self.ho = HDFEyeOperator(self.pupilDir + self.hdf5_filename)
		self.trial_times = self.ho.read_session_data(self.alias, 'trials')
		self.trial_phase_times = self.ho.read_session_data(self.alias, 'trial_phases')        
		self.trial_parameters = self.ho.read_session_data(self.alias, 'parameters')
		self.blocks = np.unique(self.trial_parameters['block']).shape[0]
		self.block_duration = (self.trial_parameters['block']==0).sum()
		self.first_trials_in_block = [self.block_duration * b for b in range(self.blocks)]
		session_start_EL_time = np.array(self.trial_times['trial_start_EL_timestamp'])[0]
		session_stop_EL_time = np.array(self.trial_times['trial_end_EL_timestamp'])[-1]
		self.sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], self.alias)

	def preproc(self):
		files = glob.glob(os.path.join(dataDir, self.task,str(self.subject),'*.edf'))
		filename = glob.glob(os.path.join(dataDir, self.task, str(self.subject),str(self.subject) + '_' + str(self.index) +'*.edf'))[0]
		print 'now preprocessing file ' + filename
	
		alias = filename.split('/')[-1][:4]
		preprocDir = os.path.join(dataDir, self.task,str(self.subject), 'preproc/')

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
		shutil.copy(filename, os.path.join(preprocDir, 'raw', '{}.edf'.format(alias)))
		ho.add_edf_file(os.path.join(preprocDir, 'raw', '{}.edf'.format(alias)))
		ho.edf_message_data_to_hdf(alias = alias)
		ho.edf_gaze_data_to_hdf(alias=alias,
                                    sample_rate=self.analysis_params['sample_rate'],
                                    pupil_lp=self.analysis_params['lp'],
                                    pupil_hp=self.analysis_params['hp'],
                                    normalization=self.analysis_params['normalization'],
                                    regress_blinks=self.analysis_params['regress_blinks'],
                                    regress_sacs=self.analysis_params['regress_sacs'],
                                    use_standard_blinksac_kernels=self.analysis_params['use_standard_blinksac_kernels'],)

	def load_pupil(self, events, omissions, data_type='pupil_bp',requested_eye = 'L'):
		events_times = {}
		for ev in events:
			events_times[ev + '_times'] = []
		trial_indices = []
		blink_times = []
		end_time_trial = []
		nr_blinks =[]
		pupil_data = []
		pupil_raw = []
		pupil_baseline = []
		dx_data = []
		samples_per_run = [0]
		trials_per_run = [0]
		session_time = 0

		for ftib in self.first_trials_in_block:
			session_start_EL_time = np.array(self.trial_times['trial_start_EL_timestamp'])[ftib]
			if ftib == self.first_trials_in_block[-1]:
				session_stop_EL_time = np.array(self.trial_times['trial_end_EL_timestamp'])[-1]
			else:
				session_stop_EL_time = np.array(self.trial_times['trial_end_EL_timestamp'])[ftib+self.block_duration-1]

			total_time = np.array(((session_stop_EL_time - session_start_EL_time)/1000)/60) #total time in minutes

			trial_indices.append(self.trial_parameters['trial_nr'])
			trials_per_run.append(len(self.trial_parameters['trial_nr']))
			        
			self.sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], self.alias)

			for ev in events:
				events_times[ev + '_times'].append((np.array(self.trial_phase_times[self.trial_phase_times['trial_phase_index']==events[ev]]['trial_phase_EL_timestamp'][ftib:ftib+self.block_duration] - session_start_EL_time + session_time) / self.sample_rate))

			end_time_trial.append((np.array(self.trial_times['trial_end_EL_timestamp'][ftib:ftib+self.block_duration]) - session_start_EL_time + session_time) / self.sample_rate)

			eyelink_blink_data = self.ho.read_session_data(self.alias, 'blinks_from_message_file')
			eyelink_blink_data_L = eyelink_blink_data[eyelink_blink_data['eye'] == requested_eye] #only select data from left eye
			b_start_times = np.array(eyelink_blink_data_L.start_timestamp)
			b_end_times = np.array(eyelink_blink_data_L.end_timestamp)

			#evaluate only blinks that occur after start and before end experiment
			b_indices = (b_start_times>session_start_EL_time)*(b_end_times<session_stop_EL_time) 
			b_start_times_t = (b_start_times[b_indices] - session_start_EL_time) #valid blinks (start times) 
			b_end_times_t = (b_end_times[b_indices] - session_start_EL_time) 
			blinks = np.array(b_start_times_t)            
			blink_times.append(((blinks + session_time) / self.sample_rate ))

			pupil = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = self.alias, signal = data_type, requested_eye = requested_eye))
			pupil_data.append((pupil - np.median(pupil))/ pupil.std())
			pupil_raw.append(np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = self.alias, signal = data_type, requested_eye = requested_eye)))
			samples_per_run.append(len(pupil))
			pupil_baseline.append(np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = self.alias, signal = 'pupil_lp', requested_eye = requested_eye))) # 

			#eye jitter data #
			xy_data = self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = self.alias, signal = 'gaze_x_int', requested_eye = 'L')
			vel_data = self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = self.alias, signal = 'vel', requested_eye = 'L')

			# get xposition gaze data for eye jitter based estimation
			x = np.squeeze(xy_data['L_gaze_x_int'])
			# z-score eye movement data 
			x = (x-np.median(x)) / x.std()
			# calculate derivative of eye movement --> velocity 
			dx = np.r_[0, np.diff(x)]
			dx_data.append((dx - dx.mean()) / dx.std())

			session_time += session_stop_EL_time - session_start_EL_time

		nr_blinks.append(np.array([len(blink_times[x]) for x in range(len(blink_times))]) )
		blink_rate = nr_blinks/total_time  #blink rate per minute 
		shell()
		for ev in events:
			setattr(self,ev + '_times', np.concatenate(events_times[ev + '_times']))
			print np.concatenate(events_times[ev + '_times']).shape
			# get
		self.oms = np.array([self.trial_parameters[omissions.keys()[i]] == omissions[omissions.keys()[i]] for i in range(len(omissions))])
		self.oms = np.any(self.oms,0)
		self.end_time_trial =  np.concatenate(end_time_trial)[~self.oms]
		self.blink_times = np.concatenate(blink_times)
		self.blink_times_b = np.copy(blink_times)
		self.nr_blinks= np.hstack(nr_blinks)
		self.blink_rate= np.concatenate(blink_rate)    
		self.dx_data = np.concatenate(dx_data)            
		self.pupil = np.hstack(pupil)
		self.pupil_data = np.hstack(pupil_data)
		self.pupil_baseline = np.hstack(pupil_baseline)
		self.trial_indices = np.hstack(trial_indices)[~self.oms ]
		self.pupil_raw = np.hstack(pupil_raw)

		# Also remove omitted trials from parameter DataFrame
		parameters = self.ho.read_session_data(self.alias, 'parameters')
		try:
			parameters = parameters.iloc[self.trial_indices]
		except:
			print '\n\n\n\n\n\n No trials omitted in parameters DF \n\n\n\n\n\n'
			pass

		self.run_sample_limits = np.array([np.cumsum(samples_per_run)[:-1],np.cumsum(samples_per_run)[1:]]).T
		self.run_trial_limits = np.array([np.cumsum(trials_per_run)[:-1],np.cumsum(trials_per_run)[1:]]).T

		# Store everything in H5-file
		self.ho.data_frame_to_hdf(self.alias, 'parameters', parameters)
		folder_name = 'pupil_data'
		with pd.HDFStore(self.ho.input_object) as h5_file:
			h5_file.put("/%s/%s/%s"%(self.alias,folder_name, 'pupil_data'), pd.Series(np.array(self.pupil_data)))
			# 	h5_file.put("/%s/%s/%s"%(self.alias,folder_name, 'sample_rate'), pd.Series(self.sample_rate))

	def calcTPR(self, baseline_ev, target_ev, data_type = 'pupil_bp_clean_psc', requested_eye = 'L'):
		# Calculate task-evoked pupillary response (TPR) per trial, defined as mean aplitude pupil (-1s to 1.5s from choice)
		# minus baseline (-0.5 to 0s from stim)
		for ev in [baseline_ev, target_ev]:
			if not hasattr(self,ev + '_times'):
				print ev + ' is missing'			
		shell()
		parameters = self.ho.read_session_data(self.alias, 'parameters')

		session_start = np.array(self.trial_times['trial_start_EL_timestamp'])[0]
		session_end = np.array(self.trial_times['trial_start_EL_timestamp'])[-1]
		pupil_data = self.ho.data_from_time_period((session_start, session_end), self.alias)

		pupil = np.array(pupil_data[(requested_eye + '_' + data_type)])
		time = (np.array(pupil_data['time']) - session_start)/self.sample_rate

		bl = np.array([np.mean(pupil[(time>i-0.5)*(time<i)]) for i in getattr(self, baseline_ev +'_times')]) 
		tpr = np.array([np.mean(pupil[(time>i-1)*(time<i+1.5)]) for i in getattr(self, target_ev +'_times')]) 
		# Check if num baseline events matches num target events. If not, check how the two are aligned.
		# If e.g. first baseline event is missing, remove first trial in its entirety.
		while bl.shape[0] != tpr.shape[0]:
			print 'shapes dont match, removing trial'
			if getattr(self, baseline_ev +'_times')[0]> getattr(self, target_ev +'_times')[0]:
				tpr=tpr[1:]
				parameters = parameters.drop([0])
			else:
				tpr=tpr[:-1]
				parameters = parameters.drop(parameters.index[-1])
		# Subtract the baseline from the evoked pupil response to get the actual TPR
		tpr_bl = tpr - bl

		# add to parameters
		parameters['tpr_bl_' + data_type] = tpr_bl
		parameters['tpr_' + data_type ] = tpr
		parameters['bl_' + data_type] = bl

		# and store in H5-file
		self.ho.data_frame_to_hdf(self.alias, 'parameters', parameters)

	def splitTPR(self,data_type = 'pupil_bp_clean_psc'):
		parameters = self.ho.read_session_data(self.alias,'parameters')
		pupil_d = np.array(parameters['tpr_bl_' + data_type])
		parameters = parameters[~np.isnan(pupil_d)]
		pupil_d = pupil_d[~np.isnan(pupil_d)]

		p_h = pupil_d <= np.percentile(pupil_d, 40) 
		p_l = pupil_d >= np.percentile(pupil_d, 60)

		pups = {'p_h':p_h,'p_l':p_l}
		for p in pups.keys():
			target = parameters['signal_present'][pups[p]]
			correct = parameters['correct'][pups[p]]

			hit = (target==1) * (correct==1)
			fa = (target==0) * (correct==0)

			(dprime, c) = SDT(target, hit, fa)
			print "d-prime for %s + is: %.2f" %(p,dprime) 
			print "c for %s + is: %.2f" %(p,c) 

		parameters['p_l'] = pd.Series(p_l)
		parameters['p_h'] = pd.Series(p_h)

def main():
	# Initiate pupil object

	pS = pupilSession(subject=subs[s], index = ids[id], task = task,analysis_params=analysis_params)
	pS.preproc()
	pS.load_pupil(events={'fix':1, 'stim':2, 'resp':3}, omissions={'restarted':1})
	pS.calcTPR(baseline_ev = 'fix', target_ev='resp')
	pS.splitTPR()

if __name__ == '__main__':
    main()