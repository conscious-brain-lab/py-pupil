import numpy as np
import scipy as sp
import seaborn as sn
import matplotlib.pylab as plt
from IPython import embed as shell 
import glob
import os 
import shutil
from hedfpy import *
import scipy.stats as stats 

from fir import FIRDeconvolution
import pandas as pd
from funcs import SDT

sn.set(style="ticks")

dataDir =  '/Users/stijnnuiten/Documents/UvA/Data/perception/'
task = 'loc'
subs = [25]
overwrite = True

analysis_params = {
                'sample_rate' : 1000.0,
                'lp' : 6.0,
                'hp' : 0.01,
                'normalization' : 'psc',
                'regress_blinks' : True,
                'regress_sacs' : True,
                'regress_xy' : False,
                'use_standard_blinksac_kernels' : False,
                }

class pupilSession(object):
	def __init__(self, subject, index, task, analysis_params ):
		self.subject = subject
		self.index = index
		self.task = task
		self.analysis_params = analysis_params
		self.alias = str(subject) + '_' + str(index) 
		self.pupilDir = os.path.join(dataDir,self.task, str(self.subject),'preproc/') 

	def preproc(self):
		files = glob.glob(os.path.join(dataDir, self.task,str(self.subject),'*.edf'))
		for f in files:
			print 'now preprocessing file ' + f
			alias = f.split('/')[-1][:4]
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
			shutil.copy(f, os.path.join(preprocDir, 'raw', '{}.edf'.format(alias)))
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

	def load_h5(self ):
		self.hdf5_filename = self.alias + '.h5'
		self.ho = HDFEyeOperator(self.pupilDir + self.hdf5_filename)
		self.trial_times = self.ho.read_session_data(self.alias, 'trials')
		self.trial_phase_times = self.ho.read_session_data(self.alias, 'trial_phases')        
		self.trial_parameters = self.ho.read_session_data(self.alias, 'parameters')
		self.blocks = np.unique(self.trial_parameters['block']).shape[0]
		self.block_duration = (self.trial_parameters['block']==0).sum()
		self.first_trials_in_block = [self.block_duration * b for b in range(self.blocks)]

	def load_pupil(self,data_type='pupil_bp',requested_eye = 'L' ):
		trial_indices = []
		fix_times = []
		blink_times = []
		stim_times = []
		resp_times = []
		feedback_times = []
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
			# shell()
			session_start_EL_time = np.array(self.trial_times['trial_start_EL_timestamp'])[ftib]
			if ftib == self.first_trials_in_block[-1]:
				session_stop_EL_time = np.array(self.trial_times['trial_end_EL_timestamp'])[-1]
			else:
				session_stop_EL_time = np.array(self.trial_times['trial_end_EL_timestamp'])[ftib+self.block_duration-1]

			total_time = np.array(((session_stop_EL_time - session_start_EL_time)/1000)/60) #total time in minutes

			trial_indices.append(self.trial_parameters['trial_nr'])
			trials_per_run.append(len(self.trial_parameters['trial_nr']))
			        
			self.sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], self.alias)

			fix_times.append((np.array(self.trial_phase_times[self.trial_phase_times['trial_phase_index']==1]['trial_phase_EL_timestamp'][ftib:ftib+self.block_duration] - session_start_EL_time + session_time) / self.sample_rate))
			stim_times.append((np.array(self.trial_phase_times[self.trial_phase_times['trial_phase_index'] == 2]['trial_phase_EL_timestamp'][ftib:ftib+self.block_duration] - session_start_EL_time + session_time) / self.sample_rate ))
			resp_times.append((np.array(self.trial_phase_times[self.trial_phase_times['trial_phase_index'] == 3]['trial_phase_EL_timestamp'][ftib:ftib+self.block_duration] - session_start_EL_time + session_time) / self.sample_rate ))
			feedback_times.append((np.array(self.trial_phase_times[self.trial_phase_times['trial_phase_index'] == 4]['trial_phase_EL_timestamp'][ftib:ftib+self.block_duration] - session_start_EL_time + session_time) / self.sample_rate ))
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

			###eye jitter data ### 
			xy_data = self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = self.alias, signal = 'gaze_x_int', requested_eye = 'L')
			vel_data = self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = self.alias, signal = 'vel', requested_eye = 'L')
			 # get xposition gaze data for eye jitter based estimation
			x = np.squeeze(xy_data['L_gaze_x_int'])
			# z-score eye movement data 
			x = (x-np.median(x)) / x.std()
			# calculate derivative of eye movement --> velocity 
			dx = np.r_[0, np.diff(x)]
			dx_data.append((dx - dx.mean()) / dx.std())
			#####detect microsaccades##### 
			session_time += session_stop_EL_time - session_start_EL_time

		nr_blinks.append(np.array([len(blink_times[x]) for x in range(len(blink_times))]) )
		blink_rate = nr_blinks/total_time  #blink rate per minute 

		self.end_time_trial =  np.concatenate(end_time_trial)
		self.fix_times = np.concatenate(fix_times)
		self.stim_times = np.concatenate(stim_times)
		self.resp_times = np.concatenate(resp_times)
		self.feedback_times = np.concatenate(feedback_times)


		self.blink_times = np.concatenate(blink_times)
		self.blink_times_b = np.copy(blink_times) 
		self.nr_blinks= np.hstack(nr_blinks)
		self.blink_rate= np.concatenate(blink_rate)    
		#self.microsaccade_times = np.concatenate(microsaccade_times)
		self.dx_data = np.concatenate(dx_data)            
		self.pupil = np.hstack(pupil)
		self.pupil_data = np.hstack(pupil_data)
		self.pupil_baseline = np.hstack(pupil_baseline)
		self.trial_indices = np.hstack(trial_indices)
		self.pupil_raw = np.hstack(pupil_raw)


		self.run_sample_limits = np.array([np.cumsum(samples_per_run)[:-1],np.cumsum(samples_per_run)[1:]]).T
		self.run_trial_limits = np.array([np.cumsum(trials_per_run)[:-1],np.cumsum(trials_per_run)[1:]]).T
		# shell()	
			# folder_name = 'pupil_data'
			# with pd.HDFStore(self.ho.input_object) as h5_file:
			# 	# h5_file.put("/%s/%s/%s"%(self.alias,folder_name, 'pupil_data'), pd.Series(np.array(self.pupil_data)))
			# 	h5_file.put("/%s/%s/%s"%(self.alias,folder_name, 'sample_rate'), pd.Series(self.sample_rate))
		
	def deconvolve_colour_sound(self, analysis_sample_rate = 25, interval = [-0.5,5.5],  data_type = 'pupil_bp_psc', requested_eye = 'L', microsaccades_added=False):
			"""raw deconvolution, to see what happens to pupil size when the fixation colour changes, 
			and when the sound chimes."""

			# self.logger.info('starting basic pupil deconvolution with data of type %s and sample_rate of %i Hz in the interval %s, microsaccades_added = %s' % (data_type, analysis_sample_rate, str(interval), str(microsaccades_added)))

			if not hasattr(self, 'pupil_data'): # we assume that we'll grab the same events and data whatever the present deconvolve_reward_probabilities method does
				self.events_and_signals_in_time(data_type = data_type, requested_eye= requested_eye, microsaccades_added=microsaccades_added)
				self.events_and_signals_in_time_behav(data_type = data_type, requested_eye= requested_eye, microsaccades_added=microsaccades_added)

			# input_signal = sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate))

			#add microsaccade events 
			if microsaccades_added: 
			    events = [self.blink_times + interval[0], self.colour_times + interval[0], self.flop_times + interval[0], selfturn_times + interval[0], self.keypresstimes + interval[0],self.microsaccade_times + interval[0]] 
			else:
			    # events = [self.blink_times + interval[0], self.colour_times + interval[0], self.flop_times + interval[0], self.turn_times + interval[0], self.keypresstimes + interval[0]]
			    events = [ self.blink_times + interval[0], self.stim_times + interval[0], self.resp_times + interval[0], self.feedback_times + interval[0]]

			#add eye jitter events
			dx_signal = np.array(sp.signal.decimate(self.dx_data, int(self.sample_rate / analysis_sample_rate)), dtype = np.float32)    
			nr_sample_times = np.arange(interval[0], interval[1], 1.0/analysis_sample_rate).shape[0]
			added_jitter_regressors = np.zeros((nr_sample_times, dx_signal.shape[0]))
			for i in range(nr_sample_times):
			    added_jitter_regressors[i,(i+1):] = dx_signal[:-(i+1)]
	
			downsample_rate = 10
    
			analysis_sample_rate = self.sample_rate//downsample_rate
			fd = FIRDeconvolution(
			            signal =sp.signal.decimate(self.pupil_data, downsample_rate, 1), 
			            events = events, 
			            event_names = ['blinks','stim', 'resp','feedback'], 
			            sample_frequency = analysis_sample_rate,
			            deconvolution_frequency = analysis_sample_rate,
			            deconvolution_interval = interval
			            )

			fd.create_design_matrix()
			fd.regress()
			fd.betas_for_events()
			blink_response = np.array(fd.betas_per_event_type[0]).ravel()
			stim_response = np.array(fd.betas_per_event_type[1]).ravel()
			resp_response = np.array(fd.betas_per_event_type[2]).ravel()
			feedback_response = np.array(fd.betas_per_event_type[3]).ravel()

			# baseline the kernels:
			blink_response = blink_response - blink_response[0].mean()
			stim_response = stim_response - blink_response[0].mean()

			# plot:
			x=np.linspace(interval[0],interval[1],len(blink_response))
			f = plt.figure(figsize = (10,3.5))

			plt.plot(x, blink_response, label='blink response')
			plt.plot(x, stim_response, label='stim response')
			plt.xlabel('Time from event (s)')
			plt.ylabel('Pupil size')
			plt.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
			plt.legend(loc=2)
			sn.despine(offset=10)

	def calcTPR(self,data_type = 'pupil_bp_psc', requested_eye = 'L'):
		# Calculate task-evoked pupillary response (TPR) per trial, defined as mean aplitude pupil (-1s to 1.5s from choice)
		# minus baseline (-0.5 to 0s from stim)

		# shell()
		parameters = self.ho.read_session_data(self.alias, 'parameters')
		parameters_joined = parameters
		target_joined = parameters_joined['signal_present']#[(-parameters_joined['omissions'])]
		correct_joined = parameters_joined['correct']
		
		hit_joined = (parameters_joined['signal_present']==1) * (parameters_joined['correct']==1)
		fa_joined = (parameters_joined['signal_present']==0) * (parameters_joined['correct']==0)

		# hit_joined = parameters_joined['hit']#[(-parameters_joined['omissions'])]
		# fa_joined = parameters_joined['fa']#[(-parameters_joined['omissions'])]

		(dprime, c) = SDT(target_joined, hit_joined, fa_joined)

		# target = [ param['present'][(-param['omissions'])] for param in parameters ]
		# hit = [ param['hit'][(-param['omissions'])] for param in parameters ]
		# fa = [ param['fa'][(-param['omissions'])] for param in parameters ]        

		pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)

		session_start = self.trial_times['trial_start_EL_timestamp'][0]
		pupil_lp = np.array(pupil_data[(requested_eye + '_pupil_lp_clean_psc')])
		pupil_bp = np.array(pupil_data[(requested_eye + '_pupil_bp_clean_psc')])
		time = np.array(pupil_data['time']) - session_start

		bp_lp = np.array([np.mean(pupil_lp[(time>i-500)*(time<i)]) for i in self.stim_times]) 
		bp_bp = np.array([np.mean(pupil_bp[(time>i-500)*(time<i)]) for i in self.stim_times]) 
		tpr_lp =( np.array([np.mean(pupil_lp[(time>i-1000)*(time<i+1500)]) for i in self.resp_times]) - bp_lp[-1] )
		tpr_bp = np.array([np.mean(pupil_bp[(time>i-1000)*(time<i+1500)]) for i in self.resp_times]) - bp_bp[-1] 

		parameters_joined['pupil_d'] = tpr_bp
		parameters_joined['pupil_b'] = bp_bp
		parameters_joined['pupil_d'] = tpr_bp
		parameters_joined['pupil_b_lp'] = bp_lp
		parameters_joined['pupil_d_lp'] = tpr_lp
		parameters_joined['d_prime'] = dprime
		parameters_joined['criterion'] = c
		parameters_joined['subject'] = self.subject
		self.ho.data_frame_to_hdf('', 'parameters_joined', parameters_joined)

	def splitTPR(self):
		parameters = self.ho.read_session_data('parameters_joined','')
		pupil_d = np.array(parameters['pupil_d'])
		# plt.plot(parameters['pupil_d'])

		p_h = pupil_d <= np.percentile(pupil_d, 40) 
		p_l = pupil_d >= np.percentile(pupil_d, 60) 
		for s in np.array(np.unique(parameters['session']), dtype=int):
		    pupil_b = np.array(parameters['pupil_b'])[np.array(d.session) == s]
		    pupil_d = np.array(d['pupil_d'])[np.array(d.session) == s]
		    pupil = pupil_d
		    p_l.append( pupil <= np.percentile(pupil, 40) )
		    p_h.append( pupil >= np.percentile(pupil, 60) )


def main():
	pS = pupilSession(subject=25, index = 0, task = task,analysis_params=analysis_params)
	pS.preproc()
	pS.load_h5()
	pS.load_pupil()
	pS.deconvolve_colour_sound()
	pS.calcTPR()
	pS.splitTPR()

if __name__ == '__main__':
    main()