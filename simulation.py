import simpy
import numpy as np
import itertools
from distributions import *
import pandas as pd
from joblib import Parallel, delayed


############### Simulation parameters #################
# Resources
N_BEDS = 24

# Patient inter-arrival time (IAT) distributions
MEAN_IAT_ae = 22.72
MEAN_IAT_ward = 26.0
MEAN_IAT_emer = 37.0
MEAN_IAT_oth = 47.2
MEAN_IAT_xray = 575.0

# Patient length of stay (LOS) distributions
MEAN_LOS_ae = 128.79
STD_LOS_ae = 267.51
MEAN_LOS_ward = 177.89 
STD_LOS_ward = 276.54
MEAN_LOS_emer = 140.15 
STD_LOS_emer = 218.02
MEAN_LOS_oth = 212.86
STD_LOS_oth = 457.67
MEAN_LOS_xray = 87.53
STD_LOS_xray = 108.15
MEAN_LOS_es = 57.34
STD_LOS_es = 99.78

# Elective surgery parameters
DAILY_SAMPLE_SIZE = int(1182 / 12 / 30)

# Changeover period
MEAN_changeover = 5
MIN_changeover = 2
MAX_changeover = 8

# trace function
TRACE = False

# random seeds
DEFAULT_RNG_SET = None
N_STREAMS = 12

# results collection & warm-up period
DEFAULT_RESULTS_COLLECTION_PERIOD = 12 * 30 * 24  # 12 months
WARM_UP = 30 * 24 # 1 month

# warmup auditing
DEFAULT_WARMUP_AUDIT_INTERVAL = 48 

# default number of replications
DEFAULT_N_REPS = 5

########################################################


# Helper function to print out messages    
def trace(msg):
    '''
    Turning printing of events on and off.
    
    Params:
    -------
    msg: str
        string to print to screen.
    '''
    if TRACE:
        print(msg)
        
        


class Scenario:
    '''
    Parameter class for CCU simulation model inputs.
    '''
    def __init__(self, random_number_set=None):
        '''
        The init method sets up our defaults. 
        '''
        # resource
        self.n_beds = N_BEDS
        
        # warm-up
        self.warm_up = 0.0
        
        
        # sampling
        self.random_number_set = random_number_set
        self.init_sampling()
        
        
    def set_random_no_set(self, random_number_set):
        '''
        Controls the random sampling 

        Parameters:
        ----------
        random_number_set: int
            Used to control the set of psuedo random numbers
            used by the distributions in the simulation.
        '''
        self.random_number_set = random_number_set
        self.init_sampling()
        
        
    def init_sampling(self):
        '''
        Create the distributions used by the model and initialise 
        the random seeds of each.
        '''
        # create random number streams
        rng_streams = np.random.default_rng(self.random_number_set)
        self.seeds = rng_streams.integers(0, 999999999, size=N_STREAMS)
        
        # Inter-arrival time (IAT) distributions for five types of patients
        self.ae_arrival_dist = Exponential(MEAN_IAT_ae, 
                                           random_seed=self.seeds[0])
        self.ward_arrival_dist = Exponential(MEAN_IAT_ward, 
                                             random_seed=self.seeds[1])
        self.emer_arrival_dist = Exponential(MEAN_IAT_emer, 
                                             random_seed=self.seeds[2])
        self.oth_arrival_dist = Exponential(MEAN_IAT_oth, 
                                            random_seed=self.seeds[3])
        self.xray_arrival_dist = Exponential(MEAN_IAT_xray, 
                                             random_seed=self.seeds[4])

        # Length of stay (LOS) distributions for six types of patients
        self.ae_los_dist = Lognormal(MEAN_LOS_ae, STD_LOS_ae, 
                                     random_seed=self.seeds[5])
        self.ward_los_dist = Lognormal(MEAN_LOS_ward, STD_LOS_ward, 
                                       random_seed=self.seeds[6])
        self.emer_los_dist = Lognormal(MEAN_LOS_emer, STD_LOS_emer, 
                                       random_seed=self.seeds[7])
        self.oth_los_dist = Lognormal(MEAN_LOS_oth, STD_LOS_oth, 
                                      random_seed=self.seeds[8])
        self.xray_los_dist = Lognormal(MEAN_LOS_xray, STD_LOS_xray, 
                                       random_seed=self.seeds[9])
        self.es_los_dist = Lognormal(MEAN_LOS_es, STD_LOS_es, 
                                     random_seed=self.seeds[10])
        
        # Changeover distributions
        self.changeover_dist = Triangular(MIN_changeover, 
                                          MEAN_changeover,
                                          MAX_changeover,
                                          random_seed=self.seeds[11])
        
        
        
class Patient:
    '''
    Patient in the CCU
    '''
    def __init__(self, identifier, env, source, args):
        '''
        Constructor method
        
        Params:
        -----
        identifier: int
            a numeric identifier for the patient.
            
        env: simpy.Environment
            the simulation environment
            
        args: Scenario
            The input data for the scenario
        '''
        # patient and environment
        self.identifier = identifier
        self.env = env        
        self.source = source        
        self.beds = args.beds
        

        # Length of stay (LOS) distributions for five types of patients
        self.ae_los_dist = args.ae_los_dist
        self.ward_los_dist = args.ward_los_dist
        self.emer_los_dist = args.emer_los_dist
        self.oth_los_dist = args.oth_los_dist
        self.xray_los_dist = args.xray_los_dist
        # changeover distribution
        self.changeover_dist = args.changeover_dist
        
        # individual parameter
        self.wait_time = 0.0
        self.los = 0.0


    def service(self):
        '''
        simulates the process for unplanned admissions in CCU 
        
        1. request and wait for a bed
        2. stay in CCU for a period of LOS
        3. exit system
        4. changeover period
        
        '''
        # record the time that patient entered the system
        arrival_time = self.env.now

        # request a bed 
        with self.beds.request() as req:
            yield req
            
            # waiting time
            self.wait_time = self.env.now - arrival_time
            
            # sample LOS
            self.los = self.sample_los()
            trace(f'Patient {self.identifier} from {self.source} waited for {self.wait_time:.2f} hours. '\
                  + f'LOS: {self.los:.2f}')
            
            yield self.env.timeout(self.los)            
            
            trace(f'Patient {self.identifier} from {self.source} left at {self.env.now:.2f}')
            
            # add changeover time
            changeover = self.changeover_dist.sample()
            yield self.env.timeout(changeover)
            trace(f'Bed released at {self.env.now:.2f}')
            
            
    def sample_los(self):
        '''
        Sample the LOS distribution 
        according to different type of sources.
        '''
        if self.source == 'A&E':
            self.los = self.ae_los_dist.sample()
        elif self.source == 'Ward':
            self.los = self.ward_los_dist.sample()
        elif self.source == 'Emergency':
            self.los = self.emer_los_dist.sample()
        elif self.source == 'Other Hospital':
            self.los = self.oth_los_dist.sample()
        elif self.source == 'X-ray':
            self.los = self.xray_los_dist.sample()
                        
        return self.los
    


class ElectivePatient(Patient):
    '''
    Elective surgery patient in the CCU
    '''
    # Track patients who cancel surgery
    cancelled_surgeries = []
          
    def __init__(self, identifier, env, source, args):
        '''
        Constructor method
        
        Params:
        -----
        identifier: int
            a numeric identifier for the patient.
            
        env: simpy.Environment
            the simulation environment
            
        args: Scenario
            The input data for the scenario
        '''
        super().__init__(identifier, env, source, args)
        self.es_los_dist = args.es_los_dist
        self.changeover_dist = args.changeover_dist
        self.warm_up = args.warm_up
        
        self.los = 0.0
    
    @classmethod
    def reset_cancellations(cls):
        cls.cancelled_surgeries = []
    
    def service(self):
        '''
        simulates the process for planned admissions in CCU 
        
        1. request a bed or cancel the surgery
        2. stay in CCU for a period of LOS
        3. exit system.
        
        '''
        # record the time that patient entered the system
        arrive_time = self.env.now

        # check if there is available bed
        if self.beds.count < self.beds.capacity:
            # request a bed
            with self.beds.request() as req:
                yield req
                # sample LOS
                self.los = self.es_los_dist.sample()
                trace(f'Patient {self.identifier} from {self.source}'\
                      + f' admitted with LOS: {self.los:.2f}')
                
                yield self.env.timeout(self.los)
                
                trace(f'Patient {self.identifier} from {self.source}'\
                      + f' left at {self.env.now:.2f}')
                
                # add changeover time
                changeover = self.changeover_dist.sample()
                yield self.env.timeout(changeover)
                trace(f'Bed released at {self.env.now:.2f}')
        else:
            # Add in the calcelled list
            # after the warm up
            if self.env.now >= self.warm_up:
                ElectivePatient.cancelled_surgeries.append(self.identifier)
            
            trace(f'Patient {self.identifier} from {self.source}'\
                  + f' cancelled at {arrive_time:.2f}')
            
            
class CCU:  
    '''
    Model of a CCU
    '''
    def __init__(self, args):
        '''
        Contructor
        
        Params:
        -------
        env: simpy.Environment
        
        args: Scenario
            container class for simulation model inputs.
        '''
        self.env = simpy.Environment()
        self.args = args
        self.init_resources()
        self.patients = []
        
        self.ae_arrival_dist = args.ae_arrival_dist
        self.ward_arrival_dist = args.ward_arrival_dist
        self.emer_arrival_dist = args.emer_arrival_dist
        self.oth_arrival_dist = args.oth_arrival_dist
        self.xray_arrival_dist = args.xray_arrival_dist
        
        # generate patient identifier
        self.identifier_generator = itertools.count()
                
        
        
    def init_resources(self):
        '''
        Init the number of resources
        and store in the arguments container object
        '''
        self.args.beds = simpy.Resource(self.env, 
                                        capacity=self.args.n_beds)
        
        
        
    def run(self, results_collection_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
            warm_up=0, daily_sample_size=DAILY_SAMPLE_SIZE):
        '''
        Conduct a single run of the model

        run length = results_collection_period + warm_up

        Parameters:
        ----------
        results_collection_period, float, optional
            default = DEFAULT_RESULTS_COLLECTION_PERIOD

        warm_up, float, optional (default=0)
            length of initial transient period to truncate
            from results.

        Returns:
        --------
            None

        '''
        
        # calculate the parameters
        RUN_LENGTH = results_collection_period+warm_up
        NUM_WEEKS = math.ceil(RUN_LENGTH / 24 / 7)
                
           
        # setup the arrival process        
        self.env.process(self.ae_arrivals_generator())
        self.env.process(self.ward_arrivals_generator())
        self.env.process(self.emer_arrivals_generator())
        self.env.process(self.oth_arrivals_generator())
        self.env.process(self.xray_arrivals_generator())
        self.env.process(self.es_arrivals_generator(NUM_WEEKS, 
                                                    daily_sample_size))
                
        # run the model
        self.env.run(until=RUN_LENGTH)
        
        
            
    def ae_arrivals_generator(self):
        '''
        IAT generator for ae patients
        '''
        while True:
            inter_arrival_time = self.ae_arrival_dist.sample()
            yield self.env.timeout(inter_arrival_time)
            
            patient_count = next(self.identifier_generator)
            trace(f'Patient {patient_count} from A&E'\
                  + f' arrived at {self.env.now:.2f}')

            # create a new patient and pass in env and args
            new_patient = Patient(patient_count, self.env, 'A&E', self.args)
            
            # keep a record of the patient for results calculation
            # after the warm up
            if self.env.now >= self.args.warm_up:
                self.patients.append(new_patient)
            
            # init the service process for this patient
            self.env.process(new_patient.service())
            
            
            
    def ward_arrivals_generator(self):
        '''
        IAT generator for ward patients
        '''
        while True:
            inter_arrival_time = self.ward_arrival_dist.sample()
            yield self.env.timeout(inter_arrival_time)

            patient_count = next(self.identifier_generator)
            trace(f'Patient {patient_count} from Ward'\
                  + f' arrived at {self.env.now:.2f}')

            # create a new patient and pass in env and args
            new_patient = Patient(patient_count, self.env, 'Ward', self.args)
            
            # keep a record of the patient for results calculation
            # after the warm up
            if self.env.now >= self.args.warm_up:
                self.patients.append(new_patient)
            
            # init the service process for this patient
            self.env.process(new_patient.service())
            
            
    def emer_arrivals_generator(self):
        '''
        IAT generator for emergency patients
        '''
        while True:
            inter_arrival_time = self.emer_arrival_dist.sample()
            yield self.env.timeout(inter_arrival_time)

            patient_count = next(self.identifier_generator)
            trace(f'Patient {patient_count} from Emergency'\
                  + f' arrived at {self.env.now:.2f}')

            # create a new patient and pass in env and args
            new_patient = Patient(patient_count, self.env, 'Emergency', self.args)
            
            # keep a record of the patient for results calculation
            # after the warm up
            if self.env.now >= self.args.warm_up:
                self.patients.append(new_patient)
            
            # init the service process for this patient
            self.env.process(new_patient.service())
            
            
    def oth_arrivals_generator(self):
        '''
        IAT generator for other hospital patients
        '''
        while True:
            inter_arrival_time = self.oth_arrival_dist.sample()
            yield self.env.timeout(inter_arrival_time)

            patient_count = next(self.identifier_generator)
            trace(f'Patient {patient_count} from Other Hospital'\
                  + f' arrived at {self.env.now:.2f}')

            # create a new patient and pass in env and args
            new_patient = Patient(patient_count, self.env, 'Other Hospital', self.args)
            
            # keep a record of the patient for results calculation
            # after the warm up
            if self.env.now >= self.args.warm_up:
                self.patients.append(new_patient)
            
            # init the service process for this patient
            self.env.process(new_patient.service())
            
            
    def xray_arrivals_generator(self):
        '''
        IAT generator for xray patients
        '''
        while True:
            inter_arrival_time = self.xray_arrival_dist.sample()
            yield self.env.timeout(inter_arrival_time)

            patient_count = next(self.identifier_generator)
            trace(f'Patient {patient_count} from X-ray'\
                  + f' arrived at {self.env.now:.2f}')

            # create a new patient and pass in env and args
            new_patient = Patient(patient_count, self.env, 'X-ray', self.args)
            
            # keep a record of the patient for results calculation
            # after the warm up
            if self.env.now >= self.args.warm_up:
                self.patients.append(new_patient)
            
            # init the service process for this patient
            self.env.process(new_patient.service())
            
            
    def es_arrivals_generator(self, num_weeks, daily_sample_size):
        '''
        Arrival times generator for elective surgery patients
        '''
        ElectivePatient.reset_cancellations()
        
        while True:
            for week in range(num_weeks):
                for day_of_week in range(7):  
                    # Calculate the absolute day 
                    current_day = week * 7 + day_of_week  

                    # Check for weekdays
                    if 0 <= day_of_week <= 4:  
                        # Sample the arrival times 
                        daily_arrival_times = sample_daily_arrival_times(daily_sample_size, 
                                                                         random_seed=current_day)

                        # calculate the scheduled arrival times
                        last_arrival_time = current_day * 24  
                        for arrival_time in daily_arrival_times:
                            actual_arrival_time = last_arrival_time + arrival_time
                            
                            # Prevent negative values
                            es_iat = max(0, actual_arrival_time - self.env.now)  
                            yield self.env.timeout(es_iat)

                            patient_count = next(self.identifier_generator)
                            trace(f'Patient {patient_count} from Elective Surgery'\
                                  + f' arrived at {self.env.now:.2f}')
                            
                            # create a new patient and pass in env and args
                            new_patient = ElectivePatient(patient_count, self.env, 'Elective Surgery', self.args)
                            
                            # keep a record of the patient for results calculation
                            # after the warm up
                            if self.env.now >= self.args.warm_up:
                                self.patients.append(new_patient)

                            # init the service process for this patient
                            self.env.process(new_patient.service())
                            
                    else:
                        # skip the weekends by fastforwarding 2 days
                        yield self.env.timeout(24 * 2)
                        break
                        
    
    def run_summary(self):
        '''
        Function for results collection
        '''
        # admissions from various sources
        ae_admissions = sum(patient.source == 'A&E' for patient in self.patients)
        ward_admissions = sum(patient.source == 'Ward' for patient in self.patients)
        emer_admissions = sum(patient.source == 'Emergency' for patient in self.patients)
        oth_admissions = sum(patient.source == 'Other Hospital' for patient in self.patients)
        xray_admissions = sum(patient.source == 'X-ray' for patient in self.patients)

        # Calculate the number of cancelled elective surgery patients
        cancelled_es = len(ElectivePatient.cancelled_surgeries)
        es_admissions = sum(patient.source == 'Elective Surgery' for patient in self.patients) - cancelled_es

        # total admissions
        total_admissions = len(self.patients) - cancelled_es       
        
        
        # waiting time = sum(waiting times) / no. patients
        mean_wait_time = np.array([patient.wait_time 
                                    for patient in self.patients]).mean()

        # adjust util calculations for warmup period
        rc_period = self.env.now - self.args.warm_up
        # bed days utilisation = sum(los) / (run length X no. beds)
        bed_day_util = np.array([patient.los 
                         for patient in self.patients]).sum() / \
                        (rc_period * self.args.n_beds)

        # append to results 
        df = pd.DataFrame({'1':{'Total_admissions': total_admissions,
                                'A&E_admissions': ae_admissions,
                                'Ward_admissions': ward_admissions,
                                'Emergency_admissions': emer_admissions,
                                'Other_hospital_admissions': oth_admissions,
                                'Xray_admissions': xray_admissions,
                                'Elective_Surgery_admissions': es_admissions,
                                'Cancelled_Surgeries': cancelled_es, 
                                'Mean_wait_hours': mean_wait_time, 
                                'Bed_days_util': bed_day_util}})


        return df
    
    
class Auditor:
    def __init__(self, model, args, 
                 run_length=DEFAULT_RESULTS_COLLECTION_PERIOD, 
                 first_obs=WARM_UP, 
                 interval=DEFAULT_WARMUP_AUDIT_INTERVAL):
        '''
        Auditor Constructor
        
        '''
        self.env = model.env
        self.model = model
        self.args = args
        self.run_length = run_length
        self.first_obs = first_obs
        self.interval = interval
        
        self.queues = []
        self.services = []
        
        # dict to hold states
        self.metrics = {}
        
        # scheduled the periodic audits
        if not first_obs is None:
            self.env.process(self.scheduled_observation())
            self.env.process(self.run_summary())
        
            
    def add_resource_to_audit(self, resource, name='bed', audit_type='qs'):
        if 'q' in audit_type:
            self.queues.append((name, resource))
            self.metrics[f'queue_length_{name}'] = []
        
        if 's' in audit_type:
            self.services.append((name, resource))
            self.metrics[f'occupied_{name}'] = []  
                    
    def record_queue_length(self):
        for name, res in self.queues:
            self.metrics[f'queue_length_{name}'].append(len(res.queue)) 
               
    def record_occupied_bed(self):
        for name, res in self.services:
            self.metrics[f'occupied_{name}'].append(res.count) 

            
    def scheduled_observation(self):
        '''
        simpy process to control the frequency of 
        auditor observations of the model.  
        
        The first observation takes place at self.first_obs
        and subsequent observations are spaced self.interval
        apart in time.
        '''
        # delay first observation until warm-up
        yield self.env.timeout(self.first_obs)
        self.record_queue_length()
        self.record_occupied_bed()
        
        while True:
            yield self.env.timeout(self.interval)
            self.record_queue_length()
            self.record_occupied_bed()
               
        
    def run_summary(self):
        '''
        Create an end of run summary
        
        Returns:
        ---------
            pd.DataFrame
        '''
        
        yield self.env.timeout(self.run_length - 1)
        
        run_results = {}

        for name, res in self.queues:
            queue_length = np.array(self.metrics[f'queue_length_{name}'])
            run_results[f'mean_queue_{name}'] = queue_length.mean()
            
        for name, res in self.services:
            serviced_beds = np.array(self.metrics[f'occupied_{name}'])
            run_results[f'mean_occupied_{name}'] = serviced_beds.mean()
            run_results[f'occupancy_rate'] = (serviced_beds.mean() / self.args.n_beds) 

        self.summary_frame = pd.Series(run_results).to_frame()
        self.summary_frame.columns = ['1'] 

        
        
def single_run(scenario, 
               rc_period=DEFAULT_RESULTS_COLLECTION_PERIOD, 
               warm_up=WARM_UP, 
               random_no_set=DEFAULT_RNG_SET, 
               daily_sample_size=DAILY_SAMPLE_SIZE):
    '''
    Perform a single run of the model and return the results
    
    Parameters:
    -----------
    
    scenario: Scenario object
        The scenario/paramaters to run
        
    rc_period: int
        The length of the simulation run that collects results
        
    warm_up: int, optional (default=0)
        warm-up period in the model.  The model will not collect any results
        before the warm-up period is reached.  
        
    random_no_set: int or None, optional (default=1)
        Controls the set of random seeds used by the stochastic parts of the 
        model.  Set to different ints to get different results.  Set to None
        for a random set of seeds.
        
    daily_sample_size: int
        
    Returns:
    --------
        pandas.DataFrame:
        results from single run.
    '''  
        
    # set random number set - this controls sampling for the run.
    scenario.set_random_no_set(random_no_set)

    # create an instance of the model
    model = CCU(scenario)
    
    # create an auditor
    auditor = Auditor(model, scenario)
    auditor.add_resource_to_audit(scenario.beds)
    
    model.run(results_collection_period=rc_period,
                warm_up=warm_up, daily_sample_size=DAILY_SAMPLE_SIZE)
    
    # return the results table
    results_model = model.run_summary()
    results_auditor = auditor.summary_frame
    
    results = pd.concat([results_model, results_auditor])    
    results = results.T
    
    return results.round(2)




def multiple_replications(scenario, 
                          rc_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
                          warm_up=WARM_UP,
                          n_reps=DEFAULT_N_REPS, 
                          daily_sample_size=DAILY_SAMPLE_SIZE,
                          n_jobs=-1):
    '''
    Perform multiple replications of the model.
    
    Params:
    ------
    scenario: Scenario
        Parameters/arguments to configurethe model
    
    rc_period: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.  
        the number of minutes to run the model beyond warm up
        to collect results
    
    warm_up: float, optional (default=0)
        initial transient period.  no results are collected in this period

    n_reps: int, optional (default=DEFAULT_N_REPS)
        Number of independent replications to run.

    n_jobs, int, optional (default=-1)
        No. replications to run in parallel.
        
    Returns:
    --------
    List
    '''    
    res = Parallel(n_jobs=n_jobs)(delayed(single_run)(scenario, 
                                                      rc_period, 
                                                      warm_up,
                                                      random_no_set=rep,
                                                      daily_sample_size=DAILY_SAMPLE_SIZE)
                                      for rep in range(n_reps))


    # format and return results in a dataframe
    df_results = pd.concat(res)
    df_results.index = np.arange(1, len(df_results)+1)
    df_results.index.name = 'rep'
    return df_results
