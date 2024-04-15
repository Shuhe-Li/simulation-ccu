import simpy
import numpy as np
import itertools
from scipy.stats import lognorm, truncnorm
import math


# Custom classes for distributions
class Exponential:
    def __init__(self, mean, random_seed=None):
        self.mean = mean
        self.rng = np.random.default_rng(seed=random_seed)

    def sample(self):
        return self.rng.exponential(self.mean)

class Lognormal:
    def __init__(self, mean, std, random_seed=None):
        self.mean = mean
        self.std = std
        self.rng = np.random.default_rng(seed=random_seed)
        self.sigma = np.sqrt(np.log(1 + (std/mean)**2))
        self.scale = np.exp(np.log(mean) - 0.5*self.sigma**2)

    def sample(self):
        return self.rng.lognormal(np.log(self.scale), self.sigma)
    

# Sample function for elective surgery
def sample_daily_arrival_times(mean, std, lower_bound, upper_bound, sample_size, random_seed):
    """
    Sample daily arrival times from a truncated normal distribution.
    """
    np.random.seed(random_seed)
    a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
    daily_arrival_times = truncnorm.rvs(a, b, loc=mean, scale=std, size=sample_size)
    daily_arrival_times.sort()
    return daily_arrival_times

    
# Patient class
class Patient:
    def __init__(self, env, patient_id, source, los_dist):
        self.env = env
        self.patient_id = patient_id
        self.source = source
        self.los_dist = los_dist

    def service(self, ccu):
        arrive_time = self.env.now
        print(f"Patient {self.patient_id} from {self.source} arrived at {arrive_time:.2f}")

        with ccu.request() as request:
            yield request
            wait_time = self.env.now - arrive_time
            los = self.los_dist.sample()
            print(f"Patient {self.patient_id} from {self.source} waited for {wait_time:.2f} hours, LOS: {los:.2f}")
            yield self.env.timeout(los)
            print(f"Patient {self.patient_id} from {self.source} left at {self.env.now:.2f}")
            
            
# ElectivePatient class            
class ElectivePatient(Patient):
    def __init__(self, env, patient_id, source, los_dist):
        super().__init__(env, patient_id, source, los_dist)
        self.cancelled_surgeries = []  # Track patients who cancel surgery
    
    def service(self, ccu):
        arrive_time = self.env.now
        print(f"Patient {self.patient_id} from {self.source} arrived at {arrive_time:.2f}")

        if ccu.count < ccu.capacity:
            with ccu.request() as request:
                yield request
                los = self.los_dist.sample()
                print(f"Patient {self.patient_id} from {self.source} admitted with LOS: {los:.2f}")
                yield self.env.timeout(los)
                print(f"Patient {self.patient_id} from {self.source} left at {self.env.now:.2f}")
        else:
            self.cancelled_surgeries.append(self.patient_id)
            print(f"Patient {self.patient_id} from {self.source} cancelled at {arrive_time:.2f}")
            

# CCU class
class CCU:
    def __init__(self, env, capacity):
        self.env = env
        self.resource = simpy.Resource(env, capacity=capacity)

    def ae_arrivals_generator(self, iat_dist, los_dist):
        while True:
            yield self.env.timeout(iat_dist.sample())
            patient_id = next(patient_id_generator)
            patient = Patient(self.env, patient_id, "ae", los_dist)
            self.env.process(patient.service(self.resource))

    def ward_arrivals_generator(self, iat_dist, los_dist):
        while True:
            yield self.env.timeout(iat_dist.sample())
            patient_id = next(patient_id_generator)
            patient = Patient(self.env, patient_id, "ward", los_dist)
            self.env.process(patient.service(self.resource))

    def emer_arrivals_generator(self, iat_dist, los_dist):
        while True:
            yield self.env.timeout(iat_dist.sample())
            patient_id = next(patient_id_generator)
            patient = Patient(self.env, patient_id, "emer", los_dist)
            self.env.process(patient.service(self.resource))

    def oth_arrivals_generator(self, iat_dist, los_dist):
        while True:
            yield self.env.timeout(iat_dist.sample())
            patient_id = next(patient_id_generator)
            patient = Patient(self.env, patient_id, "oth", los_dist)
            self.env.process(patient.service(self.resource))

    def xray_arrivals_generator(self, iat_dist, los_dist):
        while True:
            yield self.env.timeout(iat_dist.sample())
            patient_id = next(patient_id_generator)
            patient = Patient(self.env, patient_id, "xray", los_dist)
            self.env.process(patient.service(self.resource))
            
    def es_arrivals_generator(self, num_weeks, mean, std, lower_bound, upper_bound, daily_sample_size, los_dist):
        while True:
            for week in range(num_weeks):
                for day_of_week in range(7):  # Loop through each day of the week
                    current_day = week * 7 + day_of_week  # Calculate the absolute day 

                    if 0 <= day_of_week <= 4:  # Check for weekdays
                        # Adjust random seed based on week and day for variety
                        random_seed = week * 7 + day_of_week
                        daily_arrival_times = sample_daily_arrival_times(mean, std, lower_bound, upper_bound, 
                                                                         daily_sample_size, random_seed)

                        last_arrival_time = current_day * 24  # Convert current day to hours
                        for arrival_time in daily_arrival_times:
                            actual_arrival_time = last_arrival_time + arrival_time
                            yield self.env.timeout(actual_arrival_time - env.now)
                            patient_id = next(patient_id_generator)
                            patient = ElectivePatient(self.env, patient_id, "es", los_dist)
                            self.env.process(patient.service(self.resource))
                    else:
                        continue

    
    
# Simulation parameters
RUN_LENGTH = 12 * 30 * 24  # 12 months
N_BEDS = 24
patient_id_generator = itertools.count()

# Elective surgery parameters
daily_sample_size = int(1182 / 12 / 30 / 3)  
mean = 17.91
std = 3.16
lower_bound = 0
upper_bound = 24
num_weeks = math.ceil(RUN_LENGTH / 24 / 7)


# Initialize simulation environment
env = simpy.Environment()
ccu = CCU(env, N_BEDS)

# Patient arrival source distributions
IAT_DISTRIBUTIONS = {
    "ae": Exponential(22.72),
    "ward": Exponential(26.0),
    "emer": Exponential(37.0),
    "oth": Exponential(47.2),
    "xray": Exponential(575.0)
}


# Patient LOS distributions
LOS_DISTRIBUTIONS = {
    "ae": Lognormal(128.79, 267.51),
    "ward": Lognormal(177.89, 276.54),
    "emer": Lognormal(140.15, 218.02),
    "oth": Lognormal(212.86, 457.67),
    "xray": Lognormal(87.53, 108.15),
    "es": Lognormal(57.34, 99.78)
}


# Starting the process
env.process(ccu.ae_arrivals_generator(IAT_DISTRIBUTIONS["ae"], LOS_DISTRIBUTIONS["ae"]))
env.process(ccu.ward_arrivals_generator(IAT_DISTRIBUTIONS["ward"], LOS_DISTRIBUTIONS["ward"]))
env.process(ccu.emer_arrivals_generator(IAT_DISTRIBUTIONS["emer"], LOS_DISTRIBUTIONS["emer"]))
env.process(ccu.oth_arrivals_generator(IAT_DISTRIBUTIONS["oth"], LOS_DISTRIBUTIONS["oth"]))
env.process(ccu.xray_arrivals_generator(IAT_DISTRIBUTIONS["xray"], LOS_DISTRIBUTIONS["xray"]))
env.process(ccu.es_arrivals_generator(num_weeks, mean, std, lower_bound, upper_bound, 
                                      daily_sample_size, LOS_DISTRIBUTIONS["es"]))

# Run the simulation
env.run(until=RUN_LENGTH)

