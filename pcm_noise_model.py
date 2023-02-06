import torch as t

class pcm_noise_model:

    def __init__(self, t0=60,rd_noise_mu=-0.010373, rd_noise_std=2.20041720,neu_std_rel=0.09,neu_mean=0.0313729,G0_mu=26.783358,G0_noise_std_rel=0.2193646,G0_noise_mu=0.9202640,G0_mu_spatial=22.6969838):
        self.rd_noise_mu = rd_noise_mu
        self.rd_noise_std = rd_noise_std
        self.neu_std_rel = neu_std_rel
        self.neu_mean = neu_mean
        self.G0_mu =  G0_mu
        self.G0_noise_std_rel = G0_noise_std_rel
        self.G0_noise_mu = G0_noise_mu
        self.G0_mu_spatial = G0_mu_spatial
        self.t0 = t0

    def noise_conductance(self,time, size):

        start_cond = self.prog_cond_noisy(size)
        drift_noise = self.drift_noisy_exp(size)
        a = start_cond.unsqueeze(0).repeat(time.shape[0], 1, 1)
        b = drift_noise.unsqueeze(0).repeat(time.shape[0], 1, 1)
        tt = time / self.t0
        tt = time.unsqueeze(1).repeat(1,size[0])
        tt = tt.unsqueeze(2).repeat(1,1,size[1])
        time_drift = t.pow(tt, -b)
        cond_drift = a * time_drift
        rd_noise = t.normal(self.rd_noise_mu, self.rd_noise_std, time.shape + size)
        noisy_cond = cond_drift + rd_noise
        return noisy_cond,cond_drift,start_cond


        #write_batch_tiled * (drift_time ** (-nu))

    def drift_noisy_exp(self,size):
        neu_mean = 0.026759882788222403
        #new nu based on mean spatial drift values
        neu_mean = 0.03137293518172905
        neu_std_rel = 1.2667551094129317
        #reduce neu_std_rel to fusion values
        neu_std_rel = 0.09
        drift_variability  = t.normal(0,self.neu_std_rel,size)
        nue = self.neu_mean * (1 + drift_variability)
        return nue

    def prog_cond_noisy(self,size):
        G0_noise_mu = 0.9202640713859175
        G0_noise_std = 0.3408815267494223
        G0_noise_std_rel = 0.3704170763029516
        G0_mu = 29.226653006273448
        #new spatial values
        G0_mu_spatial = 22.69698387871316
        G0_noise_std = 4.978915148863486
        G0_noise_std_rel = 0.21936461582162314
        #new mean spatial drift G0
        G0_mu =  26.78335824681155
        G0_noise_std_rel = self.G0_noise_std_rel * self.G0_mu_spatial/self.G0_mu
        start_condutance = t.ones(size)*self.G0_mu
        prog_noise = self.G0_noise_mu+t.normal(0,G0_noise_std_rel,size)
        start_condutance = start_condutance * (prog_noise)
        return start_condutance


#start block comment
