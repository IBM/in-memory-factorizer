#!/usr/bin/env python3

''' Bipolar encoder and decoder for multiplicative binding
    This is a simplified version which allows *batched* encoding/decoding
'''


import sys, os
import numpy as np
from scipy.linalg import pinv
import scipy.io as sio
import torch as t
import torch.nn.functional as F
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
import pickle



import pdb
from pcm_noise_model import *

__author__ = "Michael Hersche"
__email__ = "her@zurich.ibm.com"

class decodingProgress:
    def __init__(self, u, s, x_hat, itr,  max_iter, state_converged, convergence_idx, conv_idx, sim):
        self.u = u
        self.s = s
        self.x_hat = x_hat
        self.itr = itr
        self.max_iter = max_iter
        self.state_converged = state_converged
        self.convergence_idx = convergence_idx
        self.conv_idx = conv_idx
        self.sim = sim



class densebipolarbatched:
    def __init__(self, D, F, Mx,
                 noise=0,
                 separation="identity",
                 activation="identity",
                 decodingSequential=True,
                 topa=0,
                 activationThreshold=0,
                 convergenceDetectionThreshold=0.625,
                 noisyAttn=False,
                 noisyAttnType="gaussian",
                 time_for_hermes_iter=256,
                 time0_hermes = 60,
                 noisyAttnStd=0,
                 rd_noise_mu=-0.00253,
                 rd_noise_std=2.044650,
                 neu_std_rel=0.6643037,
                 neu_mean=0.042881,
                 G0_mu=25.87082,
                 G0_noise_std_rel=0.23273327,
                 G0_noise_mu=0.89666,
                 G0_mu_spatial=22.696983,
                 total_noise = None,
                 G0_noise_source_same = True,
                 pullUp=False,
                 useCuda=True,
                 gpu=None,
                 seed=None,
                 IM=None,
                 IMclassification=False,
                 simcomp = "op",
                 permutation = True,
                 accuracy = 1,
                 im_sampling = "Bernoulli",
                 savedir = "~/",
                 id = "densebipolarbatched",
                 **kwargs
                 ):
        """
        Factor code using bipolar dictionary
        Parameters
        ----------
        D: int 
            Output dimension 
        V: int 
            number of code books/factors 
        M: int
            Item memory size, code book size for every  
        device: string {"cpu", "cuda"}
            Cuda device
        IM: torch tensor (V,M,D)
            predefined IM, if None new IM will be initialized
        simcomp: string {"op","ols"}
            Estimation matrix: op - standard IM 
                               ols - ordinary least squared
        in_pcm: enable in-memory pcm forward/(backward) propagation
        im_sampling : string {"Bernoulli", "Randperm"}
            code vector sampling scheme
            Randperm (default)
        sequential_dec: bool
            Update estimation sequentially. If false, update all estimations in parallel.
        activation: string
            Activation before weighted superposition
            "relu", "softmax", "identity" (default), "ist" iterative soft thresholding, 
            "ist_mean" iterative soft thresholding based on mean
        Return
        ------
        """


        self._id = id
        self._D = D
        self._M = Mx
        self._F = F
        self._sequential_dec = decodingSequential
        self._pullUp = pullUp
        self._pullUp_thresh = convergenceDetectionThreshold
        self._permutation = permutation
        self._noisyAttn = noisyAttn
        self._noisyAttnType = noisyAttnType
        self._time_for_hermes_iter = time_for_hermes_iter
        self._G0_noise_source_same = G0_noise_source_same
        self._time0_hermes = time0_hermes
        if total_noise is None:
            self.G0_noise_std_rel = G0_noise_std_rel
            self.rd_noise_std = rd_noise_std
            self.neu_std_rel = neu_std_rel
        else:
            self.G0_noise_std_rel = total_noise[0]
            self.rd_noise_std = total_noise[1]
            self.neu_std_rel = total_noise[2]
        self._noisyAttnStd = noisyAttnStd
        self._accuracy = accuracy
        self._savedir = savedir
        self._im_sampling = im_sampling,
        self._IMclassification = IMclassification,
        self._activationName = activation
        if self._noisyAttnType == "pcm":
            if self._G0_noise_source_same:
                self.pcm_noise_model = pcm_noise_model(t0=time0_hermes, rd_noise_mu=rd_noise_mu, rd_noise_std=self.rd_noise_std,neu_std_rel=self.neu_std_rel,
                                                       neu_mean=neu_mean,G0_mu=G0_mu,G0_noise_std_rel=self.G0_noise_std_rel,
                                                       G0_noise_mu=G0_noise_mu,G0_mu_spatial=G0_mu_spatial)
            else:
                self.pcm_noise_model = pcm_noise_model(t0=time0_hermes,rd_noise_mu=rd_noise_mu, rd_noise_std=self.rd_noise_std,
                                                       neu_std_rel=self.neu_std_rel,
                                                       neu_mean=neu_mean, G0_mu=G0_mu,
                                                       G0_noise_std_rel=self.G0_noise_std_rel,
                                                       G0_noise_mu=G0_noise_mu, G0_mu_spatial=G0_mu_spatial)
                self.pcm_noise_model_bwd = pcm_noise_model(t0=time0_hermes,rd_noise_mu=rd_noise_mu, rd_noise_std=self.rd_noise_std,
                                                       neu_std_rel=self.neu_std_rel,
                                                       neu_mean=neu_mean, G0_mu=G0_mu,
                                                       G0_noise_std_rel=self.G0_noise_std_rel,
                                                       G0_noise_mu=G0_noise_mu, G0_mu_spatial=G0_mu_spatial)
        if not seed is None:
            print("setting seed to {}".format(seed))
            t.manual_seed(seed)

        # Cuda 
        if useCuda and t.cuda.is_available():
            if gpu is None:
                self._device = f'cuda:{t.cuda.current_device()}'
            else:
                self._device = f'cuda:{gpu}'
        else:
            self._device='cpu'

        print("Use device {:}".format(self._device))


        if IM is not None:
            self._IM = IM
        else:
            self._IM = t.empty(self._F, self._M, self._D).to(self._device)
            if self._im_sampling == "Bernoulli":
                self._IM = t.empty(self._F, self._M, self._D).uniform_().to(self._device)
                self._IM[self._IM > 0.5] = 1
                self._IM[self._IM <= 0.5] = -1
            # initialize item memory (assume that all factors use same sized _IM)
            else:
                for f in range(self._F):
                    for m in range(self._M):
                        self._IM[f, m] = (t.argsort(t.rand(self._D)) < int(self._D/2))*2 - 1

            if self._permutation:
                # premute factor 0 to retrieve factors 1..V
                for f in range(self._F):
                    self._IM[f] = t.roll(self._IM[0].detach(),f,1)

        # generate inital guess
        if self._M %2 ==0: # even number
            tie_breaker = t.zeros(self._F, self._D).to(self._device)
            for f in range(self._F):
                tie_breaker[f] = (((t.argsort(t.argsort(t.sum(self._IM[f],0))))>=int(D/2))*2-1)
        ## This is a deterministic tie breaker for even number of code book elements
        #if self._M %2 ==0: # even number
        #    tie_breaker = self._IM[:,0,:]*self._IM[:,1,:]
        else:
            tie_breaker = t.zeros(self._F, self._D).to(self._device)

        self._init_guess = (t.sign(t.sum(self._IM,1)+tie_breaker)).to(self._device) # similarities
        
        #########################################################################
        # Activation
        self._threshold = 0
        if activation=="topa" and topa!=1:
            self._activation = self._topa_sparse
            if topa < 1:
                self._topa = int(self._M * self.topa)
            else:
                self._topa = int(topa)
        elif activation=="topaP" and topa!=1:
            self._activation = self._topa_sparse_positiv
            if topa < 1:
                self._topa = int(self._M * self.topa)
            else:
                self._topa = int(topa)
        elif activation=="topaPT":
            self._activation = self._topa_sparse_threshold_positiv
            # calculate threshold based on bayesian learning piecewise linear approximated 
            if activationThreshold:
                self._threshold = activationThreshold
            else:
                std = np.sqrt(1/(4*self._D))*2
                precentages = 1-topa/self._M

                self._threshold = norm.ppf(precentages,0,std)
                self._threshold = np.round(self._threshold, 4)
            
                if self._threshold < 0 or np.isnan(self._threshold):
                    self._threshold = 0
            print("threshold={}".format(self._threshold))
        elif activation=="relu":
            self._activation = t.nn.ReLU()
        elif activation == "softmax":
            self._activation = t.nn.Softmax()
        elif activation=="identity":
            self._activation = t.nn.Identity()
        elif activation=="ist":
            # iterative soft thresholding
            self._activation = self._ist
        elif activation=="ist_mean":
            # iterative soft thresholding
            self._activation = self._ist_mean
        else:
            raise ValueError(f"Nonvalid activation, got {activation}")

        #########################################################################
        # Noise 
        # calculate noise based on bayesian learning piecewise linear approximated 
        self._noise = 0
        if noisyAttn:
            if noisyAttnStd:
                self._noise = noisyAttnStd
            else:
                dimension = [256, 512, 1024, 2048, 3072, 4096, 5120, 6144]
                noise = [0.0136, 0.01074, 0.008, 0.0058, 0.0045, 0.0038, 0.0033, 0.0029]
                self._noise = np.interp(self._D, dimension, noise)
                self._noise = round(self._noise, 4)

            print("noise={}".format(self._noise))

        # Readout
        if simcomp =="op":
            self._readout = self._IM
        elif simcomp =="ols":
            print("Use OLS decoding")
            self._readout= t.empty(self._F, self._M, self._D)

            for f in range(self._F):
                self._readout[f] = t.from_numpy(pinv(self._IM[f].cpu().numpy()).T)

            self._readout=self._readout.to(self._device)
        else:
            raise ValueError(f"No valid decode. got {simcomp}")

        # Normally, reconstruction matrix and IM are the same
        self._reconstruction = self._IM.clone()

    def encode(self,u):
        """
        Multiplicative encoding
        Parameters
        ----------
        u: torch int tensor (B,_V) with values {0,_M-1}
            input index array with batchsize B       
        Return
        ------
        x: torch FloatTensor (_D,)
            encoded HD vector 
        """
        assert t.max(u) < self._M

        B = u.shape[0]
        x = t.ones(B, self._D).to(self._device)

        # encode and multiplicate HD vector 
        # CAN BE ACCELERATED!!!!
        for b in range(B):
            for f in range(self._F):
                x[b] = x[b]*self._IM[f,u[b,f]]
        return x


    def decode(self, s, max_iter=2000, maxsim_pred=False, recoverDecodingProgress=False, u=None):
        """
        Decode HD vector to data with factorizer circuit
        Parameters
        ----------
        s: torch FloatTensor (B,_D)
            real valued input tensor
        max_iter: int 
            max number of decoding iterations  
        return_state: bool
        Return
        ------
        u_hat: torch int tensor (B,_V) with values {0,_M-1}
            input index array      
        sim: torch FloatTensor (B,_V,_M)
            similarity (attention) tensor
        x_hat: torch FloatTensor (B,_V,_D)
            estimated states  
        """

        B = s.shape[0]
        if recoverDecodingProgress:
            with open(os.path.join(self._savedir, 'resonator_{}_DecodingProgress.pickle'.format(self._id)), 'rb') as handle:
                self.dp = pickle.load(handle)
            print("Reloaded at iteration {}/{}... (converged: {})".format(self.dp.itr, self.dp.max_iter, int(t.sum(self.dp.state_converged)/self._F)))
        else:
            x_hat = self._init_guess.clone().view(1,self._F,self._D).repeat(B,1,1)
            
            state_converged = t.Tensor(B,self._F).zero_().type(t.bool).to(self._device)
            convergence_idx = t.Tensor(B).zero_().type(t.bool).to(self._device)
            conv_idx = t.Tensor(B,1).fill_(max_iter-1).to(self._device)
            sim = t.zeros(1, B, self._F, self._M).to(self._device) # similarities


            self.dp = decodingProgress(u=u, s=s, x_hat=x_hat, itr=0, max_iter=max_iter, state_converged=state_converged, convergence_idx=convergence_idx, conv_idx=conv_idx, sim=sim)

        # allocate memory
        noise = t.Tensor(B, self._M).zero_().to(self._device)
        
        #DEBUG
        mu, std, act, cnt = 0,0,0,0
        time_space = t.linspace(self._time0_hermes, max_iter * self._time_for_hermes_iter, max_iter)
        while self.dp.itr < self.dp.max_iter:
            # increase iteration step
            self.dp.itr += 1
            
            previous_x_hat = self.dp.x_hat.clone().detach()

            for f in range(self._F):
                s_f = self.dp.s.clone()

                # un-factor the input vector with current estimates 
                for f_dec in range(self._F):
                    if f != f_dec:
                        if self._sequential_dec:
                            # here we use the most recent estimation (sequential)
                            s_f = t.mul(s_f, self.dp.x_hat[:,f_dec])
                        else:
                            # decoding in parallel
                            s_f = t.mul(s_f, previous_x_hat[:,f_dec])

                # Do the attention computation
                if self._noisyAttnType == "pcm":
                    #calculate time
                    time = t.ones((1)) * time_space[self.dp.itr - 1]
                    noisy_cond = self.pcm_noise_model.noise_conductance(time, (self._D, self._M))[0]
                    noisy_cond = noisy_cond[0, :, :]
                    attn_read2 = F.linear(s_f[~self.dp.convergence_idx], self._readout[f] * noisy_cond)
                    attn_read2 = attn_read2/t.mean(noisy_cond)
                    attn_read = attn_read2
                else:
                    attn_read = F.linear(s_f[~self.dp.convergence_idx], self._readout[f])
                
                # Scale the results by D to optain a normalized attention
                attn_read = attn_read/self._D
                #plot scatter


                # Add desired noise
                if self._noisyAttn:
                    if self._noisyAttnType == "pcm":
                        #no additional noise
                        attn_read = attn_read
                    else:
                        attn_read = attn_read + self._noise * t.normal(0, 1, attn_read.shape, out=noise)

                # Call activation function
                sim_act = self._activation(attn_read)

                # DEBUG
                mu_temp = t.mean(attn_read)
                std_temp = t.std(attn_read)
                act_temp = float(t.sum(sim_act != 0)/sim_act.shape[0])

                cnt += 1
                mu += mu_temp
                std += std_temp
                act += act_temp
                
                # Add desired noise
                if self._noisyAttn:
                    if self._noisyAttnType != "pcm":
                        sim_act = sim_act + 0.001 * t.normal(0,1, sim_act.shape, out=noise)
                
                self.dp.sim[0, ~self.dp.convergence_idx, f] = sim_act
                if self._noisyAttn:
                    if self._noisyAttnType != "pcm":
                        state_est = t.sign(F.linear(sim_act,t.t(self._reconstruction[f])))
                    else:
                        if self._G0_noise_source_same:
                            state_est = t.sign(F.linear(sim_act, t.t(self._reconstruction[f] * noisy_cond)))
                        else:
                            noisy_cond2 = self.pcm_noise_model_bwd.noise_conductance(time, (self._D, self._M))[0]
                            noisy_cond2 = noisy_cond2[0, :, :]
                            state_est = t.sign(
                                F.linear(sim_act, t.t(self._reconstruction[f] * noisy_cond2)))
                else:
                    state_est = t.sign(F.linear(sim_act,t.t(self._reconstruction[f])))
                state_est[state_est==0]=1


                self.dp.x_hat[~self.dp.convergence_idx, f] = state_est.clone()
                
                # save state for state convergence detection
                # pullUP
                self.dp.state_converged[~self.dp.convergence_idx, f] = (t.max(attn_read, dim=1)[0]>self._pullUp_thresh) 
                # xhat const
                # self.dp.state_converged[~self.dp.convergence_idx, f] = t.all(previous_x_hat[~self.dp.convergence_idx, f] == self.dp.x_hat[~self.dp.convergence_idx, f], dim=1)
                
            # state convergence detection
            # single detection
            self.dp.convergence_idx = (t.sum(self.dp.state_converged, 1) > 0).type(t.bool) 
            # all F
            # self.dp.convergence_idx = (t.sum(self.dp.state_converged, 1)==self._F).type(t.bool)
            
            self.dp.conv_idx[self.dp.convergence_idx] = t.min(self.dp.conv_idx[self.dp.convergence_idx],t.ones_like(self.dp.conv_idx[self.dp.convergence_idx]) * self.dp.itr)
            
            # if reaching 100% accuracy, stop experiment
            if t.sum(self.dp.convergence_idx) == B:
                break
                
            # If reaching 99% accuracy, stop experiment
            # Used for operational capacity experiments
            if self._accuracy:
                if t.sum(self.dp.convergence_idx) >= B * self._accuracy:
                    break
                
        file_object = open(os.path.join(self._savedir, f'{self._id}_statistics.txt'), 'a+')
        file_object.write(f'{self._id}: D={self._D}, Mx={self._M}, F={self._F}, acc={t.sum(self.dp.convergence_idx)/B}, mu={mu/cnt}, std={std/cnt}, act={act/cnt}, T={self._threshold}\n')
        file_object.close()
        
        if self._IMclassification:
            # compute the classes based on similarity to fixed IM
            u_hat, x_hat = self._get_est_idx_IM(self.dp.x_hat)
        elif maxsim_pred:
            u_hat = self._get_est_idx_maxsim(self.dp.sim, self.dp.conv_idx, self.dp.max_iter)
        else:
            print("_get_est_idx")
            u_hat, x_hat = self._get_est_idx(self.dp.sim, self.dp.conv_idx, self.dp.x_hat)
        if recoverDecodingProgress:
            return self.dp.u, u_hat
        else:
            return u_hat


    def _get_number_iter(self):
        """
        Return the number of decoding iterations of last decoding 
        """
        return self.dp.conv_idx+1

    def _topa_sparse(self,sim):
        """
        Generate a new vector based on similarities
        Just use the _topa entries 
        Parameters
        ----------
        sim: torch FloatTensor (_M,)
            result of inner product 
        IM: torch FloatTensor (_M,_D) 
            item memory of that index
        
        Return
        ------
        x:  torch FlaotTensor (_D,), bipolar 
            output vector    
        """
        if self._M - self._topa > 0:
            # get highest abs values
            _, idx = t.topk(t.abs(sim),(self._M-self._topa), largest=False)

            B = sim.size(0)

            idx1= t.arange(B).to(self._device).repeat_interleave(self._M-self._topa).view(-1,1)
            idx2 = idx.view(-1,1)
            superidx=t.cat((idx1,idx2),dim=1)
            sim=sim.index_put(tuple(superidx.t()),t.zeros(superidx.shape[0]).to(self._device))

            if self._pullUp:
                attn_values, attn_index = t.max(sim, dim=1)
                attn_values_PU_mask = attn_values >= self._pullUp_thresh

                # HIGH LEVEL CYCLES
                sim[attn_values_PU_mask] = 0                
                sim[attn_values_PU_mask, attn_index[attn_values_PU_mask]] = attn_values[attn_values_PU_mask]
                
        return sim

    def _topa_sparse_positiv(self,sim):
        """
        Generate a new vector based on similarities
        Just use the _topa entries 
        Parameters
        ----------
        sim: torch FloatTensor (_M,)
            result of inner product 
        IM: torch FloatTensor (_M,_D) 
            item memory of that index
        
        Return
        ------
        x:  torch FlaotTensor (_D,), bipolar 
            output vector    
        """

        if self._M - self._topa > 0:
            # get highest values
            _, idx = t.topk(sim,(self._M-self._topa), largest=False)

            B = sim.size(0)

            idx1= t.arange(B).to(self._device).repeat_interleave(self._M-self._topa).view(-1,1)
            idx2 = idx.view(-1,1)
            superidx=t.cat((idx1,idx2),dim=1)
            sim=sim.index_put(tuple(superidx.t()),t.zeros(superidx.shape[0]).to(self._device))
            
            if self._pullUp:
                attn_values, attn_index = t.max(sim, dim=1)
                attn_values_PU_mask = attn_values >= self._pullUp_thresh

                # HIGH LEVEL CYCLES
                sim[attn_values_PU_mask] = 0            
                sim[attn_values_PU_mask, attn_index[attn_values_PU_mask]] = attn_values[attn_values_PU_mask]

        return sim

    def _topa_sparse_threshold_positiv(self,sim):
        """
        Generate a new vector based on similarities
        Just use the _topa entries 
        Parameters
        ----------
        sim: torch FloatTensor (_M,)
            result of inner product 

        Return
        ------
        x:  torch FlaotTensor (_D,), bipolar 
            output vector    
        """

        sim_t = sim.clone()
        sim_t[sim_t < self._threshold] = 0
        
        if self._pullUp:
            attn_values, attn_index = t.max(sim, dim=1)
            attn_values_PU_mask = attn_values >= self._pullUp_thresh

            # HIGH LEVEL CYCLES
            sim[attn_values_PU_mask] = 0
            sim[attn_values_PU_mask, attn_index[attn_values_PU_mask]] = attn_values[attn_values_PU_mask]
            
        # sum = t.sum(sim_t,1)
        # sim_t[sum == 0] = sim[sum == 0] # if all zero recover by not applying any theshold

        return sim_t

    def _ist(self,sim):
        """
        Generate a new vector based on similarities
        Iterative soft thresholding
        Parameters
        ----------
        sim: torch FloatTensor (_M,)
            result of inner product 
        
        Return
        ------
        x:  torch FlaotTensor (_D,), bipolar 
            output vector    
        """
        tau = self._topa

        sim =F.relu(t.abs(sim)-tau)*t.sign(sim)

        return sim

    def _ist_mean(self,sim):
        """
        Generate a new vector based on similarities
        Iterative soft thresholding
        Parameters
        ----------
        sim: torch FloatTensor (_M,)
            result of inner product 
        
        Return
        ------
        x:  torch FlaotTensor (_D,), bipolar 
            output vector    
        """

        tau = t.mean(t.abs(sim))*self._topa

        sim =F.relu(t.abs(sim)-tau)*t.sign(sim)
        return sim


    def _get_est_idx(self,sim,conv_idx,x_hat):
        """
        Estimate index according to similarities, 
        consider also the sign flips 
        Parameters
        ----------result
        sim: torch FloatTensor (_V,_M)
            similarity
        
        Return
        ------
        u_hat: torch FloatTensor (_V)
        """
        B = sim.shape[1]
        u_hat = t.Tensor(B,self._F).zero_().to(self._device)
        for b in range(B):

            # u_hat[b] = t.argmax(t.abs(sim[conv_idx[b],b]),1)
            u_hat[b] = t.argmax(t.abs(sim[0,b]),1)

            # correct the states if sign is negative (degeneracy)
            for v in range(self._F):
                # if sim[conv_idx[b],b,v,int(u_hat[b,v])] < 0.0:
                if sim[0,b,v,int(u_hat[b,v])] < 0.0:

                    x_hat[b,v] = x_hat[b,v]* -1

        return u_hat, x_hat


    def _get_est_idx_maxsim(self,sim,conv_idx,max_iter):
        """
        Estimate index according to similarities, 
        consider also the sign flips 
        Parameters
        ----------result
        sim: torch FloatTensor (L,B,_V,_M)
            similarity
        
        Return
        ------
        u_hat: torch FloatTensor (_V)
        """
        L,B,V,M = sim.shape
        u_hat = t.Tensor(B,V).zero_().to(self._device)
        for b in range(B):
            if conv_idx[b] != L-1:
                u_hat[b] = t.argmax(t.abs(sim[conv_idx[b],b]),1)
            else:
                sign_correct=False
                cnt = 0
                final_idx = L-1
                val,M_idx=t.max(t.abs(sim[:,b]),-1)
                while (not sign_correct) and cnt<L:

                    idx = t.argmax(t.sum(val,-1)-1)
                    u_hat[b] = t.argmax(t.abs(sim[idx,b]),1)

                 # check the sign
                    sign_cnt=0
                    for v in range(V):
                        if sim[idx,b,v,int(u_hat[b,v])] < 0.0:
                            sign_cnt +=1
                    sign_correct = (sign_cnt%2==0)

                    if not sign_correct:
                        val[idx,:]=0

                    cnt+=1
        return u_hat


    def _get_est_idx_IM(self,x_hat):
        """
        Estimate index according to similarities 
        based on similarity to estimated state, not activation
        consider also the sign flips 
        Parameters
        ----------
        x_hat: torch Tensor (_B,_V,_D)
        
        Return
        ------
        u_hat: torch FloatTensor (_V)
        """

        B = x_hat.shape[0]
        u_hat = t.Tensor(B,self._F).zero_().to(self._device)
        sim = t.Tensor(B,self._F,self._M).zero_().to(self._device)

        for v in range(self._F):

            #pdb.set_trace()
            sim[:,v] = F.linear(x_hat[:,v],self._IM[v])
            u_hat[:,v] = t.argmax(t.abs(sim[:,v]),1)

        # correct the states if sign is negative (degeneracy) 
        for b in range(B):
            for v in range(self._F):
                if sim[b,v,int(u_hat[b,v])] < 0.0:
                    x_hat[b,v] = x_hat[b,v]* -1

        return u_hat, x_hat
    
    def filename(self):
        """
        Retruns the file name for the npz arrays
        """
        return "{}_{}_D{:01d}_V{:01d}_{}_T{:04d}_N{:04d}".format(self._id, "{}".format(__file__).split("/")[-1].split(".")[0], self._D, self._F, self._activationName, int(1000*self._threshold), int(self._noise*1000))

def plot_Meas_vs_Ideal(measured, ideal, cmap=plt.cm.brg, title="", xlabel="Ideal", ylabel="Measured", alpha=0.5, xlim=None, ylim=None, path="fig"):
    plt.figure(title)
    plt.clf()
    active_range = np.arange(0, measured.shape[1])

    std = ((measured-ideal)).std()

    for i in (active_range):
        plt.scatter(ideal[:,i], measured[:,i], color=cmap(i/len(active_range)), alpha=alpha)

    plt.title("{}, std={:.4f}".format(title, std))
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(os.path.join("png", path))
    plt.close()