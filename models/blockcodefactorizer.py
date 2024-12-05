#!/usr/bin/env python3

''' Sparse binary block code resonator network
'''

import sys
import torch as t
import torch.nn.functional as F
import random

__author__ = "Michael Hersche"
__email__ = "her@zurich.ibm.com"

class blockcodefactorizer:
    def __init__(self, D, F, Mx, B,
                 similarity = "inf",
                 convergenceDetectionThreshold = 0.5, 
                 A=10, 
                 threshold=0.5,
                 decoding="sequential", 
                 IM=None, 
                 permutation=False, 
                 topaPU=True, 
                 useCuda=False, 
                 gpu=None, 
                 seed=7,
                 id="",
                 **kwargs):

        """
        Implementation of the resonator network with a threshold similarity activation and random sampling restart.

        Main parameters
        ----------
        id: str
            Experiment id
        D: int 
            Vector dimension
        F: int 
            Number of factors
        Mx: int
            Item memory size, code book size for every factor
        B: int
            Number of sparse blocks

        Nonlinearities and the similarity function
        ------------------------------------------
        similarity: str
            Similarity function ["dotp", "inf", "l1"]

        Hyperparameters
        ---------------
        convergenceDetectionThreshold: float
            Factor estimate convergence detection similarity threshold
        threshold: float
            Similarity activation threshold
        A: int
            Number of similarities to sample

        General options
        -------------------------
        decoding: str
            Update strategy in resontator: "sequential" , "parallel"
        IM: torch tensor (V,Mx,D)
            predefined IM, if None new IM will be initialized
        permutation: bool
            Generate codebooks for all factors by blockwise permuting the first factor codebook on/off
        topaPU: bool
            Similarity pullup on/off
        batchsize: int
            Compute in batches of given size
        use_cuda: bool
            use of CPU (False, default) or CUDA (True)
        gpu: str
            Indicate specific GPU
        seed: int
        """

        random.seed(seed)
        t.manual_seed(seed)

        self._id = id
        self._D = D
        self._F = F
        self._Mx = Mx
        self._B = B
        self._L = self._D//self._B
        self._permutation = permutation
        self._decoding = decoding
        self._topaPU = topaPU
        self._A = A
        self._similarityName = similarity
        self._pullup_thresh = convergenceDetectionThreshold


        print(f'D={self._D}')

        # Declare the similarity metric
        similarities = {'dotp': self._dotp_similarity,
                        'inf': self._inf_similarity,
                        'l1': self._manhattan_similarity}

        self._similarity = similarities[similarity]

        # Table for scaling the pullup threshold according to used similarity metric
        # Corresponds to the largest possible value of the given similarity metric
        pullup_scaling = {
            "dotp": self._B,
            "manh": 2 * self._B,
            "inf": 1
        }

        self._pullup_thresh *= pullup_scaling[similarity]
        self._threshold = threshold
        
        # Cuda
        if useCuda and t.cuda.is_available():
            if gpu is None:
                self._device = f'cuda:{t.cuda.current_device()}'
            else:
                self._device = f'cuda:{gpu}'
        else:
            self._device = 'cpu'
        
        print("Use device {:}".format(self._device))

        #### IM Initialization ####

        # Load a pre-specified item memory
        if IM is not None:
            print("Loading specified codebook...")
            self._IM = IM.to(self._device)

            # Permute
            if self._permutation:
                for f in range(self._F):
                    self._IM[f] = t.roll(self._IM[0].detach(),f,1)
            self._matIM = t.zeros((F, Mx, B, self._L,), dtype=t.float32, device=self._device)
            for f in range(F):
                for m in range(Mx):
                    for k_it in range(B):
                        self._matIM[f, m, k_it,self._IM[f,m,k_it].int()]=1
            print("Finished loading specified codebook.")

        # Create a random item memory
        else:
            print("Constructing random codebook...")
            self._IM = t.randint(0, self._L, (F, Mx, B,), dtype=t.float32).to(self._device)
            
            # Permute
            if self._permutation:
                for f in range(self._F):
                    self._IM[f] = t.roll(self._IM[0].detach(),f,1)
            self._matIM = t.zeros((F, Mx, B, self._L,), dtype=t.float32).to(self._device)
            for f in range(F):
                for m in range(Mx):
                    for k_it in range(B):
                        self._matIM[f, m, k_it,self._IM[f,m,k_it].int()]=1

        #### Initial guess generation ####

        # Generate initial guess as a superposition of A-many random codevectors
        self._init_guess = t.sum(self._matIM, dim=1).to(self._device)
        self._init_guess /= self._Mx
        
        print(f"Activation threshold: {self._threshold}")
        print(f"Number of sampled vectors: {self._A}")
        print("(B, D, F, M) = ({}, {}, {}, {})".format(self._B, self._D, self._F, self._Mx))
        
    def binding_circular(self, A, B):

        """
        Binds two hypervectors by performing block-wise circular convolution. 
        Robust method since erroneous bits or noise in a particular block
        only affect the binding result in that segment. The binding operation commutes.
        Result will be a vector of dimension D that does not resemble any of the
        input vectors.

        Parameters
        ----------
        A: torch FloatTensor (batchsize, k, L)
            sparse binary hypervector of dimension D=k*L
        B: torch FloatTensor (batchsize, k, L)
            sparse binary hypervector of dimension D=k*L

        Returns
        -------
        C: torch FloatTensor (batchsize, k, L)
            sparse binary hypervector of dimension D=k*L, he result of the binding operation.
        """

        batchSize = A.shape[0]
        C = t.zeros(batchSize, self._B, self._L).to(self._device)
        
        # prepare inputs
        A = t.unsqueeze(A,1) # input
        B = t.unsqueeze(B,2) # filter weigths
        B = t.flip(B, [3]) # flip input
        B = t.roll(B, 1, dims=3) # roll by one to fit addition

        # reshape for single CONV
        A = t.reshape(A, (1, A.shape[0]*A.shape[2], A.shape[3]))
        B = t.reshape(B, (B.shape[0]*B.shape[1], B.shape[2], B.shape[3]))

        # calculate C = t.remainder(B+A*alpha, self._L)
        C = F.conv1d(F.pad(A, pad=(0,self._L-1), mode='circular'), B, groups=self._B*batchSize)
        
        # reshape to be resonator complaint 
        C = t.reshape(C, (batchSize, self._B, self._L))

        return C

    def inv_binding_circular(self, A, C):

        """
        Block-wise circular correlation.
        Acts as an approximate inverse of block-wise cicular convolution.

        Parameters
        ----------
        A: torch FloatTensor (batchsize, k, L)
            sparse binary hypervector of dimension D=k*L
        C: torch FloatTensor (batchsize, k, L)
            sparse binary hypervector of dimension D=k*L

        Returns
        -------
        B: torch FloatTensor (batchsize, k, L)
            sparse binary hypervector of dimension D=k*L, result of unbinding A from C
        """

        batchSize = A.shape[0]
        B = t.zeros(self._B*batchSize, self._L).to(self._device)

        A = t.unsqueeze(A,1) # input
        C = t.unsqueeze(C,2) # filter weigths

        A = t.reshape(A, (1, A.shape[0]*A.shape[2], A.shape[3]))
        C = t.reshape(C, (C.shape[0]*C.shape[1], C.shape[2], C.shape[3]))
        
        B = F.conv1d(F.pad(A, pad=(0,self._L-1), mode='circular'), C, groups=self._B*batchSize)
        B = t.reshape(B, (batchSize, self._B, self._L))
        
        B = t.flip(B, [2]) # flip input
        B = t.roll(B, 1, dims=2) # roll by one to fit addition

        return B

    def encode(self, u):

        """
        Multiplicative encoding
        Parameters
        ----------
        u: torch int tensor (B,_F,) with integer values {0,_M-1}
            input index array with batch size B       
        Return
        ------
        x: torch FloatTensor (B,_k,)
            encoded HD vector 
        """
        
        B = u.shape[0]        
        x_circular = t.zeros(B, self._B, self._L).to(self._device)
        x_circular[:,:,0] = 1

        for v in range(self._F):
            x_circular = self.binding_circular(x_circular, self._matIM[v, u[:,v]])

        x_circular = x_circular.type(dtype=t.int32)

        return x_circular

    def decode(self, s, max_iter=2000):

        """
        Decode HD vector to data with resonator circuit

        Parameters
        ----------
        s: torch FloatTensor (B,_k,_L)
            sparse, binary input tensor
        max_iter: int 
            max number of decoding iterations

        Return
        ------
        u_hat: torch int tensor (_F,) with values {0,_M-1}
            input index array       
        """

        B = s.shape[0]
        s = s.type(t.float)

        # Initialize factor estimates
        x_hat = self._init_guess.clone().view(1,self._F,self._B, self._L).repeat(B,1,1,1)
        
        # Initialize similarity matrix
        sim = t.Tensor(B, self._F, self._Mx).zero_().to(self._device) 
        
        # Initialize convergence detection matrix
        state_converged = t.Tensor(B,self._F).zero_().to(self._device)

        # Keep track of convergence speed
        self.conv_idx = t.Tensor(B,1).fill_(max_iter-1).to(self._device)

        # In each iteration of the decoding loop
        for itr in range(max_iter):

            state_converged.zero_()
            previous_x_hat = x_hat.clone() # Used for parallel decoding

            for f in range(self._F):
                s_f = s.clone()

                # Unbind estimates from query
                for f_dec in range(self._F):
                    if f != f_dec:
                        if self._decoding=="sequential": 
                            # here we use the most recent estimation (sequential)
                            s_f = self.inv_binding_circular(x_hat[:,f_dec], s_f)
                        else:
                            # decoding in parallel
                            s_f = self.inv_binding_circular(previous_x_hat[:,f_dec], s_f)

                # Measure the similarities of the unbound factor estimate and the codebook
                sim[:,f] = self._similarity(codebook=self._matIM[f], vector=s_f)

                # Threshold based similarity sparsification
                if self._threshold > 0:
                    attn = self._est_topa_treshold(sim[:,f])
                else:
                    attn = sim[:,f]

                # Indicates in which trials in the batch the largest similarity value is below the threshold
                threshold_mask = t.max(attn, dim=-1)[0] <= self._threshold

                # If the indicator is not all-zero
                if not (t.sum(threshold_mask) == 0).item() and not self._A == 0:

                    # The sampled similarity vector
                    similarity = t.zeros(self._Mx).to(self._device)

                    if self._A>= self._Mx:
                        similarity += self._A # Activate all elements
                    else:
                        locations = random.sample(range(self._Mx), self._A)
                        similarity[locations] += self._A # Activate a random subset of the elements

                    attn[threshold_mask] = similarity.view(1,self._Mx).expand(threshold_mask[threshold_mask==1].shape[0], -1)
                
                # Pull-up high similarity values
                if self._topaPU:
                    attn_values, attn_index = t.max(attn, dim=1)

                    # Pull up those values which are above the pullup threshold but were not sampled in the previous step
                    attn_values_PU_mask = (attn_values >= self._pullup_thresh) * (~threshold_mask)
                    attn[attn_values_PU_mask] = 0
                    attn[attn_values_PU_mask, attn_index[attn_values_PU_mask]] = attn_values[attn_values_PU_mask]

                # Detect convergence
                state_converged[:,f] = (t.max(attn, dim=1)[0]>=self._pullup_thresh) * (~threshold_mask)

                # Calculate the  weighted superposition
                superpos = t.matmul(t.transpose(t.transpose(self._matIM[f], 0, 1), 1, 2), t.transpose(attn, 0, 1))
                x_hat[:,f] = t.transpose(t.transpose(superpos, 1, 2), 0, 1)

                # Normalize blocks of superposition
                x_hat[:,f] = t.div(x_hat[:,f], t.sum(x_hat[:,f], dim=2).unsqueeze(-1))

            convergence_idx = t.sum(state_converged,1)==self._F
            self.conv_idx[convergence_idx] = t.min(self.conv_idx[convergence_idx],t.ones_like(
                                                            self.conv_idx[convergence_idx])*itr)

            # Convergence detection
            if t.sum(state_converged)==self._F*B:
                break

        u_hat = self._get_est_idx(sim)

        return u_hat

    def _est_topa_treshold(self, sim):

        """
        Sparsify the similarity matrix using the threshold activation

        Input
        ------
        sim:    torch FloatTensor (batchsize, Mx)

        Return
        ------
        sim:    torch FloatTensor (batchsize, Mx),
                In each batch, only those similarities that are larger than the threshold are preserved, all others are 0.
                In case no similarity is larger than the threshold, all are preserved.
        """
        
        sim_t = sim.clone()
        sim_t[sim_t < self._threshold] = 0

        # if all similarities are == 0, do not apply threshold
        sum = t.sum(sim_t,1)
        sim_t[sum == 0] = sim[sum == 0]

        return sim_t

    def _dotp_similarity(self, codebook, vector):

        """
        Inputs:
        ------- 
        codebook:   torch FloatTensor of shape (Mx, k, L).
                    Stores all hypervectors of a given factor.

        vector:     torch FloatTensor of shape (batchsize, k, L). 
                    The factor estimates generated by unbinding other estimates from the query.

        Output:
        ------- 
        similarity: torch FloatTensor of shape (batchsize, Mx). 
                    The dot product similarity of the input vector with all of the codevectors.
        """

        batchsize = vector.shape[0]
        similarity = t.transpose(t.sum(t.sum(t.transpose(codebook.repeat(batchsize, 1, 1, 1), 0, 1) * vector.float(), dim=2), dim=2), 0, 1)
        return similarity

    def _manhattan_similarity(self, codebook, vector):

        """
        Inputs:
        ------- 
        codebook:   torch FloatTensor of shape (Mx, k, L).
                    Stores all hypervectors of a given factor.

        vector:     torch FloatTensor of shape (batchsize, k, L). 
                    The factor estimates generated by unbinding other estimates from the query.

        Output:
        ------- 
        similarity: torch FloatTensor of shape (batchsize, Mx). 
                    The L-1 (Manhattan) similarity of the input vector with all of the codevectors.
        """

        batchsize = vector.shape[0]
        similarity =  2*self._B - t.sum(t.sum(t.abs(vector.repeat(self._Mx,1,1,1).permute(1,0,2,3)-codebook.repeat(batchsize,1,1,1)), dim=2), dim=2)
        return similarity

    def _inf_similarity(self, codebook, vector):

        """
        Inputs:
        ------- 
        codebook:   torch FloatTensor of shape (Mx, k, L).
                    Stores all hypervectors of a given factor.

        vector:     torch FloatTensor of shape (batchsize, k, L). 
                    The factor estimates generated by unbinding other estimates from the query.

        Output:
        ------- 
        similarity: torch FloatTensor of shape (batchsize, Mx). 
                    The L-infinity similarity of the input vector with all of the codevectors.
        """

        batchsize = vector.shape[0]
        similarity = 1 - t.amax(t.amax(t.abs(vector.view(batchsize, 1, self._B, self._L).expand(-1, self._Mx, -1, -1)-codebook.view(1, self._Mx, self._B, self._L).expand(batchsize, -1, -1, -1)), dim=-1), dim=-1)
        return similarity

    def _get_est_idx(self, sim):
        """
        Estimate index according to similarities
        
        Parameters
        ----------
        sim: torch FloatTensor (B,_F,_M)
            similarity

        Return
        ------
        u_hat: torch FloatTensor (B,_F)
        """
        u_hat = t.argmax(t.abs(sim), dim = 2)
        return u_hat
    
    def _get_number_iter(self):
        """
        Return the number of decoding iterations of last decoding 
        """
        return self.conv_idx+1

    def filename(self):
        """
        Retruns the file name for the npz arrays
        """
        return "{}_{}_A_{}_thresh_{}_pullup_{}_pullupthresh_{}_k_{}_F_{}_D_{}_".format(self._id, self._similarityName, \
            self._A, self._threshold, self._topaPU, self._pullup_thresh, self._B, self._F, self._D)