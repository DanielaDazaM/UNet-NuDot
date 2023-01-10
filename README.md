# UNet-NuDot.
UNet development for the NuDot experiment


## Fake Data generation.

### General set up.
We begin defining the function "SER" based on the paper --- with $\tau = 4$ and $\sigma = 0.5$. After this, we define the folloring variables:
* Test: It is useful if we generate different test and it is only involved when naming the files. 
* n_events: Determines the number of events generated.
* max_photons: Determines the maximum number of photos generated in each event. 
* n_samples: Indicates the number of time samples considered for generation. 
* n_noise: Specifies how much noise is added to the fake data after the generation of the wave forms. 

Writen by: Daniela Daza Marroquín.

### Generation of the number of photons for each event.

We define the number of photons considered for an event. It is done by generating an array of $n_{events}$ such that each element is a random number between 0 and $max\  photons$.

### Signals for each event.
The following is made for each event through a loop.

Initially, we extract the number of photons for that event from the array $n_{photons}$ and with this, we generate the hitting time $t_0$ of each photon by generating random numbers between 15 and 80. Also, we generate the maximum voltage $V_0$ obtained for the signal of each photon individually such that they follow a normal distribution. 


### Generation of the True files.
These are two important variables:
* ys:
* ccs: It is an array full with zeros of size $len(t)\times 2(max\ photons)$. The true values will be saved in this array. 

The following is done for each photon through a loop.

We define the the selection rules (this are boolean arrays in which the elements are True if the time conditions are satisfied): 
* sel: It is True for every sample with $t< t_0$.
* sel_inc: It is True if the wave form is increasing or is at its maximum.
* sel_dec: It is True if the wave form is decreasing.

After this, we specify the wave form for each photon by taking $$y=SER(V_0, t-t_0)$$ and set all values of $y$ before $t_0$ to be zero.


Later, we save the data in ccs such that the components of the form $$ccs\[:,2\ jj\]$$ will be 1 if the wave form of the $jj$-th photon is increasing and 0 if it is not. Analogously, the components of the form $$ccs\[:,2\ jj+1\]$$ will be 1 if the wave form of the $jj$-th photon is decreasing and 0 if it is not.

## U Network for NuDot 
