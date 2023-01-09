# UNet-NuDot
UNet development for the NuDot experiment


## Fake Data generation 

### General set up
We begin defining the function "SER" based on the paper --- with $$ \tau = 4$$ and $$ \sigma = 0.5$$. After this, we define the folloring variables:
* Test: It is useful if we generate different test and it is only involved when naming the files. 
*n_events: Determines the number of events generated.
*max_photons: Determines the maximum number of photos generated in each event. 
*n_samples: Indicates the number of time samples considered for generation. 
*n_noise: Specifies how much noise is added to the fake data after the generation of the wave forms. 

### Generation of the number of photons for each event

We define the number of photons considered for an event. It is done by generating an array of $$n_{events}$$ such that each element is a random number between 0 and $$max photons$$.

### Signals for each event

The following is made for each event through a loop.

Initially, we extract the number of photons for that event from the array $$n_{photons}$$ and with this, we generate the hitting time $$t_0$$ of each photon by generating random numbers between 15 and 80. Also, we generate the maximum voltage obtained for the signal of each photon individually such that they follow a normal distribution. 



## U Network for NuDot 
