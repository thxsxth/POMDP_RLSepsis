# Unifying Cardiovascular Modelling with Deep Reinforcement Learning for Uncertainty Aware Control of Sepsis Treatment.

This repository conatins code for Reinforcement Learning based Dynamic Treatment learning, respecting partial observability for Sepsis Treatment. We use a novel Cardiovascular Physiology based autoencoder, with learns patient specific Cardiovasular states, this structure is expected to convey important Physiological information which can't be directly observed. We also use another denoising Autoencoder to represent the history of the labs, and then use Offline Reinforcement Learning to learn optimal treatment strategies.

## Still work in progress

### Data
Open [mimic] database was used as the data source. Relationships of the schema can be found [here][schema]. You need permission to access MIMIC Data, more information on this is given on their website. After getting access I would recommend using Big Query MIMIC project to quickly access already processed schema.

Most of the preprocessing was done on Google Big Query Mimic-iii project,and preprocessed and cleansed using SQL on Big Query and Pandas. Pivoted Labs, Sofa Score related scores and Vitals are all available on Big Query in Derived Tables. 

SQL based data cleaning and processing code is available on the SQL folder. These are combined before using for Modelling/RL.

To Run the analysis we need, Pivoted Vitals,Sofa Scores and hourly vasopressor and fluid treatments. The RL cohort is included in terms on icustay ids for convenience.


## Progress
### Physiology Aware Sequential Autoencoder
We propose an inference structure which could infer patient and intervention specific cardiovascular states, by a Sequential AutoEncoder, which is implicitly regularized by constraining the latent representation to have phyisilogical meaning and the decoder to be a mathematical model. This has the benefit from a simplicity, but has the potential to give useful Cardiovascular information in the ICU settings.


### Denoising GRU AutoEncoder for Labs
Since the labs are much more sparse, we use a denoising stacked sequential autoencoder structure with low dimensional final hidden layer to encode the history of the labs, This is supposed to give a much better representation which then could be used as a state component in the (PO)MDP



### Reinforcement Learning
For the RL implemntations, for convenience, I had saved states (which includes derived states), actions and rewards in (a rather large) csv file. This makes the replay buffer/batching straightforward application of PyTorch Dataset and Dataloader.

Initially tried Continous Batch Constrained Deep Q Learning ([BCQ]) and [Discrete BCQ] (with some minor modifications) as the DRL alogrithms. BCQ is an offline batch RL algorithm with as the name suggests regulaizes state actions pairs based on what is included in the batch.

Final work uses Distributinal RL (C51) algorithm, further we use uncertainty Quantification to quatify epistemic uncertainty.

# Redproducing Results
To reproduce our results, you would first need to get access to MIMIC-iii data, and by using SQL queries in SQL folder (or otherwise), derive equivalent vasopressor and fluid treatments. Careview and Metavision tables have different formats of input events. Once this is done you could use the PyTorch dataset, dataloader classes to train the representation learning modules. 

Finally to run RL, you need to way to generate  (s,a,s',reward,done) tuples, where s includes the inferred hidden states. For this work, we save the states in a csv file, and then used a Pytorch Dataloader to generate batches for RL. A basic/naive way of this process is shown in the RL/RL_csv.ipynb notebook.

Refer RL folder for RL modules. distRL has all the RL modules.





  [schema]:<https://mit-lcp.github.io/mimic-schema-spy/index.html>
   [mimic]:<https://mimic.physionet.org/mimicdata>
   [Discrete BCQ]:<https://arxiv.org/abs/1910.01708>
   [BCQ]:<https://arxiv.org/abs/1812.02900>
