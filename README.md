# Reinforcement Learning based Learning of Dynamic Sepsis Treatment Strategies, in a Partially Observed MDP setting

## Still work in progress
## Progress
### Physiology Aware Sequential Autoencoder
We propose an inference structure which could infer patient and intervention specific cardiovascular states, by a Sequential AutoEncoder, which is implicitly regularized by constraining the latent representation to have phyisilogical meaning and the decoder to be a mathematical model. This has the benefit from a simplicity, but has the potential to give useful Cardiovascular information in the ICU settings.


### Denoising GRU AutoEncoder for Labs
Since the labs are much more sparse, we use a denoising stacked sequential autoencoder structure with low dimensional final hidden layer to encode the history of the labs, This is supposed to give a much better representation which then could be used as a state component in the (PO)MDP



### Data
Open [mimic] database was used as the data source. Relationships of the schema can be found [here][schema]. You need permission to access MIMIC Data, more information on this is given on their website. After getting access I would recommend using Big Query MIMIC project to quickly access already processed schema.

Most of the preprocessing was done on Google Big Query Mimic-iii project,and preprocessed and cleansed using SQL on Big Query and Pandas. Pivoted Labs, Sofa Score related measurements and (Most Vasopressors) and Vitals are all available on Big Query in Derived Tables. 

SQL based data cleaning and processing code is available on the SQL folder. Input_cv.sql and Input_mv.sql are the SQL files used to extract the fluids and the equivalent volumes from BigQuery. (CV and MV denotes Carevue and MetaVision as inputs are in two seperate databases). These are combined before using for Modelling/RL.

To Run the analysis we need, Pivoted Vitals,Sofa Scores which includes Vasopressors (including Vasopressin),Labs and Fluids. The RL cohort is included in terms on icustay ids for convenience.


### Reinforcement Learning
For the RL implemntations, for convenience, I had saved states (which includes derived states), actions and rewards in (a rather large) csv file. This makes the replay buffer/batching straightforward application of PyTorch Dataset and Dataloader.









  [schema]:<https://mit-lcp.github.io/mimic-schema-spy/index.html>
   [mimic]:<https://mimic.physionet.org/mimicdata>
