# Reinforcement Learning based 'Learning' of Dynamic Sepsis Treatment Strategies

## Still work in progress
#### Progress
## Physiology Aware Sequential Autoencoder
We propose an inference structure which could infer patient and intervention specific cardiovascular states, by a Sequential AutoEncoder, which is implicitly regularized by constraining the latent representation to have phyisilogical meaning and the decoder to be a mathematical model. This has the benefit from a simplicity, but has the potential to give useful Cardiovascular information in the ICU settings.





### Data
Open [mimic] database was used as the data source. Relationships of the schema can be found [here][schema].

Most of the preprocessing was done on Google Big Query Mimic-iii project,and preprocessed and cleansed using SQL on Big Query and Pandas.









  [schema]:<https://mit-lcp.github.io/mimic-schema-spy/index.html>
   [mimic]:<https://mimic.physionet.org/mimicdata>
