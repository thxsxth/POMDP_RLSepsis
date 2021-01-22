The following image illustrates the archiecture of the physiology driven autoencoder. The mechanistic model is a set of algebriac equations derived by analytically solving 2 parameter lumped Windkessell ODE. 

The notebook Physiology_informed_Sequential_AutoEncoder_(With_Denoising).ipynb presents training of the physiologic model. load_and_process deals with the imports processing and defines the dataset and dataloader classes.
  
### Note: 
We have made state dictionaries of the autoencoder available under ./State_Dicts ( 0.1_denoise_auto_17.72.pt for 10% corruption and autoen_final (when trained with no corruption) training from sratch requires a low learning rate, and multiple epochs


![alt text](https://github.com/thxsxth/POMDP_RLSepsis/blob/master/Images/auto_en_diag%20(1).png)
