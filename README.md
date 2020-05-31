# Beta-VAE 
Original paper: https://openreview.net/pdf?id=Sy2fzU9gl <br>
Implementation includes improvements from: https://arxiv.org/abs/1804.03599

<b>Note:</b> Models were trained on random subset of 150,000 images from the dsprites data set due to lack of compute power close to NeurIPS deadline 2020.

Performance was evaluated on DCI metric. <br>
Black and white dsprites data was used.

### DCI Metric performance
![DCI](figures/Figure_4_flat.png)


<p> <b>Color:</b> Since we are using black-white no latent variable is responsible<br>
<b>Shape:</b> μ_10 <br>
<b>Scale:</b> Unclear, looks to be combination of μ_4, μ_6, μ_8  <br>
<b>Orientation:</b> μ_10 <br>
<b>X-axis Position:</b> μ_8 <br>
<b>Y-axis Position:</b>  μ6 </p>


It appears the latent space has not learned that the shape and orientiation as a continuous space. Likely due to the small subset of samples shown. I expect performance to increase as training data set in increased. 

### Reconstructions:
![Recons](figures/Figure_3.png)

### Latent Traversals:
![Trav2](figures/mu_gifs/mu1_var1.gif)
![Trav2](figures/mu_gifs/mu1_var2.gif)
![Trav2](figures/mu_gifs/mu1_var3.gif)
![Trav2](figures/mu_gifs/mu1_var4.gif)
![Trav2](figures/mu_gifs/mu1_var5.gif)
![Trav2](figures/mu_gifs/mu1_var6.gif)
![Trav2](figures/mu_gifs/mu1_var7.gif)
![Trav2](figures/mu_gifs/mu1_var8.gif)
![Trav2](figures/mu_gifs/mu1_var9.gif)
![Trav2](figures/mu_gifs/mu1_var10.gif)

![Trav2](figures/mu_gifs/mu2_var1.gif)
![Trav2](figures/mu_gifs/mu2_var2.gif)
![Trav2](figures/mu_gifs/mu2_var3.gif)
![Trav2](figures/mu_gifs/mu2_var4.gif)
![Trav2](figures/mu_gifs/mu2_var5.gif)
![Trav2](figures/mu_gifs/mu2_var6.gif)
![Trav2](figures/mu_gifs/mu2_var7.gif)
![Trav2](figures/mu_gifs/mu2_var8.gif)
![Trav2](figures/mu_gifs/mu2_var9.gif)
![Trav2](figures/mu_gifs/mu2_var10.gif)


![Trav2](figures/mu_gifs/mu3_var1.gif)
![Trav2](figures/mu_gifs/mu3_var2.gif)
![Trav2](figures/mu_gifs/mu3_var3.gif)
![Trav2](figures/mu_gifs/mu3_var4.gif)
![Trav2](figures/mu_gifs/mu3_var5.gif)
![Trav2](figures/mu_gifs/mu3_var6.gif)
![Trav2](figures/mu_gifs/mu3_var7.gif)
![Trav2](figures/mu_gifs/mu3_var8.gif)
![Trav2](figures/mu_gifs/mu3_var9.gif)
![Trav2](figures/mu_gifs/mu3_var10.gif)



![Trav2](figures/mu_gifs/mu4_var1.gif)
![Trav2](figures/mu_gifs/mu4_var2.gif)
![Trav2](figures/mu_gifs/mu4_var3.gif)
![Trav2](figures/mu_gifs/mu4_var4.gif)
![Trav2](figures/mu_gifs/mu4_var5.gif)
![Trav2](figures/mu_gifs/mu4_var6.gif)
![Trav2](figures/mu_gifs/mu4_var7.gif)
![Trav2](figures/mu_gifs/mu4_var8.gif)
![Trav2](figures/mu_gifs/mu4_var9.gif)
![Trav2](figures/mu_gifs/mu4_var10.gif)


![Trav2](figures/mu_gifs/mu5_var1.gif)
![Trav2](figures/mu_gifs/mu5_var2.gif)
![Trav2](figures/mu_gifs/mu5_var3.gif)
![Trav2](figures/mu_gifs/mu5_var4.gif)
![Trav2](figures/mu_gifs/mu5_var5.gif)
![Trav2](figures/mu_gifs/mu5_var6.gif)
![Trav2](figures/mu_gifs/mu5_var7.gif)
![Trav2](figures/mu_gifs/mu5_var8.gif)
![Trav2](figures/mu_gifs/mu5_var9.gif)
![Trav2](figures/mu_gifs/mu5_var10.gif)


### Latent_traversal plots:
![Trav1](figures/Traversal1.png)
![Trav2](figures/Traversal2.png)


### Loss curves:
![Loss](figures/Figure_2.png)
