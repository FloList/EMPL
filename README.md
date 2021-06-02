# The Earth Mover's Pinball Loss: <br> Quantiles for Histogram-Valued Regression
This is the Tensorflow implementation of the paper *The Earth Mover's Pinball Loss: Quantiles for Histogram-Valued Regression* (**ICML 2021**).
The Earth Mover's Pinball Loss (EMPL) is a loss function for Deep Learning-based histogram regression, which incorporates cross-bin information and yields *distributions* over plausible histograms, expressed in terms of quantiles of the cumulative histogram in each bin. The EMPL compares two (normalised) histograms <a href="#"><img src="https://render.githubusercontent.com/render/math?math=u=(u_j)_{j=1}^N"></a> and <a href="#"><img src="https://render.githubusercontent.com/render/math?math=v=(v_j)_{j=1}^N"></a> as

<p align="center">
  <img width="300" height="50" src="https://github.com/FloList/EMPL/blob/master/EMPL.png">
</p>

where <a href="#"><img src="https://render.githubusercontent.com/render/math?math=U_j=\sum_{r=1}^j u_j"></a> and <a href="#"><img src="https://render.githubusercontent.com/render/math?math=V_j=\sum_{r=1}^j v_j"></a> are the cumulative histograms. Here, <a href="#"><img src="https://render.githubusercontent.com/render/math?math=\tau \in [0, 1]"></a> is the quantile level of interest. For the particular case of the median (<a href="#"><img src="https://render.githubusercontent.com/render/math?math=\tau = 0.5"></a>), the EMPL reduces to the *Earth Mover's Distance* (or 1-Wasserstein distance) between two 1D histograms (e.g., [Ramdas, Trillos & Cuturi 2017](https://www.mdpi.com/1099-4300/19/2/47)). Therefore, the EMPL is an asymmetric generalisation of the Earth Mover's Distance that enables the regression of *arbitrary quantiles* of the cumulative histogram in each bin (conditional on some input) by harnessing the idea of the *pinball loss* (e.g., [Koenker & Bassett 1978](https://www.jstor.org/stable/1913643)).

<p align="center">
  <img width="600" height="526" src="https://github.com/FloList/EMPL/blob/master/comic.png">
</p>

*Author*: Florian List (Sydney Institute for Astronomy, School of Physics, A28, The University of Sydney, NSW 2006, Australia).

For any queries, please contact me at florian dot list at sydney dot edu dot au.

# Overview
<b>Toy example</b> (histograms generated by drawing numbered balls from an urn)
  - ```Toy_example.py```: trains / loads the neural network for the toy example and generates the plots in the manuscript.
  - ```Simulate_urn_draws.py```: simulates drawing from the urn and compares the numerical results with the analytical solution.
  - ```EMD_for_single_draw.py```: computes the expected EMD between the median / mean and the outcome for a single draw.
<p align="center"><img width="490" height="600" src="https://github.com/FloList/EMPL/blob/master/toy_example.png"></p>

<b>Bimodal example</b> (distribution of cumulative histograms in each bin is bimodal)
  - ```Bimodal_example.py```: trains / loads the neural network for the bimodal example.
  - ```Bimodal_example_make_plots.py```: generates the plots for the bimodal example.
<p align="center"><img width="443" height="600" hspace="20" src="https://github.com/FloList/EMPL/blob/master/bimodal_example.png"></p>
    
<b>Bundesliga example</b> (histograms of the league table position after every week)
  - ```Bundesliga_example.py```: trains / loads the neural network for the Bundesliga example and generates the plots.
  - ```make_bundesliga_table.py```: generates the training and testing datasets from the match results in ```Bundesliga_Results.csv```.
  - ```make_leave_one_out_hists.py```: computes the bootstrapping uncertainties by "replaying" the seasons.

**NOTE**: The file ```Bundesliga_Results.csv``` needs to be downloaded from [Kaggle](https://www.kaggle.com/thefc17/bundesliga-results-19932018) (contains Bundesliga results from 1993/94 to 2017/18).
<p align="center"><img width="320" height="600" src="https://github.com/FloList/EMPL/blob/master/bundesliga_example.png" hspace="60"><p/>

<b>Astrophysical example</b> (estimating brightness histograms from γ-ray photon-count maps)

The code for the astrophysical example will shortly be added to [this](https://github.com/FloList/GCE_NN) repository.
<p align="center"><img width="536" height="600" hspace="10" src="https://github.com/FloList/EMPL/blob/master/astrophysics_example.png"><p/>


# Citation
If you find this code or the paper useful, please consider citing 
> ADD CITATION.
