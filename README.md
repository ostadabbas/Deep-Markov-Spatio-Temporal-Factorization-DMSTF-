# Deep Markov Spatio-Temporal Factorization (DMSTF)

Codes and experiments for the following paper: 

A. Farnoosh, B. Rezaei, E. Sennesh, Z. Khan1, J. Dy, A. Satpute, J.
Hutchinson, J. van de Meent, and S, Ostadabbas, "Deep Markov Spatio-Temporal Factorization."

Contact: 
[Amirreza Farnoosh](afarnoosh@ece.neu.edu),

[Sarah Ostadabbas](ostadabbas@ece.neu.edu)


### Dependencies: 
Numpy, Scipy, Pytorch, Nibabel, Tqdm, Matplotlib, Sklearn, Json, Pandas  

## Autism Dataset:

Run the following snippet to restore results from pre-trained checkpoints for Autism dataset in `./fMRI_results` folder. A few instances from each dataset are included to help the code run without errors. 
You may replace `{site}` with `Caltec`, `Leuven`, `MaxMun`, `NYU_00`, `SBL_00`, `Stanfo`, `Yale_0`, `USM_00`, `DSU_0`, `UM_1_0`, or set `-exp autism` for the full dataset. Here, checkpoint files for `Caltec`, `SBL_00`, `Stanfo` are only included due to storage limitations.

`python dmfa_fMRI.py -t 75 -exp autism_{site} -dir ./data_autism/ -smod ./ckpt_fMRI/ -dpath ./fMRI_results/ -restore`

or run the following snippet for training with batch size of 10 (full dataset needs to be downloaded and preprocessed/formatted beforehand):

`python dmfa_fMRI.py -t 75 -exp autism_{site} -dir ./data_autism/ -smod ./ckpt_fMRI/ -dpath ./fMRI_results/ -bs 10`

After downloading the full Autism dataset, run the following snippet to preprocess/format data:

`python generate_fMRI_patches.py -T 75 -dir ./path_to_data/ -ext /*.gz -spath ./data_autism/`

## Depression Dataset:

Run the following snippet to restore results from pre-trained checkpoints for Depression dataset in `./fMRI_results` folder. A few instances from the dataset are included to help the code run without errors.
You may replace `{ID}` with `1`, `2`, `3`, `4`, `5`. ID `4` corresponds to the first experiment on Depression dataset in the paper. IDs `2`, `3` correspond to the second experiment on Depression dataset in the paper. For ID `5` you need access to a larger part of dataset (which is not included here).

`python dmfa_fMRI.py -exp depression_{ID} -dir ./data_depression/ -smod ./ckpt_fMRI/ -dpath ./fMRI_results/ -restore`

or run the following snippet for training with batch size of 10 (full dataset needs to be downloaded and preprocessed/formatted beforehand):

`python dmfa_fMRI.py -exp depression_{ID} -dir ./data_depression/ -smod ./ckpt_fMRI/ -dpath ./fMRI_results/ -bs 10`

After downloading the full Depression dataset, run the following snippet to preprocess/format data:

`python generate_fMRI_patches_depression.py -T 6 -dir ./path_to_data/ -spath ./data_depression/`

If you have the full Depression dataset downloaded, for ID `5` run the following snippet to restore results:

`python dmfa_fMRI.py -exp depression_5 -dir ./data_depression/ -smod ./ckpt_fMRI/ -dpath ./fMRI_results/ -restore -c 1 -du 1 -predict`

## Traffic datasets:

Run the following snippets to restore results from pre-trained checkpoints for traffic dataset. 
Here, data files for Hangzhou, Birmingham, Guangzhou are included due to storage limitations.

### Birmingham:
`python dmfa_traffic.py -t 18 -k 10 -file ./data_traffic/tensor_b.mat -smod ./ckpt_traffic/b/ -dpath ./Birmingham_results/ -restore -ID birmingham -days 7`

### Hangzhou:
`python dmfa_traffic.py -t 108 -k 10 -file ./data_traffic/tensor_h.mat -smod ./ckpt_traffic/h/ -dpath ./Hangzhou_results/ -restore -ID hangzhou -days 5`

### Seattle:
`dmfa_traffic.py -t 288 -k 30 -file ./data_traffic/tensor_s.txt -smod ./ckpt_traffic/s/ -dpath ./Seattle_results/ -restore -ID seattle -days 5`

### Guangzhou:
`python dmfa_traffic.py -t 144 -k 30 -file ./data_traffic/tensor_g.mat -smod ./ckpt_traffic/g/ -dpath ./Guangzhou_results/ -restore -ID guangzhou -days 5`

or run the following snippets for training, and then predicting (e.g., Hangzhou, Birmingham datasets):

### Hangzhou:
`python dmfa_traffic.py -t 108 -k 10 -epoch 500 -bs 25 -file ./data_traffic/tensor_h.mat -smod ./Hangzhou_results/ -dpath ./Hangzhou_results/ -days 5`
`python dmfa_traffic.py -t 108 -k 10 -epoch 500 -bs 25 -file ./data_traffic/tensor_b.mat -smod ./Hangzhou_results/ -dpath ./Hangzhou_results/ -days 5 -predict`

### Birmingham:
`python dmfa_traffic.py -t 18 -k 10 -epoch 5000 -bs 30 -file ./data_traffic/tensor_b.mat -smod ./Birmingham_results/ -dpath ./Birmingham_results/ -days 7`
`python dmfa_traffic.py -t 18 -k 10 -epoch 5000 -bs 30 -file ./data_traffic/tensor_b.mat -smod ./Birmingham_results/ -dpath ./Birmingham_results/ -days 7 -predict`

## Toy example:
Run the following snippet for the toy example.

`python dmfa_toy.py`

## Synthetic fMRI data:
Run the following snippet to restore results from the pre-trained checkpoint for the synthetic experiment in `./synthetic_results` folder (synthetic fMRI data is not included due to storage limitations).

`python dmfa_synthetic.py`

## License 
* This code is for non-commertial purpose only. For other uses please contact Augmented Cognition Lab at Northeastern University: http://www.northeastern.edu/ostadabbas/ 

