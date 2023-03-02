# XGB-DIM-for-RSVP
An ensemble learning method for EEG classification in RSVP tasks

## XGB_DIM
A CPU version, i.e., the version in paper 'Assembling global and local spatial-temporal filters to extract discriminant information of EEG in RSVP task'.
Journal of Neural Engineering, https://iopscience.iop.org/article/10.1088/1741-2552/acb96f
In this version, you can find the whole details of parameter optimization, including the extreme gradient boosting, gradient calculation and the specific implementation of Adam. You can compare them with the derivation in the paper.

## XGB_DIM_GPU_v2
A GPU version based on Torch lib.
Almost all the 'for' cycles have been replaced by tensor calculation, which greatly improves the speed. The models and their losses are clearly defined in each class. Because the optimizer of torch is used, only the details of extreme gradient boosting are retained. 
The code in this version is more concise, and it is easy to adjust the internal structure or use it to generate improved versions.

## multi_XGB_DIM_GPU_v1
This version is based on XGB_DIM_GPU_v2 and makes full use of negative samples. Its performance is better than the ordinary GPU version. If the random sampling mode is used, the stability is also improved.

## Example Data (Subject 64)
Please find the example EEG in master:https://github.com/bowenliee/XGB-DIM-for-RSVP/tree/master
'sub n_k_data.mat' means the k-th block of the n-th subject 

# Updated Log
## Updated 2023-02-28
The GPU-version based on Torch has been upload. The implementation of the optimizer is slightly different from the CPU version, thus some parameters should be changed. At present, the performance of the GPU version is not as good as that of the CPU version. We will continue to optimize it. The GPU version has reduced more than 90% of the training time, so multiple-XGB-DIM models will be introduced later to further improve performance.

## Updated 2023-03-01
I found 2 bugs in GPU version and solved it. And the loss function has been re-defined to replace nn.crossentropy. 
Now, the performance of GPU version is very close to the CPU one. 
The GPU version is really faster, so a new version of mutilple-XGB-DIM is coming soon.

## Updated 2023-03-02
### multiple-XGB-DIM upload
The multiple-XGB-DIM has been uploaded. This version makes full use of negative samples. Negative samples are randomly or evenly divided into multiple groups, and the number of negative samples in each group is the same as the total number of positive samples. Each group combined with the positive sample group to train the XGB-DIM model. Finally, the models are bagged. 
This version requires a big GPU ( > 8GB). And I will continue to optimize it.
### XGB-DIM-GPU version debug
I found a bug that consumes a lot of GPU memory and fixed it.
