# XGB-DIM-for-RSVP
An ensemble learning method for EEG classification in RSVP tasks
Please find the example EEG in master:https://github.com/bowenliee/XGB-DIM-for-RSVP/tree/master
'sub n_k_data.mat' means the k-th block of the n-th subject 

Journal of Neural Engineering
PAPER
Assembling global and local spatial-temporal filters to extract discriminant information of EEG in RSVP task
https://iopscience.iop.org/article/10.1088/1741-2552/acb96f

# Updated 2023-02-28
The GPU-version based on Torch has been upload. The implementation of the optimizer is slightly different from the CPU version, thus some parameters should be changed. At present, the performance of the GPU version is not as good as that of the CPU version. We will continue to optimize it. The GPU version has reduced more than 90% of the training time, so multiple-XGB-DIM models will be introduced later to further improve performance.

# Updated 2023-03-01
I found 2 bugs in GPU version and solved it. And the loss function has been re-defined to replace nn.crossentropy. 
Now, the performance of GPU version is very close to the CPU one. 
The GPU version is really faster, so a new version of mutilple-XGB-DIM is coming soon.
