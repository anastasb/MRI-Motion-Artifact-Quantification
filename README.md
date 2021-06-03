# MRI-Motion-Artifact-Quantification

The project aims at quantifying motion artifact in the 3D T1 MRI images.
We use motion simulation  package to find the motion parameters of
the images labeled as bad, by comparing them to the simulated images with  
the motion parameters of 3D translation and rotation, with the help of Bayesian optimization.
(https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomMotion)

We further use the obtained parameters to create a simulated motion affected
train set and build the regression based CNN model with additional ranking loss,
to quantify the motion effect.

https://user-images.githubusercontent.com/44216377/120604160-9a706280-c401-11eb-9aab-0f6e593fce61.png)
