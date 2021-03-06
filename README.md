# Adversarial Bayesian Optimization for Quantifying Motion Artifact within MRI  


The paper (A. Butskova, R. Juhl, D. Zukic, A. Chaudhary, K.M. Pohl, **Q. Zhao**, Adversarial Bayesian Optimization for Quantifying Motion Artifact within MRI, 4th Workshop on PRedictive Intelligence in MEdicine (PRIME), 2021) has been accepted to the PRIME-MICCAI 2021 workshop.

The paper aims at quantifying motion artifact in the 3D T1 MRI images.

Step 1: We use motion simulation package specified below to imply the motion parameters of
the images labeled as bad, by comparing them to the simulated images with  
the motion parameters of 3D translation and rotation, with the help of Bayesian optimization.
(https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomMotion)
We build a classification neural network simulating bad images from the good ones and further
differentiating between the simulated bad and the real bad images. The discriminator loss is maximized 
with the Bayesian optimization search and the Upper Confidence Bound acquisition function, 
with respect to the simulated motion parameters: 3D translations and rotations.

Step 2: We further use the obtained parameters to create a simulated motion train set and build the 
regression based CNN model with additional ranking loss, to quantify the motion artifact.

The dataset includes 831 3D T1 MRI images labelled as good (no motion artifact) and 159 3D T1 MRI images 
labelled as bad (motion artifact). 


![model](https://user-images.githubusercontent.com/44216377/120604160-9a706280-c401-11eb-9aab-0f6e593fce61.png)

