# pix2pix-GAN-Project-Revisited
Revisiting the Pix2Pix GAN I built for a Stats course, STAT 571 Statistical Methods I, for graduate credit. In this repo, we will use the old GAN as the scaffolding for building a new Pix2Pix GAN which will handle 1024x1024x3 images.

In it you will find 2 folders, "new" and "original", both containing 7 files, "run_training.slurm," "dataset.py," "config.py," "train.py," "utils.py," "discriminator_model.py," and "generator_model.py." The code in the folder "new" is a modified version of everything in the folder "original" which I can 
not claim entirely as my own. Much of the code in the folder "original" was initially built by Aladin Persson and later modified to meet my needs.  You can find Aladin Persson's original code at:
  https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/Pix2Pix

# Purpose
The original purpose of this project was to demonstrate an ability to apply the topics of the course while learning and applying new material
to achieve some goal. In particular, we set out to learn the mathematics behind GANs and implement a demonstration of a GAN.

My reason for revisiting this project is to modify it to work on images of size 1024x1024.

# To run:
- You will need to add the folders "data/" and "evaluation/" to "/pix2pix_Revisited/."
- You will also need to add "training/" and "val/" to "/pix2pix_Revisited/data/."
- You will need to populate the folder "/pix2pix_Revisited/data/training/" with training data
- You will need to add validation data the folder "/pix2pix_Revisited/data/val/" folder
- You will need to train the model by running "train.py" in "/pix2pix_Revisited/"
- Once trained, to deploy the model, run "deploy.py" in "/pix2pix_Revisited/"

# Notes:
- In our application of the code, we used the universities'(Wichita State University) HPC system. So, unless you are using the 
same system, you can ignore(or modify to fit your needs) the file "run_training.slurm" in "/pix2pix_Revisited/."
- Modification to some of the files may be necessary depending on where I am in sterilizing the file when you download 
the repo.
- For this project, we used the WikiArt data set which can be found here:
https://archive.org/details/wikiart-dataset

# Future Plans
- Add a post to my personal website(tuckersideas.net) about this repo
- Train the final working version for an art project
