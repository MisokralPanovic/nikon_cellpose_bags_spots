module load EBModules
module load Python/3.11.5-GCCcore-13.2.0


# load and save module collection
module load EBModules UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1 OpenMPI/4.1.5-GCC-12.3.0 FLTK/1.3.8-GCCcore-12.3.0

module save cuda-fltk
module restore cuda-fltk

# delete collection
cd ~/.lmod.d
rm cuda-fltk


# load anaconda
module load Anaconda3
conda create --name test_env python=3.11.13

conda env create -f environment.yml 

conda init bash  # Only needs to be run once. If not using "bash", replace with your shell.
source ~/.bashrc # Only needs to be run once
conda activate cellpose-bags

# rclone data from Dropbox
module load rclone

rclone listremotes                  # List configured endpoints
rclone lsd dropbox_wigler:                 # List directories in Dropbox
rclone ls dropbox_wigler:                  # List all files in Dropbox  
rclone copy file.txt dropbox_wigler:test   # Copy file to Dropbox test folder 
rclone copy dropbox_wigler:folder/file.txt # Copy file from some Dropbox folder