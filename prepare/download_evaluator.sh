rm -rf checkpoints
mkdir checkpoints
cd checkpoints 

cd t2m 
echo -e "Downloading evaluation models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1oLhSH7zTlYkQdUWPv3-v4opigB7pXkFk/view?usp=sharing
echo -e "Unzipping humanml3d_models.zip"
unzip humanml3d_evaluator.zip

echo -e "Clearning humanml3d_models.zip"
rm humanml3d_evaluator.zip

cd ../kit 
echo -e "Downloading pretrained models for KIT-ML dataset"
gdown --fuzzy https://drive.google.com/file/d/115n1ijntyKDDIZZEuA_aBgffyplNE5az/view?usp=sharing

echo -e "Unzipping humanml3d_models.zip"
unzip kit_evaluator.zip

echo -e "Clearning humanml3d_models.zip"
rm kit_evaluator.zip

cd ../../

echo -e "Downloading done!"