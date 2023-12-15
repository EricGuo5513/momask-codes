rm -rf checkpoints
mkdir checkpoints
cd checkpoints
mkdir t2m
mkdir kit 

cd t2m 
echo -e "Downloading pretrained models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1dtKP2xBk-UjG9o16MVfBJDmGNSI56Dch/view?usp=sharing

echo -e "Unzipping humanml3d_models.zip"
unzip humanml3d_models.zip

echo -e "Clearning humanml3d_models.zip"
rm humanml3d_models.zip

cd ../kit 
echo -e "Downloading pretrained models for KIT-ML dataset"
gdown --fuzzy https://drive.google.com/file/d/1MNMdUdn5QoO8UW1iwTcZ0QNaLSH4A6G9/view?usp=sharing

echo -e "Unzipping humanml3d_models.zip"
unzip kit_models.zip

echo -e "Clearning humanml3d_models.zip"
rm kit_models.zip

cd ../../

echo -e "Downloading done!"