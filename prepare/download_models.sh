rm -rf checkpoints
mkdir checkpoints
cd checkpoints
mkdir t2m

cd t2m 
echo -e "Downloading pretrained models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing

echo -e "Unzipping humanml3d_models.zip"
unzip humanml3d_models.zip

echo -e "Cleaning humanml3d_models.zip"
rm humanml3d_models.zip

cd ../
mkdir kit
cd kit

echo -e "Downloading pretrained models for KIT-ML dataset"
gdown --fuzzy https://drive.google.com/file/d/1FapdHNkxPouasVM8MWgg1f6sd_4Lua2q/view?usp=sharing

echo -e "Unzipping kit_models.zip"
unzip kit_models.zip

echo -e "Cleaning kit_models.zip"
rm kit_models.zip

cd ../../

echo -e "Downloading done!"
