cd checkpoints

cd t2m 
echo -e "Downloading evaluation models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/19C_eiEr0kMGlYVJy_yFL6_Dhk3RvmwhM/view?usp=sharing
echo -e "Unzipping humanml3d_evaluator.zip"
unzip humanml3d_evaluator.zip

echo -e "Clearning humanml3d_evaluator.zip"
rm humanml3d_evaluator.zip

cd ../kit/
echo -e "Downloading pretrained models for KIT-ML dataset"
gdown --fuzzy https://drive.google.com/file/d/1TKIZ3TSSZawpilC-7Kw7Ws4sNNuzb49p/view?usp=drive_link

echo -e "Unzipping kit_evaluator.zip"
unzip kit_evaluator.zip

echo -e "Clearning kit_evaluator.zip"
rm kit_evaluator.zip

cd ../../

echo -e "Downloading done!"
