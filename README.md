# MoMask: Generative Masked Modeling of 3D Human Motions
## [[Project Page]](https://ericguo5513.github.io/momask) [[Paper]](https://arxiv.org/abs/2312.00063)
![teaser_image](https://ericguo5513.github.io/momask/static/images/teaser.png)

## :postbox: News
📢 **2023-12-15** --- Release code for text2motion generation.  

📢 **2023-11-29** --- Initialized the webpage and git project.  


## :round_pushpin: Get You Ready

<details>
  
### Conda Environment
```
conda env create -f environment.yml
conda activate momask
pip install git+https://github.com/openai/CLIP.git
```
We test our code on Python 3.7.13 and PyTorch 1.7.1


### Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.sh
```

#### Download Evaluation Models and Gloves
For evaluation only.
```
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```

#### (Optional) Download Mannually
Visit [[Google Drive]](https://drive.google.com/drive/folders/1b3GnAbERH8jAoO5mdWgZhyxHB73n23sK?usp=drive_link) to download the models and evaluators mannually.

### Get Data


</details>

### To be continued.