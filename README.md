# [T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy](https://github.com/newocean-group/T-Rex2)
<a href="https://colab.research.google.com/drive/1bi7ITH8fmSR6_aleA7M698HN45ea-4n9#scrollTo=eE9AEWpMu-rA"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

This repository contains an unofficial implementation of [T-Rex2](https://arxiv.org/pdf/2403.14610). Currently, only the visual encoder has been implemented.

Deepwiki docs: https://deepwiki.com/newocean-group/T-Rex2.
### 📖 Model Architecture:
<img src="assets\paper\model_architecture.jpg" alt="model architecture" width=100%>

### 📃Datasets are used for training the model:
- **Object365**
- **OpenImagesV7**
- **CrowdHuman**
- **Hiertext**
- **LVIS**

### <img src="assets\result\process_icon.png" width="16" height="16"> To train the model without text prompts and with a batch size of 1 due to hardware limitations. I use the following training process:
<img src='assets\result\training_process.png' width='100%'> 

### 🖼️ Visual Results: 
<img src="assets\result\results.gif" width="100%">

**Note** : This model has been trained for approximately 2.7M steps (batch size = 1) and is still in the training process.

### ⚙️ Installation
To use the model, follow these steps:
1. Clone the repository:

    ```bash
    git clone https://github.com/newocean-group/T-Rex2.git
    ```

2. Download and install CUDA toolkit:
    ```
    # Make sure you have the correct version installed. For example, I installed CUDA 11.8
    ```
3. Compiling CUDA operators:

    ```bash
    cd ops
    python setup.py install
    ```

4. Install other dependencies:

    ```bash
    pip install -r requirements.txt
    ```
5. Log in to your HuggingFace account on your device to automatically download the model weights using the following command:

    ```bash
    huggingface-cli login
    Enter your token
    ```
### 🔍 Demo
I have attached a .ipynb [file](demo.ipynb) in the repository. You can refer to it to know how to use the model.

Additionally, I have provided another .ipynb [file](cls_embeddings.ipynb) that illustrates the process of learning class embeddings for the model.

**Note**: You may need to adjust the threshold value to achieve the best results.

### 💡 Conclusion
This model has been implemented based on my current knowledge and can be further improved with future research.

Additionally, the model can be modified for instance segmentation based on the approach described in this [paper](https://arxiv.org/pdf/2411.08569). The modified model architecture would resemble the following:

<img src="assets\result\model_architecture.jpg" alt="model architecture" width=100%>

### References
- [DETR](https://github.com/facebookresearch/detr)
- [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [DN-DETR](https://github.com/IDEA-Research/DN-DETR)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
