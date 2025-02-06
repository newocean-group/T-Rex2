# [T-Rex2: Towards Generic Object Detection](https://github.com/newocean-group/T-Rex2)

This repository contains an implementation of [T-Rex2](https://arxiv.org/pdf/2403.14610) Currently, only the visual encoder has been implemented.

### 📖 Model Architecture:
<img src="assets\paper\model_architecture.jpg" alt="model architecture" width=100%>

### 📃Datasets are used for training the model:
- **Object365**
- **OpenImageV7**
- **CrowdHuman**
- **Hiertext**
- **coco2017**

### 🖼️ Visual Results: 
<img src="assets\result\results.gif" width="100%">
**Note** : This model has been trained for approximately 2.7M steps, whereas in the paper, the authors trained the model for about 60M steps:
- ~24M steps for text encoder + image encoder + box decoder
- ~36M steps for visual encoder + text encoder + box decoder + image encoder

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

**Note**: You may need to adjust the threshold value to achieve the best results.

### 💡 Conclusion
This model has been implemented based on my current knowledge and can be further improved with future research.

Additionally, the model can be modified for instance segmentation based on the approach described in this [paper](https://arxiv.org/pdf/2411.08569). The modified model architecture would resemble the following:

<img src="assets\result\model_architecture.jpg" alt="model architecture" width=100%>