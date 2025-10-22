# Optical Generative Models

**Shiqi Chen, Yuhang Li, Yuntian Wang, Hanlong Chen, Aydogan Ozcan**  
📄 [**Paper** (Nature)](https://www.nature.com/articles/s41586-025-09446-5) | 🌐 [**Project Website (coming soon)**] 📦[**Pretrained Weights**](https://drive.google.com/drive/folders/1aFix08uZGfrds9fVx6MUr54MV1YkbQW4?usp=sharing)
<!-- | 📖 [**Tutorial**](https://sizhe-li.github.io/blog/2025/jacobian-fields-tutorial/) | 🎥 [**Explainer**](https://youtu.be/dFZ1RvJMN7A) | -->

> Optical generative models project the imagination of AI into the realm of analog.

<p align="center">
  <img src="assets\WM  PS-5 mod.png" alt="Image 1" width="400"/>
  <img src="assets\Poster wUCLA.jpeg" alt="Image 2" width="400"/>
</p>

---

## 📢  Announcements

- **[2025-08-29]** Code is updated, and the [**Pretrained Weights**](https://drive.google.com/drive/folders/1aFix08uZGfrds9fVx6MUr54MV1YkbQW4?usp=sharing) are released.
- **[2025-08-27]** Our paper is now published in [**Nature**](https://www.nature.com/articles/s41586-025-09446-5).

---

## 🚀 Quickstart

We provide the software implementations of:
- 🧪 Snapshot optical generative model (training and test implementation)
- ✋ Iterative optical generative model (training and test implementation, please carefully play with the noise scheduler parameters for your own data distribution)

### 📦 Installation

#### 1. Create Conda Environment

```bash
conda create --name optical-generative-models python=3.11
conda activate optical-generative-models
```

#### 2. Install Dependencies (CUDA 11.8)

```
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers accelerate transformers datasets numpy safetensors
```

## ▶️ Running the Code
### 📥 Download Pretrained Checkpoints

Download from [Google Drive](https://drive.google.com/drive/folders/1aFix08uZGfrds9fVx6MUr54MV1YkbQW4?usp=sharing) and place them under the **ckpt** folder.

## ✔️ Ready-to-Run Demos
```
python test_example_mnist.py
python test_example_celeba.py
```

## 🧪 Simulated Experiments
<p align="center">
  <img src="assets\scalability.png" alt="Image 1" width="800"/>
</p>

We conduct a lot of simulations to comprehensively evaluate the scalability of optical generative models, please check our [**paper**](https://www.nature.com/articles/s41586-025-09446-5) and [**supplementary information**](https://www.nature.com/articles/s41586-025-09446-5#Sec22) for details.

## 🦾 Real-World Experiments

### There are some notes for the experimental realization:
- Using high quality laser (preferring filtered by pinehole <50 $\mu m$)
- For the expander of laser, use commercial composite lens to prevent off-axis aberrations in large field-of-view optical generation
- The diffractive elements need perfectly aligned with multi-axes translation stages
- Distance between the components needs precisely calibration. Calibrate the propagation matrix with a camera if needed. For the reflective SLM, please change the distance into its negative after every odd-numbered reflection 

## 🏋️‍♀️ Training

### A. Train Teacher Diffusion Model

```
bash teacher_train.sh
```
### B. Train Snapshot Optical Generative Model

#### Change the path and the configuration in .sh files accordingly
```
bash snapshot_train.sh
bash multicolor_train.sh
```
### C. Train Iterative Optical Generative Model

#### Please carefully play with the noise scheduler parameters for your own data distribution to prevent training failer.
```
bash iterative_train.sh
```

## 🍻 Thanks for your attention, hope you can enjoy!

## 📚 Citation
If you find our work useful, please consider citing us:

```
@article{ChenNature2025,
  author    = {Chen, Shiqi and Li, Yuhang and Wang, Yuntian and Chen, Hanlong and Ozcan, Aydogan},
  title     = {Optical generative models},
  journal   = {Nature},
  year      = {2025},
  month     = aug,
  volume    = {644},
  pages     = {903--911},
  publisher = {Springer Nature},
  doi       = {10.1038/s41586-025-09446-5},
  url       = {https://www.nature.com/articles/s41586-025-09446-5},
  issn      = {0028-0836}
}
```
