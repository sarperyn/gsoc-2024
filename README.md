# A graphical user interface of ML Toolbox for Medical Images (gsoc2024)

This github repo has been developed under GSoC 2024

Contributor: Sarper Yurtseven <br>
Organization: Department of Biomedical Informatics, Emory University <br>
Mentors: Ozgur Kara, Babak Mahmoudi <br>

Accurate and efficient image segmentation is crucial for various medical applications, particularly when real-time processing is required. Traditional methods often face challenges in maintaining computational efficiency, leading to delays that are not acceptable in time-sensitive scenarios. This project introduces an initial version of a user-friendly Graphical User Interface (GUI) built upon the ML Methods Toolbox, specifically designed to address these challenges in medical image segmentation.

This version serves as a functional demo, it lays the foundation for future upgrades that will introduce more advanced features and capabilities. The GUI currently offers three main functionalities:

- [Low-latency model creation]: The toolbox facilitates the development of efficient, low-latency models capable of performing real-time image segmentation. This is particularly important for scenarios where minimal delay is essential for timely medical decisions.

- [Synthetic medical image generation]: The interface includes tools to generate synthetic medical images, which can be used to augment real-world datasets. These synthetic images enhance the training and performance of segmentation models, especially when classifying specific anatomical structures.
Comprehensive platform for medical professionals: By combining real-time segmentation and synthetic image generation, the ML Methods Toolbox GUI provides a robust platform that empowers medical professionals and researchers. It enables faster analysis, streamlines workflows, and ultimately contributes to improved outcomes in various medical applications.


## Table of Contents

- [Requirements](#requirements)
- [External Modules](#external-modules)
- [Usage](#usage)
- [Demo Video](#demo-video)


## Requirements

Before you begin, ensure you have met the following requirements:

- **Python**: Version 3.12.2

```
conda create -n new_env python==3.12.2
conda activate new_env
```

Check your your pip is the correct pip

```
which pip
```

Then install requirements.txt

```
pip install -r requirements.txt
```

## External Modules

This project relies on several external modules that must be installed:

- [ddim](https://github.com/sarperyn/ddim.git) - To train and sample from diffusion model
- [MedSAM](https://github.com/bowang-lab/MedSAM.git) - To use MedSAM model. Also, you should download the checkpoints using the repo

Install these modules and place them in the 'external' (which you created) folder.

## Model Checkpoints

You can download the models that is used in the repo from this link

## Usage
Place the models to correct places and run:

```
python gradio_demo.py
```

## Demo Video
![Watch the demo](https://github.com/sarperyn/gsoc-2024/blob/main/gifs/demo1.gif)

![Data generation results](https://github.com/sarperyn/gsoc-2024/blob/main/gifs/demo2.gif)