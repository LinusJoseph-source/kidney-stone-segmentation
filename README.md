# Kidney Stone Segmentation Using Deep Learning

A deep learn
ing-based system for detecting and segmenting kidney stones from CT images.

## Overview

This project uses a ResUNet-based deep learning model to perform binary image segmentation of kidney stones from medical images.
The trained model is integrated into a Streamlit web application where users can upload CT images and receive segmentation results.

## Project Structure

- streamlit_app.py - Streamlit application
- model.py - Model architecture
- notebooks/ - Training notebook
- models/ - Model metadata
- examples/ - Sample outputs

## Model Details

Architecture:
- ResUNet(Resnet50 as encoder and U-Net as decoder with skip connection)
- ImageNet pretrained encoder

Task:
- Binary segmentation

Input size:
- 256 x 256

## Results

Dice Score: 0.855
IoU: 0.771

## Technologies

- Python
- PyTorch
- OpenCV
- Albumentations
- Streamlit

## Running the Application

Install dependencies:
pip install -r requirements.txt

Run:
streamlit run streamlit_app.py

## Model Download

The trained model weights are available in the GitHub Releases section.
Download:
"final_model.pth"
and place it in the models folder.
