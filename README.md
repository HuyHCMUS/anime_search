# Anime Search Engine

## Description

Search anime characters based on uploaded images. 
- Input: Image
- Output: Related anime with input image

## Pipeline
![Pipeline Diagram](pipeline.png)  
## Example
![Example](example.png)  

## Data Collection

Crawl on [Myanimelist](https://myanimelist.net/)

Collect the information, character's image of 8000 most popular anime 

Data: [Download](https://huggingface.co/datasets/huyhamhoc/popular_anime_character)

## Detect anime character
The preprocessing stage employs [YOLOv11](https://github.com/ultralytics/ultralytics) for detect anime character's face.

Data to train model: [Download](https://huggingface.co/datasets/deepghs/anime_head_detection)

## Feature Extraction
Facial feature embeddings are extracted using various CNN models. The models explored include:
- **Vision Transformer (ViT)**: using the [timm library](https://github.com/rwightman/pytorch-image-models), fine-tuned on my dataset.
- **FaceNet**: Utilized through the [facenet-pytorch](https://github.com/timesler/facenet-pytorch) library.
- **Fine-tuned FaceNet**: Using pre-trained FaceNet model, fine-tuned on my dataset.


## Similarity Search

The project employs [Faiss](https://github.com/facebookresearch/faiss) for efficient indexing and searching of facial embeddings. Faiss allows for fast retrieval of similar embeddings from a large dataset.

## User Interface
User Interface: [Streamlit](https://streamlit.io/), enabling users to interactively upload images and view search results.
