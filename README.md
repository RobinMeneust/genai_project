# genai_project


Vous trouverez dans ce repository :

- Une [implémentation](notebook/imagic.ipynb) du modèle Imagic (décrit dans le [papier scientifique](Kawar_Imagic_Text-Based_Real_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.pdf)) utilisant Stable Diffusion pour l'édition d'images.
- Une [implémentation](notebook/Test_LoRA_implementation_Ethan.ipynb) de LoRA pour le fine-tuning.



## Proposal

[Proposal](proposal.md)


## Rapport

[Rapport](Rapport%20Gen-AI%20Meneust,%20Cabrit,%20Charlet,%20Pinto.pdf)


## Install

### Requirements

#### Without uv

`pip install -r requirements.txt` and then `pip install -e .`

#### With uv

- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- `uv venv --python 3.12.0`
- Then activate it using the command displayed
- `uv pip install -r requirements.txt`
- `uv pip install -e .`

## Authors

- Maxime Cabrit
- Antoine Charlet
- Robin Meneust
- Ethan Pinto
