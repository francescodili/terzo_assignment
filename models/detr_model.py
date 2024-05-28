import torch

def load_detr_model():
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model.eval()  # Setta il modello in modalità di valutazione
    return model
