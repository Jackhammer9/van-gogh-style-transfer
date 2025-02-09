import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

# Layers for feature extraction
CONTENT_LAYERS = ['block5_conv4']
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def build_vgg_model():
    """Load the VGG19 model and return a model that outputs selected feature maps."""
    vgg = VGG19(weights='imagenet', include_top=False)  # Pretrained on ImageNet, no fully connected layers
    vgg.trainable = False  # Freeze model weights

    # Extract layers we need for style and content
    selected_layers = CONTENT_LAYERS + STYLE_LAYERS
    outputs = [vgg.get_layer(layer).output for layer in selected_layers]

    # Create model that returns outputs from selected layers
    model = Model(inputs=vgg.input, outputs=outputs)
    return model

def extract_features(image, model):
    """Extract content and style features separately."""
    outputs = model(image)  # Get all layer outputs

    # Extract style features (list of tensors)
    style_features = outputs[:len(STYLE_LAYERS)]

    # Extract content feature (single tensor)
    content_feature = outputs[len(STYLE_LAYERS)]

    return style_features, content_feature



