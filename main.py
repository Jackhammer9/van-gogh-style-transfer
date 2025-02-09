import tensorflow as tf
import numpy as np
from src.model import build_vgg_model, extract_features
from src.loss_functions import compute_content_loss, compute_style_loss
from src.data_loader import load_image, show_image

# Training hyperparameters
CONTENT_WEIGHT = 1e4  # Content loss weight
STYLE_WEIGHT = 1e-2   # Style loss weight
LEARNING_RATE = 5.0
EPOCHS = 1000  # Training iterations

def compute_total_loss(content_features, style_features, generated_features):
    """Compute total loss (content + style) with weighted sum."""
    generated_style_features, generated_content_feature = extract_features(generated_features, vgg_model)

    # Ensure correct feature indexing for content loss
    content_loss = compute_content_loss(content_features, generated_content_feature)

    # Compute style loss separately
    style_loss = compute_style_loss(style_features, generated_style_features)

    total_loss = (CONTENT_WEIGHT * content_loss) + (STYLE_WEIGHT * style_loss)
    return total_loss




def train_style_transfer(content_path, style_path):
    """Optimizes the generated image to match the style."""
    
    # Load images
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    # Initialize VGG19 model
    global vgg_model
    vgg_model = build_vgg_model()

    # Extract content and style features
    style_features, content_features = extract_features(style_image, vgg_model), extract_features(content_image, vgg_model)

    # Initialize generated image as trainable variable
    generated_image = tf.Variable(content_image, dtype=tf.float32)

    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        with tf.GradientTape() as tape:

            loss = compute_total_loss(content_features, style_features, generated_image)

        # Compute gradients & update generated image
        grads = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])

        # Clip pixel values to valid range [0,1]
        generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
    
    return generated_image

# Train & save the output
generated = train_style_transfer("data/content/C3.jpg", "data/style/VG8.jpg")
show_image(generated, "Generated Image")
