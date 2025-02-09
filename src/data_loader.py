import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path, target_size=(512, 512)):
    """Load an image (JPEG, PNG, or WebP), resize it, and convert to tensor format."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)  # Supports WebP, JPEG, PNG
    image = tf.image.resize(image, target_size)
    image = image / 255.0  # Normalize to [0,1] range
    return tf.expand_dims(image, axis=0)  # Add batch dimension

def show_image(image_tensor, title="Image"):
    """Display an image tensor."""
    image = tf.squeeze(image_tensor, axis=0)  # Remove batch dimension
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()