import tensorflow as tf

def compute_content_loss(content_features, generated_features):
    """Computes content loss as Mean Squared Error (MSE)."""
    return tf.reduce_mean(tf.square(content_features - generated_features))

def gram_matrix(feature_map):
    """Computes the Gram matrix of a feature map."""
    # Reshape feature map: (batch, height, width, channels) → (height*width, channels)
    shape = tf.shape(feature_map)
    num_channels = shape[-1]
    flattened = tf.reshape(feature_map, [-1, num_channels])

    # Compute Gram matrix (dot product of feature vectors)
    gram = tf.matmul(flattened, flattened, transpose_a=True)
    return gram / tf.cast(tf.size(flattened), tf.float32)

def compute_style_loss(style_features, generated_features):
    """Computes style loss using Gram matrices."""
    total_style_loss = 0.0
    for style_f, gen_f in zip(style_features, generated_features):
        gram_style = gram_matrix(style_f)
        gram_gen = gram_matrix(gen_f)
        total_style_loss += tf.reduce_mean(tf.square(gram_style - gram_gen))
    
    return total_style_loss / len(style_features)
