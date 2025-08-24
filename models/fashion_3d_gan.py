#!/usr/bin/env python3
"""
Architecture GAN pour génération 2D-to-3D Fashion
Optimisée pour RTX 2000 Ada avec mixed precision
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Optional

# Configuration mixed precision pour RTX 2000 Ada
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class AttentionBlock(layers.Layer):
    """Module d'attention pour améliorer la correspondance 2D-3D"""
    
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv_query = layers.Conv2D(filters // 8, 1, activation='relu')
        self.conv_key = layers.Conv2D(filters // 8, 1, activation='relu')
        self.conv_value = layers.Conv2D(filters, 1, activation='relu')
        self.conv_out = layers.Conv2D(filters, 1, activation='relu')
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height, width = tf.shape(inputs)[1], tf.shape(inputs)[2]
        
        # Compute attention maps
        query = self.conv_query(inputs)
        key = self.conv_key(inputs)
        value = self.conv_value(inputs)
        
        # Reshape for attention computation
        query = tf.reshape(query, [batch_size, height * width, -1])
        key = tf.reshape(key, [batch_size, height * width, -1])
        value = tf.reshape(value, [batch_size, height * width, -1])
        
        # Attention weights
        attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True))
        
        # Apply attention
        attended = tf.matmul(attention, value)
        attended = tf.reshape(attended, [batch_size, height, width, self.filters])
        
        # Output projection
        output = self.conv_out(attended)
        return inputs + output  # Residual connection

class Generator3D(Model):
    """
    Générateur 2D-to-3D optimisé pour fashion items
    Architecture: Pretrained Encoder 2D -> Latent Space -> Decoder 3D
    """
    
    def __init__(self, 
                 img_shape: Tuple[int, int, int] = (256, 256, 3),
                 voxel_size: int = 64,
                 latent_dim: int = 512,
                 use_pretrained_encoder: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.img_shape = img_shape
        self.voxel_size = voxel_size
        self.latent_dim = latent_dim
        self.use_pretrained_encoder = use_pretrained_encoder
        
        # =============== ENCODER 2D ===============
        if use_pretrained_encoder:
            # Use pretrained ResNet50 as encoder (frozen)
            self.pretrained_encoder = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=img_shape,
                pooling=None
            )
            
            # Freeze pretrained layers
            self.pretrained_encoder.trainable = False
            
            # Add custom layers on top of pretrained features
            self.custom_conv1 = layers.Conv2D(512, 3, padding='same', activation='relu')
            self.custom_conv2 = layers.Conv2D(256, 3, padding='same', activation='relu')
            
            # Attention block pour capturer les détails importants
            self.attention1 = AttentionBlock(256)
            
            # Global Average Pooling + Dense pour le latent space
            self.global_pool = layers.GlobalAveragePooling2D()
            self.latent_dense = layers.Dense(latent_dim, activation='relu')
            
        else:
            # Original encoder (trainable from scratch)
            self.encoder_conv1 = layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')
            self.encoder_conv2 = layers.Conv2D(128, 4, strides=2, padding='same')
            self.encoder_bn2 = layers.BatchNormalization()
            
            self.encoder_conv3 = layers.Conv2D(256, 4, strides=2, padding='same')
            self.encoder_bn3 = layers.BatchNormalization()
            
            # Attention block pour capturer les détails importants
            self.attention1 = AttentionBlock(256)
            
            self.encoder_conv4 = layers.Conv2D(512, 4, strides=2, padding='same')
            self.encoder_bn4 = layers.BatchNormalization()
            
            self.encoder_conv5 = layers.Conv2D(1024, 4, strides=2, padding='same')
            self.encoder_bn5 = layers.BatchNormalization()
            
            # Global Average Pooling + Dense pour le latent space
            self.global_pool = layers.GlobalAveragePooling2D()
            self.latent_dense = layers.Dense(latent_dim, activation='relu')
        
        # =============== DECODER 3D ===============
        # Commencer par une représentation 3D de base
        initial_3d_size = voxel_size // 8  # 8x8x8 pour voxel_size=64
        self.initial_dense = layers.Dense(
            initial_3d_size * initial_3d_size * initial_3d_size * 512,
            activation='relu'
        )
        self.initial_reshape = layers.Reshape((initial_3d_size, initial_3d_size, initial_3d_size, 512))
        
        # Déconvolutions 3D progressives
        self.deconv3d_1 = layers.Conv3DTranspose(256, 4, strides=2, padding='same')
        self.bn3d_1 = layers.BatchNormalization()
        
        self.deconv3d_2 = layers.Conv3DTranspose(128, 4, strides=2, padding='same')
        self.bn3d_2 = layers.BatchNormalization()
        
        self.deconv3d_3 = layers.Conv3DTranspose(64, 4, strides=2, padding='same')
        self.bn3d_3 = layers.BatchNormalization()
        
        # Couche finale pour générer les voxels
        self.final_conv3d = layers.Conv3D(1, 3, padding='same', activation='sigmoid')
        
        # Dropout pour la régularisation
        self.dropout = layers.Dropout(0.3)
    
    def call(self, inputs, training=None):
        # =============== ENCODER 2D ===============
        if self.use_pretrained_encoder:
            # Use pretrained ResNet50
            x = self.pretrained_encoder(inputs, training=False)  # Always False for pretrained
            
            # Add custom layers
            x = self.custom_conv1(x)
            x = self.custom_conv2(x)
            
            # Attention mechanism
            x = self.attention1(x)
            
            # Latent representation
            x = self.global_pool(x)
            x = self.dropout(x, training=training)
            latent = self.latent_dense(x)
            
        else:
            # Original encoder path
            x = self.encoder_conv1(inputs)
            
            x = self.encoder_conv2(x)
            x = self.encoder_bn2(x, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = self.encoder_conv3(x)
            x = self.encoder_bn3(x, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            # Attention mechanism
            x = self.attention1(x)
            
            x = self.encoder_conv4(x)
            x = self.encoder_bn4(x, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = self.encoder_conv5(x)
            x = self.encoder_bn5(x, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            # Latent representation
            x = self.global_pool(x)
            x = self.dropout(x, training=training)
            latent = self.latent_dense(x)
        
        # =============== DECODER 3D ===============
        # Générer la représentation 3D initiale
        x = self.initial_dense(latent)
        x = tf.nn.relu(x)
        x = self.initial_reshape(x)
        
        # Déconvolutions 3D progressives
        x = self.deconv3d_1(x)
        x = self.bn3d_1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.deconv3d_2(x)
        x = self.bn3d_2(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.deconv3d_3(x)
        x = self.bn3d_3(x, training=training)
        x = tf.nn.relu(x)
        
        # Génération finale des voxels
        voxels = self.final_conv3d(x)
        
        return voxels, latent

class Discriminator3D(Model):
    """
    Discriminateur 3D multi-échelle pour évaluer la qualité des voxels générés
    """
    
    def __init__(self, voxel_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        
        self.voxel_size = voxel_size
        
        # Architecture progressive pour discriminateur 3D
        self.conv3d_1 = layers.Conv3D(32, 4, strides=2, padding='same')
        self.conv3d_2 = layers.Conv3D(64, 4, strides=2, padding='same')
        self.bn_2 = layers.BatchNormalization()
        
        self.conv3d_3 = layers.Conv3D(128, 4, strides=2, padding='same')
        self.bn_3 = layers.BatchNormalization()
        
        self.conv3d_4 = layers.Conv3D(256, 4, strides=2, padding='same')
        self.bn_4 = layers.BatchNormalization()
        
        self.conv3d_5 = layers.Conv3D(512, 4, strides=2, padding='same')
        self.bn_5 = layers.BatchNormalization()
        
        # Classification finale
        self.global_pool = layers.GlobalAveragePooling3D()
        self.dropout = layers.Dropout(0.3)
        self.dense_1 = layers.Dense(128, activation='relu')
        self.dense_2 = layers.Dense(1)  # Real/Fake classification
    
    def call(self, inputs, training=None):
        x = self.conv3d_1(inputs)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv3d_2(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv3d_3(x)
        x = self.bn_3(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv3d_4(x)
        x = self.bn_4(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv3d_5(x)
        x = self.bn_5(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        # Classification finale
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        x = self.dense_1(x)
        output = self.dense_2(x)
        
        return output

class Fashion3DGAN:
    """
    GAN complet pour génération 2D-to-3D Fashion
    Optimisé pour fine-tuning 8h sur RTX 2000 Ada avec pretrained encoder
    """
    
    def __init__(self, 
                 img_shape: Tuple[int, int, int] = (256, 256, 3),
                 voxel_size: int = 64,
                 latent_dim: int = 512,
                 learning_rates: Tuple[float, float] = (0.0002, 0.0001),
                 use_pretrained_encoder: bool = True):
        
        self.img_shape = img_shape
        self.voxel_size = voxel_size
        self.latent_dim = latent_dim
        self.use_pretrained_encoder = use_pretrained_encoder
        
        # Créer les modèles
        self.generator = Generator3D(img_shape, voxel_size, latent_dim, use_pretrained_encoder)
        self.discriminator = Discriminator3D(voxel_size)
        
        # Optimizers avec mixed precision
        self.gen_optimizer = tf.keras.optimizers.Adam(
            learning_rates[0], beta_1=0.5, beta_2=0.999
        )
        self.disc_optimizer = tf.keras.optimizers.Adam(
            learning_rates[1], beta_1=0.5, beta_2=0.999
        )
        
        # Loss functions
        self.adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.reconstruction_loss = tf.keras.losses.BinaryCrossentropy()
        
        # Métriques
        self.gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
        self.disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')
        self.reconstruction_metric = tf.keras.metrics.Mean(name='reconstruction_loss')
        
        print(f"[INFO] Fashion3DGAN initialized with {'pretrained' if use_pretrained_encoder else 'scratch'} encoder")
    
    def unfreeze_encoder(self, learning_rate: float = 0.0001):
        """Unfreeze the pretrained encoder for fine-tuning"""
        if self.use_pretrained_encoder:
            self.generator.pretrained_encoder.trainable = True
            # Use a lower learning rate for pretrained layers
            self.gen_optimizer = tf.keras.optimizers.Adam(
                learning_rate, beta_1=0.5, beta_2=0.999
            )
            print(f"[INFO] Encoder unfrozen for fine-tuning with lr={learning_rate}")
        else:
            print("[WARNING] No pretrained encoder to unfreeze")
    
    def freeze_encoder(self):
        """Freeze the pretrained encoder"""
        if self.use_pretrained_encoder:
            self.generator.pretrained_encoder.trainable = False
            print("[INFO] Encoder frozen")
        else:
            print("[WARNING] No pretrained encoder to freeze")
    
    def generate_3d(self, image_2d: np.ndarray) -> np.ndarray:
        """
        Générer un modèle 3D à partir d'une image 2D
        """
        # Préprocessing si nécessaire
        if image_2d.dtype != np.float32:
            image_2d = (image_2d.astype(np.float32) / 127.5) - 1.0
        
        # Ajouter batch dimension si nécessaire
        if len(image_2d.shape) == 3:
            image_2d = np.expand_dims(image_2d, axis=0)
        
        # Génération
        voxels, _ = self.generator(image_2d, training=False)
        
        # Retourner les voxels (supprimer batch dimension)
        return voxels.numpy()[0, :, :, :, 0]
    
    def save_models(self, checkpoint_dir: str):
        """Sauvegarder les modèles"""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.generator.save_weights(os.path.join(checkpoint_dir, 'generator.weights.h5'))
        self.discriminator.save_weights(os.path.join(checkpoint_dir, 'discriminator.weights.h5'))
        
        print(f"Models saved to {checkpoint_dir}")
    
    def load_models(self, checkpoint_dir: str):
        """Charger les modèles"""
        import os
        
        gen_path = os.path.join(checkpoint_dir, 'generator')
        disc_path = os.path.join(checkpoint_dir, 'discriminator')
        
        if os.path.exists(gen_path + '.index'):
            self.generator.load_weights(gen_path)
            print("Generator loaded")
        
        if os.path.exists(disc_path + '.index'):
            self.discriminator.load_weights(disc_path)
            print("Discriminator loaded")

# Configuration GPU pour RTX 2000 Ada
def setup_gpu():
    """Configuration optimale GPU pour RTX 2000 Ada"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Activer la croissance mémoire
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"[OK] {len(gpus)} GPU(s) detected and configured")
            print(f"[INFO] Mixed precision enabled for RTX 2000 Ada")
            
        except RuntimeError as e:
            print(f"[ERROR] GPU error: {e}")
    else:
        print("[WARNING] No GPU detected, using CPU")

if __name__ == "__main__":
    setup_gpu()
    gan = Fashion3DGAN()
    print("[OK] Fashion3DGAN ready!")
