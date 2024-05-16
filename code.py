import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select 10 random images from the test set
random_indices = np.random.choice(x_test.shape[0], 10, replace=False)
selected_images = x_test[random_indices]

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, std=25):
    gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# Function to apply a mean filter to an image
def apply_mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

# Add Gaussian noise and apply mean filter to the selected images
noisy_images = [add_gaussian_noise(img) for img in selected_images]
filtered_images = [apply_mean_filter(img) for img in noisy_images]

# Plot the original, noisy, and filtered images
fig, axes = plt.subplots(10, 3, figsize=(12, 40))
for i in range(10):
    axes[i, 0].imshow(selected_images[i], cmap='gray')
    axes[i, 0].set_title('Original')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(noisy_images[i], cmap='gray')
    axes[i, 1].set_title('Noisy')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(filtered_images[i], cmap='gray')
    axes[i, 2].set_title('Filtered')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()