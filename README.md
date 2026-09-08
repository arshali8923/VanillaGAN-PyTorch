GAN Image Generation with PyTorch

A simple Generative Adversarial Network (GAN) built with PyTorch to generate RGB images similar to the CelebA face dataset.

📌 Project Overview

This project implements a basic GAN consisting of:
Generator – Takes random noise (z) and generates a 64 × 64 RGB image.
Discriminator – Takes an image and predicts whether it is real or generated.
CelebA Dataset – Real images are loaded, cropped, resized, converted to tensors, and normalized.
Training – Generator and discriminator are trained together using Binary Cross Entropy loss.

GAN Flow

Random Noise (z)
       ↓
   Generator
       ↓
 Fake Image ─────────┐
                     ↓
                Discriminator
                     ↑
 Real Image ─────────┘
                     ↓
              Real / Fake

🛠️ Technologies Used

Python
PyTorch
Torchvision
NumPy
Matplotlib
PIL

📂 Dataset

The project uses the CelebA aligned images dataset.
The images are processed using the following pipeline:
transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

Normalization converts pixel values approximately from:

[0, 1] → [-1, 1]

This matches the Tanh() output used by the Generator.

🧠 Generator

The Generator receives a random latent vector of size 100 and progressively expands it:
100 → 256 → 512 → 1024 → 12288
The final 12288 values are reshaped into:
3 × 64 × 64
The final activation function is Tanh(), producing values in the range [-1, 1].

🧠 Discriminator

The Discriminator receives a 3 × 64 × 64 image.
The image is flattened and passed through fully connected layers:
12288 → 1024 → 512 → 256 → 1
Sigmoid() produces a probability:
0 → Fake
1 → Real
LeakyReLU(0.2) is used between the hidden layers.

🔄 Training Process

For every batch:

1. Train Discriminator
Load real images.
Generate fake images from random noise.
Calculate loss for real images.
Calculate loss for fake images.
Combine both losses.
Backpropagate and update the discriminator.
d_loss = (real_loss + fake_loss) / 2
The fake images are detached during discriminator training:
discriminator(fake_imgs.detach())
This prevents the discriminator update from changing the Generator.

2. Train Generator
The Generator creates fake images again and tries to make the Discriminator classify them as real.
g_loss = GAN_loss(discriminator(fake_imgs), real_labels)
Therefore, the Generator learns to create increasingly realistic images.

📊 Generated Images
After training, generated images are arranged into a grid using:
torchvision.utils.make_grid()
The images are saved after training as:
epoch_50.png
During training, loss information is printed every 50 batches:
for epoch : 1/50... batch 1... G_loss ... D_loss ...

📦 Installation

Install the required packages:
pip install torch torchvision pillow numpy matplotlib

▶️ How to Run
Download/extract the CelebA aligned image dataset.
Update the dataset path in the Python file:
root_dir_path = r"C:\Users\YourName\Desktop\prime ai&ml\GAN\img_align_celeba\img_align_celeba"

Run the Python script:
python gan.py

The model will train for 50 epochs.

Generated images will be saved as epoch_50.png.

📁 Suggested Project Structure

GAN/
│
├── gan.py
├── README.md
├── epoch_50.png
└── img_align_celeba/
    ├── 000001.jpg
    ├── 000002.jpg
    ├── 000003.jpg
    └── ...

Note: The dataset folder can be large, so it is generally better to exclude it from GitHub using .gitignore

📜 License

This project is intended for learning and educational purposes.
