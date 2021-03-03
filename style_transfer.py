import torch
import torch.nn as nn
from torchvision.models import vgg19_bn
from torchvision import transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
from tqdm import tqdm


img_size = 400
channels = 3

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.choosen_layers = [0, 7, 14, 27, 40]
        self.vgg = vgg19_bn(pretrained=True).features[:self.choosen_layers[-1] + 1]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.vgg):
            x = layer(x)

            if layer_num in self.choosen_layers:
                features.append(x)

        return features


def load_image(image_path, device):
    assert os.path.exists(image_path)

    loader = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    image = Image.open(image_path)
    image = loader(image).unsqueeze(0).to(device)

    return image

def denormalize_save_image(image, image_path):
    for i in range(channels):
        image[i] = std[i] * image[i] + mean[i]
    save_image(image, image_path)


def transfer_style(original, style, generated_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_image = load_image(original, device)
    style_image = load_image(style, device)
    generated = original_image.clone().requires_grad_(True)
    model = VGG().to(device).eval()

    # Hyperparameters
    iterations = 5000
    learning_rate = 0.001
    alpha = 1
    beta = 0.25

    optimizer = torch.optim.Adam([generated], lr=learning_rate)

    for step in tqdm(range(iterations)):
        generated_features = model(generated)
        original_features = model(original_image)
        style_features = model(style_image)

        style_loss = 0
        original_loss = 0

        for gen_feature, original_feature, style_feature in zip(
            generated_features, original_features, style_features):

            _, channel, height, width = gen_feature.shape

            original_loss += ((gen_feature - original_feature) ** 2) * 0.5

            # Compute Gram Matrix
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width)
            )

            A = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width)
            )

            style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_feature + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()

    optimizer.step()

    if step % 500 == 0:
        denormalize_save_image(generated, f"./{generated_dir}/{step}.png")

    denormalize_save_image(generated, f"./{generated_dir}/final.png")


if __name__ == "__main__":
    original_path = "./original.jpg"
    style_path = "./style-squared.jpg"
    generated_dir = "./generated"

    transfer_style(original_path, style_path, generated_dir)
