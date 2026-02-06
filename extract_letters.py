import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.io
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob

from ocr import find_letter, letter_images

images = glob('img-*.png')

unique_letters = []

for image in tqdm(images):
    image = Image.open(image).convert('RGB')
    image = transforms.ToTensor()(image)

    letter_w = 8
    cell_w = 8 - 1/5
    letter_h = 12
    line_h = letter_h+3

    letters = []

    y = 39
    while y < image.size(-2) - line_h:
        x = 61
        while x < image.size(-1) - cell_w:
            rx = int(x)
            ry = int(y)
            cropped = image[..., ry:, rx:][..., :letter_h, :letter_w]
            cropped *= 64
            cropped = cropped.round()
            cropped /= 64
            letters.append(cropped)
            x += cell_w
        y += line_h

    for letter in tqdm(letters, leave=False):
        if letter.mean() >= (1 - 1/0xFF):
            continue
        found = False
        for uletter in unique_letters:
            if F.l1_loss(letter, uletter) < 1/0xFF:
                found = True
                break
        if not found:
            unique_letters.append(letter)
            l = find_letter(letter)
            if l == '':
                l = 'blank'
            elif l == '/':
                l = 'slash'
            if l == '?':
                torchvision.io.write_png((letter*255).to(torch.uint8), f'letters/letter_{len(letter_images)+1000+len(unique_letters)}.png')
            else:
                torchvision.io.write_png((letter*255).to(torch.uint8), f'letters/letter_{l}_{len(letter_images)+1000+len(unique_letters)}.png')


