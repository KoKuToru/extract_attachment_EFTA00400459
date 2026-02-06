import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.io
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob

from ocr import find_letter, letter_images

images = glob('img-*.png')

cleaned_letters = {}

for image in tqdm(images):
    image = Image.open(image).convert('RGB')
    image = transforms.ToTensor()(image)
    image *= 64
    image = image.round()
    image /= 64

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
            letters.append(cropped)
            x += cell_w
        y += line_h

    for letter in tqdm(letters, leave=False):
        l = find_letter(letter)
        if l == '':
            l = 'blank'
        elif l == '/':
            l = 'slash'
        if l == '?':
            continue
        if l not in cleaned_letters:
            cleaned_letters[l] = []
        cl = cleaned_letters[l]
        found = None
        for i, uletter in enumerate(cl):
            if F.l1_loss(letter, uletter[0]) < 1/0xFF:
                found = i
                break
        if found is None:
            cl.append((letter, 1-letter, 1))
        else:
            x = cl[found]
            cl[found] = (x[0], x[1] + (1-letter), x[2] + 1)

for k, v in tqdm(cleaned_letters.items(), leave=False):
    for d,l in enumerate(v):
        torchvision.io.write_png(((1-(l[1]/l[2]))*255).to(torch.uint8), f'letters_reduced/letter_{k}_{d}.png')


