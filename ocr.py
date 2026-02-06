import itertools
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.io
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from concurrent.futures import ProcessPoolExecutor

letters = sorted(glob('letters_done/*.png'))

allowed_letters = [
    *'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+=',
    'slash',
    'blank'
]

letter_images = []
letter_values = []

for l in tqdm(letters):
    x = l.split('/')[-1].split('.')[0].split('_')[1]
    if x not in allowed_letters:
        print('DEAD')
        exit(1)

    image = Image.open(l).convert('RGB')
    image = transforms.ToTensor()(image)
    image *= 64
    image = image.round()
    image /= 64

    if x == 'slash':
        x = '/'
    elif x == 'blank':
        x = ''
    letter_values.append(x)
    letter_images.append(image)

letter_images = torch.stack(letter_images)

def find_letter(letter):
    global letter_images
    # letter_images must be accessible (e.g., global or passed in)
    if letter.mean() >= (1 - 1/255):
        return ""

    letters = torch.stack([letter])

    matrix = letters[None,:] - letter_images[:,None]
    matrix = matrix.abs().mean((-3, -2,-1)).argmin(-2)

    matrix = tuple(letter_values[x.item()] for x in matrix)

    return matrix[0]


if __name__ == '__main__':
    images = sorted(glob('img-*.png'))
    with open('base64_extracted.txt', 'wt') as w:
        for image in images:
            count = 0
            print(image)

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

            letters = torch.stack(letters)

            matrix = letters[None,:] - letter_images[:,None]
            matrix = matrix.abs().mean((-3, -2, -1)).argmin(-2)

            matrix = tuple(letter_values[x.item()] for x in matrix)
            for char in matrix:
                w.write(char)
                count += 1

            print(f'\tcount: {count}')
            w.flush()

    import base64

    with open('base64_extracted.txt', 'rt') as r, \
        open('base64_decoded.pdf', 'wb') as w:
        encoded_data = r.read()
        decoded_data = base64.b64decode(encoded_data)
        w.write(decoded_data)