# Extract Attachment From EFTA00400459

thanks to the one who contacted me by mail..
see also https://www.reddit.com/r/netsec/comments/1qw4sfa/

## License

https://creativecommons.org/publicdomain/zero/1.0/

## A simple OCR

0. get linux, you will have a very funny experience with windows.. 😈
1. extract images with `pdfimages EFTA00400459.pdf img`
2. edit `img-000.png` in a image-editor,\
remove everything above the base64, make everything white we don't need,\
overlay `img-001.png` as a guide,\
shift `img-000.png` up or down until `>` matches exactly with `img-001.png`,\
don't forget to remove the `img-001.png` overlay before saving :P
4. edit `img-075.png` in a image-editor,\
remove everything after the base64, make everything white we don't need
5. run `ocr.py` and profit

## Resulting SHA224

calculated with `sha224sum`

|filename|sha224|
|---|---|
| `EFTA00400459.pdf` | `bfa1d9ac2fe5d0271337a5b0809d3406d23dc2af21e25b6480183c2f` |
| `base64_decoded.pdf` | `31fb4aea870b85ed702fa77eb886274cfc400fb260a3085e2e332551` |
| `base64_extracted.txt` | `13fdfc39bce2daf1c794e521c28db30baa088d5af0546301e6c4406b` |

## How does it work?

(check out for more in depth explanation https://www.reddit.com/r/netsec/comments/1qw4sfa/comment/o3y564b/)

in this file a monospace font of size `8x12` (`w=8px, h=12px`) is used.\
we just readout all letters, starting at position `y=39px, x=61px`.\
but because the font `advance_x` is not exactly `8px` apparently .. we are using `8px - 1/5` (found by try & error)\
lineheight is `15px`.\
each letter is then matched with a letter from `letters_done`, just select best `F.l1_loss`.

`letters_done` was created by running `extract_letters.py` and then manually assigning letters and/or checking if auto assignment was correct.
and then replace `letters` with `letters_done` and repeating..

this is not the most clean extraction of the letters and templates, but I didn't care. there were not that many variations.\
first manual assignment of letters took like 30min or so (~300 letters).\
and then rerun with `F.l1_loss(letter, uletter) < 1/0xFF/???` can't remember,\
and just checking if it was correctly auto-assigned resulting in ~1000 letter variations

2 hours were spend around to find out that `letter_l_1988.png` should have been `letter_1_1988.png`. 🤷

reduced the amount of letters required / optimized the templates with `reduce_letters.py`

## How can I use this on other files?

uhm, understand the code and modify it.
