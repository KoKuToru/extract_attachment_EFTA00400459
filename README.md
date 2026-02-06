# Extract Attachment From EFTA00400459

a simple OCR

1. extract images with `pdfimages`
2. edit first and last page, to match the other pages..
2. run `extract_letters.py` (already done in this repro)
3. set the correct `letter` in the `letters` folder like `letter_?_?.png` where the first `?` is the letter.. (already done in this repro)
4. move it to `lettter_done` (already done in this repro)
5. run `ocr.py` and profit

if ocr is wrong, change `if F.l1_loss(letter, uletter) < 1/0xFF:` in `extract_letters.py` to like `if F.l1_loss(letter, uletter) < 1/0xFF/4:` and rerun `extract_letters.py`... repeat until `ocr` works..

# How can I use this on other files?

uhm, understand the code and modify it.