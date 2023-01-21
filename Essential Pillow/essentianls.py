from PIL import Image
im = Image.open(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Essential Pillow\peacock.jpg")

im.getbands()
('R', 'G', 'B')
im.mode
'RGB'
im.size
(400, 600)

import base64

with open(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Essential Pillow\peacock.jpg", "rb") as image:
    image_string = base64.b64encode(image.read())

import io

image = io.BytesIO(base64.b64decode(image_string))
Image.open(image)

im.save('peacock.png')

# thumbnails
size = 128, 128
im.thumbnail(size)
im.save('thumb.png')

# Cropping images
im = Image.open(r'C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Essential Pillow\peacock.jpg')
box = (100, 150, 300, 300)
cropped_image = im.crop(box)
cropped_image

# Rotating images
rotated = im.rotate(180)
rotated

# Merging images
logo = Image.open(r'C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Essential Pillow\peacock.jpg')

position = (38, 469)
im.paste(logo, position)
im.save('merged.jpg')

card = Image.new("RGBA", (220, 220), (255, 255, 255))
img = Image.open(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\peacock.png").convert("RGBA")
x, y = img.size
card.paste(img, (0, 0, x, y), img)
card.save("test.png", format="png")

# get rid of the black background
im = Image.open(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\peacock.png")
image_copy = im.copy()
position = ((image_copy.width - logo.width), (image_copy.height - logo.height))
image_copy.paste(logo, position,logo)
image_copy

