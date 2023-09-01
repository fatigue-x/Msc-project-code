import matplotlib.pyplot as plt
from PIL import Image

# image_filenames = [
#     ["carvana_01.jpg", "carvana_02.jpg", "carvana_03.jpg", "carvana_04.jpg",
#      "carvana_05.jpg", "carvana_06.jpg", "carvana_07.jpg", "carvana_08.jpg"],
#     ["carvana_01_pred.png", "carvana_02_pred.png", "carvana_03_pred.png", "carvana_04_pred.png",
#      "carvana_05_pred.png", "carvana_06_pred.png", "carvana_07_pred.png", "carvana_08_pred.png"],
#     ["carvana_01_mask.gif", "carvana_02_mask.gif", "carvana_03_mask.gif", "carvana_04_mask.gif",
#      "carvana_05_mask.gif", "carvana_06_mask.gif", "carvana_07_mask.gif", "carvana_08_mask.gif"]
# ]

image_filenames = [
    ["pascal_01.jpg", "pascal_02.jpg", "pascal_03.jpg", "pascal_04.jpg"],
    ["pascal_01_pred.png", "pascal_02_pred.png", "pascal_03_pred.png", "pascal_04_pred.png"],
    ["pascal_01_mask.png", "pascal_02_mask.png", "pascal_03_mask.png", "pascal_04_mask.png"],
    ["pascal_05.jpg", "pascal_06.jpg", "pascal_07.jpg", "pascal_08.jpg"],
    ["pascal_05_pred.png", "pascal_06_pred.png", "pascal_07_pred.png", "pascal_08_pred.png"],
    ["pascal_05_mask.png", "pascal_06_mask.png", "pascal_07_mask.png", "pascal_08_mask.png"]
]

num_rows = len(image_filenames)
num_cols = len(image_filenames[0])

fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4))  # Adjust the figsize as needed

for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        img = Image.open(image_filenames[i][j])
        ax.imshow(img)
        ax.axis('off')

plt.tight_layout()
plt.show()


