from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as n
from sklearn.cluster import KMeans
import os
import re
# Python 3.8.1


class KMeansCompressor:
    def __init__(self, image):
        self._pixels_, self._img_size_ = KMeansCompressor._get_img_data(image)
        self._cmp_pixels_ = None

    @staticmethod
    def _get_img_data(image):
        """Метод, возвращающий массив пикселей и размер изображения."""
        pixels = n.asarray(image)
        pixels = pixels.reshape((image.height * image.width, 3))
        return pixels, (image.height, image.width, 3)

    def perform(self, n_clusters):
        """Медод, выполняющий сжатие."""
        k_means = KMeans(n_clusters).fit(self._pixels_)
        self._cmp_pixels_ = n.array([k_means.cluster_centers_[label] for label in k_means.labels_]).astype('uint8')

    def get_compressed_image(self):
        return Image.fromarray(self._cmp_pixels_.reshape(self._img_size_))


def concatenate(path):
    source = Image.open(path)
    img_name = re.match(r'(.+)\.', path).group(1)
    img = Image.new('RGB', (source.size[0] * 3, source.size[1] * 2))
    for p in range(6):
        c_img = Image.open(f'compressed/{img_name}_compressed{2**p + 1}.png')
        drawer = ImageDraw.Draw(c_img)
        font = ImageFont.truetype("arial.ttf", source.size[0] // 10)
        drawer.text((0, 0), f'k = {2 ** p}', (255, 255, 255), font=font)
        img.paste(c_img, ((p % 3) * source.size[0], (p // 3) * source.size[1]))
    img.save('different_k.png')


def main(path):
    if not os.path.isdir('compressed'):
        os.mkdir('compressed')
    original_image = Image.open(path)
    img_name = re.match(r'(.+)\.', path).group(1)
    img_comp = KMeansCompressor(original_image)
    for cl_num in range(1, 34):
        print(f'Производится сжатие для {cl_num} кластер{"a" if cl_num % 10 == 1 else "ов"}', flush=True)
        img_comp.perform(cl_num)
        img_comp.get_compressed_image().save(f'compressed/{img_name}_compressed{cl_num}.png')
    concatenate(path)
    for x in os.listdir('compressed'):
        os.remove(f'compressed/{x}')

main('Georg.png')
