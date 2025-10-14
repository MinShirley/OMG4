from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Resize
import os
from torchvision.io import ImageReadMode, decode_image
import json

class Camera:
    def __init__(self, image_path, H, W, timestamp, downsample=2
                 ):
 
        img = decode_image(image_path, mode=ImageReadMode.RGB)
        transform = Resize((int(H / downsample), int(W / downsample)))
        image = transform(img) / 255
       
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)
        self.image = image
        self.timestamp = timestamp
        self.height = H/downsample
        self.width = W/downsample
 
 
def load_test_dataset(path, EXT="png"):
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        contents = json.load(json_file)
    test_dataset = []
    H = contents['h']
    W = contents['w']
    for frame in contents['frames']:
        image_path = f"{path}/{frame['file_path']}.{EXT}"
        test_dataset.append(Camera(
            image_path = image_path, H=H, W=W, timestamp=frame['time']
        ))
    return test_dataset