from torchvision import transforms

DATA_DIR = 'data'
DATA_FILE = 'data.gzip'
MODEL_FILE = 'model.pt'

# available actions
available_actions = [[0, 0, 0],  # no action
                     [-1, 0, 0],  # left
                     [-1, 0, 1],  # left+break
                     [1, 0, 0],  # right
                     [1, 0, 1],  # right+break
                     [0, 1, 0],  # acceleration
                     [0, 0, 1], ]  # break

# transformations for training/testing
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(1),
    transforms.Pad((12, 12, 12, 0)),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])
