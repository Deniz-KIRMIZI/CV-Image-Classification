class MonkeyDataset(Dataset):

    #def __init__(self, **kwargs):
  def __init__(self, root_dir, transform=None):

        #Args:
            #root_dir (string): Directory with all the images.
            #transform (callable, optional): Optional transform to be applied on a sample.

        self.root_dir = root_dir
        self.transform = transform
        self.labels = os.listdir(root_dir)
        self.files = []
        self.label_map = {}

        for idx, label in enumerate(self.labels):
            self.label_map[label] = idx
            class_path = os.path.join(root_dir, label)
            for img in os.listdir(class_path):
                self.files.append((os.path.join(class_path, img), idx))
    #This function should return sample count in the dataset'''
    #def __len__(self):
    #    return self.data.shape[0]
  def __len__(self):
         return len(self.files)
    #This function should return a single sample and its ground truth value from the dataset corresponding to index parameter '''
    #def __getitem__(self, index):
        #return _x, _y
  def __getitem__(self, idx):
        img_path, label = self.files[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataset(root):
    # TODO:
    # Read training, validation and test set files
    # Normalize datasets
  transform = transforms.Compose([
  transforms.Resize((128, 128)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
  train_dataset = MonkeyDataset(root_dir='/content/drive/MyDrive/monke/Colab Notebooks/dataset_monkeys/training', transform=transform)
  val_dataset = MonkeyDataset(root_dir='/content/drive/MyDrive/monke/Colab Notebooks/dataset_monkeys/validation', transform=transform)
  test_dataset = MonkeyDataset(root_dir='/content/drive/MyDrive/monke/Colab Notebooks/dataset_monkeys/test', transform=transform)

# Data loaders
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  return train_dataset, val_dataset, test_dataset