import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import GPT2Tokenizer

class XrayReportDataset(Dataset):
    def __init__(self, image_dir, report_json, tokenizer_name='gpt2', max_length=128):
        self.image_dir = image_dir
        with open(report_json, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Görüntü yükle ve dönüştür
        image_path = os.path.join(self.image_dir, item['image_id'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Rapor metnini oluştur
        text = item['findings'] + " " + item['impression']
        tokens = self.tokenizer(text, padding='max_length', truncation=True,
                                max_length=self.max_length, return_tensors="pt")

        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'text': text
        }