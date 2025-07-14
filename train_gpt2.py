import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from dataset_gpt2 import XrayReportDataset
from image2report import Image2Report

# 🔧 Ayarlar
BATCH_SIZE = 4
EPOCHS = 5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Dataset
dataset = XrayReportDataset("images", "indiana_reports.json", max_length=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 🔹 Model
model = Image2Report()
model.to(DEVICE)
model.train()

# 🔹 Optimizer
optimizer = AdamW(model.decoder.parameters(), lr=5e-5)

# 🔁 Eğitim döngüsü
for epoch in range(EPOCHS):
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in loop:
        images = batch['image'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        outputs = model(images, input_ids, attention_mask)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

# 🔚 Kaydet (isteğe bağlı)
torch.save(model.state_dict(), "image2report_model.pth")
