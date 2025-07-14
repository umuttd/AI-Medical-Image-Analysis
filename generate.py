import torch
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer
from image2report import Image2Report
from translate import translate_to_turkish
# 🔧 Ayarlar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "image2report_model.pth"
IMAGE_PATH = "images/CXR112_IM-0080-1001.png"  # test edeceğin görüntü dosyası

# 🧠 Tokenizer ve model yükle
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = Image2Report()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 🖼️ Görüntüyü yükle ve dönüştür
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

image = Image.open(IMAGE_PATH).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(DEVICE)

# 🔁 Rapor üret (adım adım token çıkar)
generated = []
input_ids = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=DEVICE)
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    img_embed = model.encoder(image_tensor).unsqueeze(1)  # (1, 1, 768)
    print("Görsel embed (ilk 5 değer):", img_embed[0, 0, :5].detach().cpu().numpy())

    for _ in range(100):  # max 100 token üret
        inputs_embeds = model.decoder.transformer.wte(input_ids)
        inputs_embeds = torch.cat((img_embed, inputs_embeds), dim=1)
        attn_mask = torch.cat((torch.ones((1, 1), device=DEVICE), attention_mask), dim=1)

        outputs = model.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits / 1.2, dim=-1)  # temperature=1.2
        next_token_id = torch.multinomial(probs, num_samples=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated.append(next_token_id.item())
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_token_id)), dim=1)

# 📄 Tokenları metne çevir
generated_text = tokenizer.decode(generated, skip_special_tokens=True)
print("\n📝 Üretilen Rapor:\n")
print(generated_text)

translated = translate_to_turkish(generated_text)
print("\n🇹🇷 Türkçe Rapor:\n")
print(translated)
