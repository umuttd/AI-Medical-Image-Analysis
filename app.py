import torch
from PIL import Image
from torchvision import transforms
import gradio as gr

from image2report import Image2Report
from translate import translate_to_turkish

# Model yükle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Image2Report()
model.load_state_dict(torch.load("image2report_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Görsel işleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Rapor üretme fonksiyonu
def generate_report_from_image(img):
    image = img.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    input_ids = torch.tensor([[50256]], device=DEVICE)  # GPT2'nin <bos> token'ı
    attention_mask = torch.ones_like(input_ids)
    generated = []

    with torch.no_grad():
        img_embed = model.encoder(image_tensor).unsqueeze(1)

        for _ in range(100):
            inputs_embeds = model.decoder.transformer.wte(input_ids)
            inputs_embeds = torch.cat((img_embed, inputs_embeds), dim=1)
            attn_mask = torch.cat((torch.ones((1, 1), device=DEVICE), attention_mask), dim=1)

            outputs = model.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits / 1.2, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            if next_token_id.item() == 50256:
                break

            generated.append(next_token_id.item())
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token_id)), dim=1)

    english = model.tokenizer.decode(generated, skip_special_tokens=True)
    turkish = translate_to_turkish(english)
    return english, turkish

# Gradio Arayüzü
interface = gr.Interface(
    fn=generate_report_from_image,
    inputs=gr.Image(type="pil", label="X-ray Görüntüsü"),
    outputs=[gr.Textbox(label="📄 İngilizce Rapor"), gr.Textbox(label="🇹🇷 Türkçe Rapor")],
    title="X-ray Otomatik Raporlama",
    description="Görsel yükleyin, model İngilizce rapor oluştursun ve Türkçeye çevirsin."
)

if __name__ == "__main__":
    interface.launch(share=True)

#venv39\Scripts\activate
#python app.py