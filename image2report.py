import torch
import torch.nn as nn
from torchvision import models
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Image2Report(nn.Module):
    def __init__(self, gpt_model_name='gpt2'):
        super(Image2Report, self).__init__()

        # Tokenizer ve GPT2 decoder
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2'nin pad token'ı yok

        self.decoder = GPT2LMHeadModel.from_pretrained(gpt_model_name)

        # Görüntü encoder (ResNet50 → GPT2 embedding boyutuna eşitlenir)
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False  # Görüntü encoder'ı dondur

        resnet.fc = nn.Linear(resnet.fc.in_features, self.decoder.config.n_embd)
        self.encoder = resnet  # Çıkış: (B, 768)

    def forward(self, images, input_ids, attention_mask):
        """
        images: (B, 3, 224, 224)
        input_ids: (B, T)
        attention_mask: (B, T)
        """
        # Görüntüden embedding çıkar
        img_embed = self.encoder(images)              # (B, 768)
        img_embed = img_embed.unsqueeze(1)            # (B, 1, 768)

        # Token embedding'leri al
        txt_embed = self.decoder.transformer.wte(input_ids)  # (B, T, 768)

        # Görsel + metni birleştir (giriş embeddingleri)
        inputs_embeds = torch.cat((img_embed, txt_embed), dim=1)  # (B, T+1, 768)

        # Attention mask'i genişlet (görsel için 1 ekle)
        extended_mask = torch.cat(
            (torch.ones((attention_mask.shape[0], 1), device=attention_mask.device), attention_mask),
            dim=1
        )

        # Loss hesaplaması için: başa -100 ekleyerek görseli loss dışında tut
        labels = input_ids.clone()
        ignore = torch.full((labels.size(0), 1), -100, device=labels.device)
        labels = torch.cat((ignore, labels), dim=1)

        # GPT2 forward
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels
        )
        return outputs
