from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

src_lang = "eng_Latn"
tgt_lang = "tur_Latn"


def translate_to_turkish(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # ✔ Dil kodunu manuel olarak ID'ye çeviriyoruz
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    inputs["forced_bos_token_id"] = tgt_lang_id

    translated = model.generate(**inputs, max_length=512)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
