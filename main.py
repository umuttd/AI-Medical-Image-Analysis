import argparse

def run_training():
    print("Eğitim başlatılıyor...")
    import train_gpt2
    # train_gpt2.py içindeki kodlar zaten çalıştırılabilir durumda olmalı

def run_generate():
    print("Rapor üretme başlatılıyor...")
    import generate
    # generate.py içindeki kod zaten çalıştırılabilir olmalı

def run_app():
    print("Web arayüzü başlatılıyor...")
    import app
    # app.py içindeki Gradio uygulaması çalışacaktır

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-ray Görüntü Raporlama Sistemi")
    parser.add_argument(
        "--mode", choices=["train", "generate", "app"],
        required=True, help="Çalıştırmak istediğiniz modu seçin."
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "generate":
        run_generate()
    elif args.mode == "app":
        run_app()
