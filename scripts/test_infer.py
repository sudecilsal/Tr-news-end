# scripts/test_infer.py
# mT5 + LoRA (veya merge edilmiş) için hızlı inference
# Özet ve QA modları, interaktif kullanım, CPU/GPU otomatik ayar

import os
import sys
import argparse
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    PeftModel = None
    HAS_PEFT = False


SUMM_PREFIX = "summarize: "
QA_PREFIX   = "answer: "


def detect_device_and_dtype():
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        major = torch.cuda.get_device_capability(0)[0]
        use_bf16 = major >= 8  # Ampere+
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        device = "cuda"
    else:
        dtype = torch.float32
        device = "cpu"
    return device, dtype


def load_model_and_tokenizer(model_dir: str,
                             base_model: str = "google/mt5-small",
                             dtype=None,
                             device="cuda"):
    """
    1) Önce base + LoRA (PEFT) yüklemeyi dener
    2) Olmazsa model_dir'den tam modeli doğrudan yükler (merge edilmiş gibi)
    """
    print(f"[INFO] Loading tokenizer from: {base_model if HAS_PEFT else model_dir}")
    if HAS_PEFT:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # PEFT yolu
    if HAS_PEFT:
        try:
            print(f"[INFO] Trying base+LoRA: base='{base_model}' + adapter='{model_dir}' (dtype={dtype})")
            base = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=dtype)
            model = PeftModel.from_pretrained(base, model_dir)
            model.to(device)
            print("[INFO] Loaded PEFT (base + LoRA adapter).")
            return tokenizer, model
        except Exception as e:
            print(f"[WARN] PEFT load failed -> {e}")

    # Merge edilmiş veya saf model yolu
    print(f"[INFO] Trying direct model load from '{model_dir}' (dtype={dtype})")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=dtype)
    model.to(device)
    print("[INFO] Loaded direct model.")
    return tokenizer, model


def build_prompt(task: str, article: str, question: str = None):
    article = (article or "").strip()
    if task == "summarize":
        return SUMM_PREFIX + article
    elif task == "qa":
        q = (question or "").strip()
        return f"{QA_PREFIX}{q} context: {article}"
    elif task == "both":
        # both modunda ayrı ayrı çağıracağız, burada tek prompt dönmüyoruz
        return None
    else:
        raise ValueError(f"Unknown task: {task}")


def generate_text(tokenizer, model, prompt: str,
                  max_source_len=384,
                  max_new_tokens=64,
                  num_beams=4,
                  device="cuda"):
    inputs = tokenizer(
        prompt,
        max_length=max_source_len,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False
        )
    out = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return out[0] if out else ""


def read_text_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="mT5 LoRA inference: summarize / QA / both")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Eğitim çıktısının bulunduğu klasör (LoRA adapter veya merge edilmiş model).")
    parser.add_argument("--base_model", type=str, default="google/mt5-small",
                        help="PEFT varsa base model adı (örn: google/mt5-small).")
    parser.add_argument("--task", type=str, default="both", choices=["summarize", "qa", "both"],
                        help="Çalıştırma modu.")
    parser.add_argument("--article", type=str, default=None, help="Haber metni (komut satırından).")
    parser.add_argument("--article_file", type=str, default=None, help="Haber metni dosyası (alternatif).")
    parser.add_argument("--question", type=str, default=None, help="QA için soru.")
    parser.add_argument("--interactive", action="store_true", help="Etkileşimli mod.")
    parser.add_argument("--max_source_len", type=int, default=384)
    parser.add_argument("--max_new_tokens_summ", type=int, default=64)
    parser.add_argument("--max_new_tokens_qa", type=int, default=24)
    parser.add_argument("--num_beams", type=int, default=4)

    args = parser.parse_args()

    device, dtype = detect_device_and_dtype()
    print(f"[INFO] Device={device} | dtype={dtype}")

    tok, model = load_model_and_tokenizer(
        model_dir=args.model_dir,
        base_model=args.base_model,
        dtype=dtype,
        device=device
    )

    # Interaktif mod
    if args.interactive:
        print("\n[INTERACTIVE MODE] Çıkmak için Ctrl+C.")
        while True:
            try:
                if args.task in ("summarize", "both"):
                    article = input("\nHaber metnini girin:\n> ").strip()
                    if not article:
                        print("Boş metin. Tekrar deneyin.")
                        continue
                    prompt = build_prompt("summarize", article)
                    summary = generate_text(
                        tok, model, prompt,
                        max_source_len=args.max_source_len,
                        max_new_tokens=args.max_new_tokens_summ,
                        num_beams=args.num_beams,
                        device=device
                    )
                    print("\n[ÖZET]\n" + summary)

                if args.task in ("qa", "both"):
                    if args.task == "qa":  # QA modunda metni ayrı al
                        article = input("\nHaber metnini girin:\n> ").strip()
                    question = input("\nSoru:\n> ").strip()
                    prompt = build_prompt("qa", article, question)
                    answer = generate_text(
                        tok, model, prompt,
                        max_source_len=args.max_source_len,
                        max_new_tokens=args.max_new_tokens_qa,
                        num_beams=args.num_beams,
                        device=device
                    )
                    print("\n[CEVAP]\n" + answer)
            except KeyboardInterrupt:
                print("\nÇıkılıyor.")
                break
        return

    # Non-interaktif mod
    # Önce makale metnini bul
    article = args.article
    if (article is None or len(article.strip()) == 0) and args.article_file:
        article = read_text_from_file(args.article_file)

    if not article:
        print("Hata: Haber metni girilmedi. --article veya --article_file kullanın.")
        sys.exit(1)

    if args.task == "summarize":
        prompt = build_prompt("summarize", article)
        summary = generate_text(
            tok, model, prompt,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens_summ,
            num_beams=args.num_beams,
            device=device
        )
        print("\n=== ÖZET ===")
        print(summary)

    elif args.task == "qa":
        if not args.question:
            print("Hata: QA için --question gerekli.")
            sys.exit(1)
        prompt = build_prompt("qa", article, args.question)
        answer = generate_text(
            tok, model, prompt,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens_qa,
            num_beams=args.num_beams,
            device=device
        )
        print("\n=== SORU ===")
        print(args.question)
        print("\n=== CEVAP ===")
        print(answer)

    elif args.task == "both":
        # 1) Özet
        prompt_s = build_prompt("summarize", article)
        summary = generate_text(
            tok, model, prompt_s,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens_summ,
            num_beams=args.num_beams,
            device=device
        )
        print("\n=== ÖZET ===")
        print(summary)

        # 2) QA (soru verilmişse)
        if args.question:
            prompt_q = build_prompt("qa", article, args.question)
            answer = generate_text(
                tok, model, prompt_q,
                max_source_len=args.max_source_len,
                max_new_tokens=args.max_new_tokens_qa,
                num_beams=args.num_beams,
                device=device
            )
            print("\n=== SORU ===")
            print(args.question)
            print("\n=== CEVAP ===")
            print(answer)
        else:
            print("\n[INFO] QA için --question verilmedi; sadece özet ürettim.")


if __name__ == "__main__":
    main()
