
### Olası Nedenler (QA'nın Kötü Olması)
1. **Veri Kalitesi ve Uyumluluğu**: QA dataseti (`TR-Extractive-QA-82K`) extractive QA için tasarlanmış – yani cevabı doğrudan metinden çıkarır. Ama model seq2seq (T5/mT5) ile eğitiliyor, bu da abstractive cevaplar üretebilir. Veri formatı ("answer: {question} context: {context}" → {answer}) uygun, ama belki cevaplar çok kısa veya tutarsız.
   
2. **Görev Ayrımı Zorluğu**: Multitask modelde prefix'ler ("summarize: " vs "answer: ") ile görev ayrımı yapılıyor. Model özetlemeye odaklanmış olabilir çünkü özetleme verisi daha fazla veya daha kolay (haber başlıkları kısa ve tutarlı).

3. **Eğitim Dengesi**: Veri dengeli (~4000 özetleme + 4000 QA), ama özetleme örnekleri daha uzun metinler içeriyor ve model bunları daha iyi öğrenmiş olabilir. QA cevapları kısa olduğu için loss daha az etkili.

4. **Metrik Uyumsuzluğu**: Eğitimde kullanılan metrikler (EM, Token F1, ROUGE-L) özetleme için ideal, ama QA için exact match (EM) çok katı olabilir – küçük farklılıklar bile başarısız sayılır.

5. **Model Kapasitesi**: `mt5-small` küçük bir model; QA için daha fazla parametre veya ince ayar gerekebilir.

### İyileştirme Önerileri
Eğitim scriptini (`train_multitask_qlora_2.py`) ve veri hazırlamayı (`prepare_data_2.py`) optimize edelim. İşte adım adım öneriler:

1. **QA Verisini Artır ve Kalitesini Kontrol Et**:
   - QA örnek sayısını artır: `MAX_QA_SAMPLES=8000` yap (şu anda 4000).
   - Veri kalitesini kontrol etmek için küçük bir analiz ekle. Aşağıdaki kodu `prepare_data_2.py`'ye ekleyebiliriz:

     ```python
     # QA kalite kontrolü ekle (build_qa fonksiyonuna)
     def build_qa() -> DatasetDict:
         # ... mevcut kod ...
         # Kalite kontrolü
         valid_rows = []
         for row in rows:
             if len(row["target"]) > 2 and len(row["source"]) > 50:  # Kısa cevapları filtrele
                 valid_rows.append(row)
         rows = valid_rows
         print(f"[QA] Filtered to {len(rows)} high-quality samples")
         # ... geri kalan kod ...
     ```

2. **Görev Spesifik Eğitim**:
   - QA için ayrı LoRA konfigürasyonu dene: Daha yüksek `r` (örneğin 16) veya farklı target_modules.
   - Eğitimde QA örneklerini daha fazla ağırlıklandır: Veri hazırlamada QA'yı çoğalt (örneğin 2x).

3. **Prefix'i Optimize Et**:
   - QA prefix'ini değiştir: "answer the question: " veya "extract answer: " dene. Bu, modelin görevi daha iyi anlamasına yardımcı olabilir.

4. **Eğitim Parametrelerini Ayarla**:
   - QA odaklı metrik ekle: Eğitimde QA örnekleri için ayrı loss hesapla.
   - Daha fazla epoch QA için: `EPOCHS=5` yap ve erken durdurma ekle.

5. **Analiz ve Test**:
   - Eğitim sonrası QA tahminlerini ayrı analiz et. Aşağıdaki kodu eğitim scriptine ekle:

     ```python
     # QA spesifik analiz (eval_predictions.jsonl'den sonra)
     qa_preds = [p for p in pred_txt if p["source"].startswith("answer: ")]
     qa_gold = [g for g in gold_txt if g["source"].startswith("answer: ")]
     qa_em = float(np.mean([1.0 if _norm(p)==_norm(g) else 0.0 for p,g in zip(qa_preds, qa_gold)]))
     print(f"QA EM: {qa_em}")
     ```

Bu değişikliklerle QA performansını %20-50 artırabiliriz. Hangi adımı önce denemek istersin? Kod değişiklikleri için dosyaları düzenleyebilirim!