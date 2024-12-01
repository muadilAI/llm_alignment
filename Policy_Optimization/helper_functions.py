import os
import re
import gc
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import single_meteor_score
from bert_score import score as bert_score
import pkg_resources
from huggingface_hub import login, Repository, HfApi
import shutil


# Daha fazla temizleme işlemine gerek varsa bu linkten bakılabilir.
"""
https://github.com/giocoal/reddit-tldr-summarizer-and-topic-modeling/blob/main/0_normalizing.py#L319
"""

available_metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "BERTScore"]

def get_from_openai_summarize_comparisons(dataset_names:list = None):
    """
        hf://datasets/CarperAI/openai_summarize_comparisons/ veri setinin istenilen(train, test, val1, val2) ya da tüm tüm veri seti döner

        Parametre:
            split_dataset: Tüm veri seti isteniyorsa herhangi bir şey yazılmasına gerek yok.
                            eğitim veri seti isteniyorsa "train"
                            test veri seti isteniyorsa "test"
                            doğrulama veri seti isteniyorsa "valid1" ya da "valid2"
        Dönen deger:
            İstenilen veri setinin "prompt", "chosen", "rejected" sütunlarına sahip pandas DataFrame yapısında döner.
    """

    if dataset_names == None:
        dataset_names = ["train", "test", "valid1", "valid2"]

    splits = {'train': 'data/train-00000-of-00001-3cbd295cedeecf91.parquet',
              'test': 'data/test-00000-of-00001-0845e2eec675b16a.parquet',
              'valid1': 'data/valid1-00000-of-00001-b647616a2be5f333.parquet',
              'valid2': 'data/valid2-00000-of-00001-2655c5b3621b6116.parquet'}

    if dataset_names not in splits.keys():
        raise ValueError(f"Invalid split_dataset value: {dataset_names}. Choose from {list(splits.keys())}.")

    df_raw = pd.DataFrame()
    for dataset_name in dataset_names:
        df_temp = pd.read_parquet("hf://datasets/CarperAI/openai_summarize_comparisons/" + splits[dataset_name])
        df_raw = pd.concat([df_raw, df_temp])

    return df_raw

# Metin verisini temizleme
def remove_html_tags(sentence):
    pattern = re.compile("<.*?>")
    cleaned_sentence = re.sub(pattern,'',sentence).strip()
    return cleaned_sentence

def clean_text(text):
    """
        Parametre:
            text: String ifade

        Dönen Deger:
            Temizlenmiş string ifade döner.
    """
    text = remove_html_tags(text)
    # Özel karakterleri kaldırma: "\n", "\r", "#"
    text = re.sub(r'\n', ' ', text)  # Yeni satır karakterlerini boşluk ile değiştir
    text = re.sub(r'\r', ' ', text)   # Carriage return karakterlerini boşluk ile değiştir
    text = re.sub(r'[#]', '', text)   # # karakterlerini kaldır
    # "r/" kombinasyonunu kaldırma
    text = re.sub(r'\br/\b', '', text)  # "r/" kelimesinin başında ve sonunda boşluk olmamalı
    # "[" ve "]" arasındaki metni kaldırma
    text = re.sub(r'\[.*?\]', '', text)  # Köşeli parantezler arasındaki kısmı kaldır
    # Noktalama işaretlerini kaldırmak
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır

    # Sayıları kelime ifadelerine çevir
    def number_to_words(match):
        # Inflect kütüphanesini başlat
        p = inflect.engine()
        number = int(match.group(0))
        return p.number_to_words(number)

    text = re.sub(r'\b\d+\b', number_to_words, text)  # Sayıları kelime ifadelerine dönüştür

    return text

def extract_from_openai_summarize_comparisons_dataset(row):
    """
        Parametre:
            row: DataFrame yapısındaki satır ifadesi lazım.

        Dönen Deger:
            Prompt sütunundaki string ifadeden özellik çıkarımı
    """

    category_post = clean_text(row["prompt"]).split("POST", 1)

    #POST anahtar kelimesi içermiyorsa özetlenecek metin aynı kalsın
    if len(category_post) < 2:
        pd.Series([row["prompt"], "belirsiz", "belirsiz"])

    #Burada özetlenecek metnin kendisi
    post = category_post[1]

    #Kategori ve başlıkları elde etmek için değişkene atanır
    raw_category = category_post[0]

    #Kategori ve başlık bilgileri elde edilir.
    subreddit_match = re.search(r'SUBREDDIT\s+(.*?)(?=\s+TITLE)', raw_category)
    title_match = re.search(r'TITLE\s*(.*)', raw_category)

    # Değerleri çıkarma
    subreddit_value = subreddit_match.group(1).strip().lower() if subreddit_match else None
    title_value = title_match.group(1).strip().lower() if title_match else None

    return pd.Series([post, subreddit_value, title_value])



def save_df_dataset_to_parquet(df, filename = "pd_dataset.parquet"):
    """
        DataFrame yapısında olan veriyi kaydeder
        Parameter:
            df: veriler pandas DataFrame yapısındadır
            filename: kaydedilecek dosya uzantısı

    """
    df.to_parquet("data.parquet")
    print(f"{filename} dosya uzantısına kaydedildi.")

def laod_df_dataset_to_parquet(filename = "pd_dataset.parquet"):
    # Parquet formatındaki dosyayı yükle
    df_loaded = pd.read_parquet(filename)
    return df_loaded

import pickle


def save_dataset_to_pickle(data, filename="dataset.pkl"):
    """
    Veriyi pickle dosyasına kaydeder. Gerekirse dosya yolunu oluşturur.

    Parametreler:
        data: Sözlük formatında veri (DPOTrainer gibi kütüphaneler bu formatı destekler).
        filename: Kaydedilecek dosyanın adı ve uzantısı (varsayılan: "dataset.pkl").
    """
    # Dosya yolunun dizin kısmını al
    directory = os.path.dirname(filename)

    # Dizin yoksa oluştur
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Dizin oluşturuldu: {directory}")

    # Dosyayı kaydet
    with open(filename, "wb") as file:
        pickle.dump(data, file)
    print(f"{filename} dosyasına kaydedildi.")

def laod_df_dataset_to_pickle(filename = "pd_dataset.pkl"):
    """
        Parameter:
            filename = Verinin alınacağı dosya uzantısı
        Return:
            DPOTrainer kütüphanesinin kabul edeceği şekilde yani sözlük şeklinde döner
    """
    # Kaydedilmiş objeyi yüklemek
    with open(filename, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data


def correct_me(text):
    """
     Parameter:
        text: String ifade
    Return:
        Kelime düzeyinde hataları bulup düzeltme
        Örnek: soome --> some
    """
    textBlb = TextBlob(text)
    textCorrected = str(textBlb.correct())   # Correcting the text
    return textCorrected


def character_repeatation(text):
    """
        Parameter:
            text: Temizlenecek string ifade
        Return:
            Tekrarlı harfler ve noktalama işaretlerini temizleyip string ifadeyi döndürür.
    """
    # Pattern matching for all case alphabets
    # \1   It refers to the first capturing group.
    # {2,} It means we are matching for repetition that occurs more than two times (or equal).
    # r’\1\1' → It limits all the repetition to two characters.
    Pattern_alpha = re.compile(r"([A-Za-z])\1{2,}", re.DOTALL)
    # Limiting all the  repeatation to two characters.
    Formatted_text = Pattern_alpha.sub(r"\1\1", text)
    # Pattern matching for all the punctuations that can occur
    Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
    # Limiting punctuations in previously formatted string to only one.
    Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
    return Combined_Formatted


def export_library_versions(file_path=None):
    """
    Sistemde yüklü olan tüm kütüphanelerin adlarını ve versiyonlarını yazdırır.
    İsteğe bağlı olarak bir txt dosyasına kaydeder.

    Parametreler:
    - file_path (str): Kütüphane ve versiyon bilgilerini kaydetmek için dosya yolu.
                       None ise dosya kaydedilmez.

    Döndürülen Değer:
    - library_versions (str): Kütüphane ve versiyon bilgilerini içeren bir string.
    """
    # Tüm yüklü paketleri al
    installed_packages = pkg_resources.working_set
    library_versions = ""

    # Kütüphane adlarını ve versiyonlarını birleştir
    for package in installed_packages:
        library_versions += f"{package.project_name}=={package.version}\n"

    # Konsola yazdır
    print(library_versions)

    # Dosya kaydedilmek istenirse
    if file_path:
        # Dosya yolunun dizin kısmını al
        directory = os.path.dirname(file_path)

        # Dizin yoksa oluştur
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Dizin oluşturuldu: {directory}")

        # Dosyayı kaydet
        with open(file_path, "w") as file:
            file.write(library_versions)
            print(f"\nKütüphane versiyonları {file_path} dosyasına kaydedildi.")

    return library_versions

def t5_DPO_format_dataset(examples):
    """
    Girdi olarak bir örnek listesini alır ve Hugging Face Dataset formatında döner.

    Parametre:
        examples: List türünde, 'prompt', 'chosen', 'rejected' anahtarlarına sahip veri.

    Dönen Değer:
        Hugging Face Dataset formatında dönüştürülmüş veri.
    """
    # Her bir örnek için formatlama
    prompts = [f"System: I want you to summarize this text\nDocument: {example['prompt']}" for example in examples]
    chosens = [example['chosen'] for example in examples]
    rejecteds = [example['rejected'] for example in examples]

    # Dataset formatında döndür
    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds
    })

def pipeline_dpo(
    model_name: str,
    dataset: Dataset,
    target_modules: list = None,
    base_path: str = "/content/drive/MyDrive/Bitirme/",
    dataset_name: str = "openai_summarize_comparisons",
    beta: float = 0.1,
    max_prompt_length: int = 512,
    max_length: int = 768,
    per_device_train_batch_size: int = 2,
    num_train_epochs: int = 1,
    learning_rate: float = 5e-5,
    gradient_checkpointing: bool = True,
    bf16: bool = True,
    is_ref_model: bool = True
):
    """
    Birden fazla model için yeniden kullanılabilir DPO eğitim fonksiyonu.

    Parametreler:
    - model_name (str): Hugging Face model ismi.
    - dataset (Dataset): Eğitim veri seti (train, prompt, chosen, rejected sütunları içermeli).
    - target_modules (list): LoRA adaptasyonunda değiştirilecek katmanlar (varsayılan: T5 için `['q', 'v', 'k', 'o']`).
    - base_path (str): Temel dosya yolu.
    - dataset_name (str): Kullanılan veri setinin ismi.
    - beta (float): DPO'nun beta hiperparametresi.
    - max_prompt_length (int): Maksimum giriş uzunluğu.
    - max_length (int): Maksimum çıktı uzunluğu.
    - per_device_train_batch_size (int): Her cihazda kullanılacak batch boyutu.
    - num_train_epochs (int): Eğitimde kullanılacak epoch sayısı.
    - learning_rate (float): Öğrenme oranı.
    - gradient_checkpointing (bool): Bellek optimizasyonu için gradyan kontrolü.
    - bf16 (bool): BF16 (16-bit floating point) kullanımı.

    Döndürmez, modeli kaydeder.
    """
    import os

    # Model ve veri seti adlarını dosya için uygun hale getir
    model_part = model_name.replace("/", "_")
    dataset_part = dataset_name.replace("/", "_")
    save_path = f"{base_path}{model_part}_with_{dataset_part}"
    os.makedirs(save_path, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # LoRA yapılandırması
    if target_modules is None:
        target_modules = ['q', 'v', 'k', 'o']  # T5 varsayılan modüller
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=target_modules
    )

    # Model ve referans model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    if is_ref_model:
        ref_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
    else:
        ref_model = None

    # DPO yapılandırması
    dpo_config = DPOConfig(
        beta=beta,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        output_dir=save_path,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=gradient_checkpointing,  # Dinamik kontrol
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=100,
        num_train_epochs=num_train_epochs,
        bf16=bf16  # Dinamik kontrol
    )

    # DPOTrainer oluşturma
    force_use_ref_model = is_ref_model
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        force_use_ref_model= force_use_ref_model
    )

    # Model eğitimi
    dpo_trainer.train()

    # Model ve tokenizer'ı kaydetme
    dpo_trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # GPU bellek temizleme
    del dpo_trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # LoRA adaptörünü birleştirme ve kaydetme
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, save_path)
    model = model.merge_and_unload()

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model and tokenizer saved to {save_path}")


def preprocess_openai_summarize_comparisons_dataset(df, topic_list: list = None):
    """
    OpenAI summarize comparisons veri setini işler ve belirtilen topic_list'e göre filtreler.

    Parametreler:
    - df (pd.DataFrame): Girdi veri seti.
    - topic_list (list): İlgili konuların listesi. Varsayılan olarak tüm konular dahil edilir.

    Döndürür:
    - pd.DataFrame: İşlenmiş ve filtrelenmiş veri seti.
    """
    # Orijinal DataFrame'in kopyası
    df2 = df.copy()

    # extract_from_openai_summarize_comparisons_dataset fonksiyonunu uygula
    df2[["prompt", "topic", "title"]] = df2.apply(
        extract_from_openai_summarize_comparisons_dataset, axis=1
    )

    # `chosen` ve `rejected` sütunlarını temizle
    df2["chosen"] = df2["chosen"].apply(clean_text)
    df2["rejected"] = df2["rejected"].apply(clean_text)

    # Eğer topic_list sağlanmamışsa tüm veriyi döndür
    if topic_list is None:
        return df2[["prompt", "chosen", "rejected"]]

    # Belirtilen topic_list'e göre filtreleme
    df2 = df2[df2["topic"].isin(topic_list)]

    # Eğer filtreleme sonucunda boş DataFrame varsa uyarı ver
    if df2.empty:
        print("Uyarı: Belirtilen topic_list'e uyan veri bulunamadı.")

    # Sadece gerekli sütunları döndür
    return df2[["prompt", "chosen", "rejected"]]


def calculate_summary_metrics(references, predictions, metrics=None):
    """
    Büyük dil modellerinde başarı metriklerini hesaplamak için fonksiyon.

    Varsayılan metrikler
    available_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'BERTScore']
    Args:
        references (list): Gerçek özetlerin listesi.
        predictions (list): Model tarafından üretilen özetlerin listesi.
        metrics (list, optional): Hesaplanmasını istediğiniz metriklerin isimleri.
                                  Geçerli metrikler: 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'BERTScore'.

    Returns:
        dict: Seçilen metriklerin ortalamalarını içeren bir sözlük.
    """
    if len(references) != len(predictions):
        raise ValueError("References ve predictions aynı uzunlukta olmalı!")

    if metrics is None:
        metrics = available_metrics  # Tüm metrikleri hesapla
    else:
        # Geçerli olmayan metrikleri kontrol et
        invalid_metrics = [metric for metric in metrics if metric not in available_metrics]
        if invalid_metrics:
            raise ValueError(f"Geçersiz metrik(ler): {invalid_metrics}. Geçerli metrikler: {available_metrics}")

    # ROUGE skorları için scorer tanımlanıyor
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    meteor_scores = []

    # ROUGE ve METEOR hesaplamaları
    for ref, pred in zip(references, predictions):
        if any(metric.startswith('ROUGE') for metric in metrics):
            rouge_scores = rouge_scorer_obj.score(ref, pred)
            if 'ROUGE-1' in metrics:
                rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            if 'ROUGE-2' in metrics:
                rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            if 'ROUGE-L' in metrics:
                rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

        if 'METEOR' in metrics:
            tokenized_ref = nltk.word_tokenize(ref)
            tokenized_pred = nltk.word_tokenize(pred)
            meteor_scores.append(single_meteor_score(tokenized_ref, tokenized_pred))

    # BERTScore hesaplaması
    bert_f1 = None
    if 'BERTScore' in metrics:
        P, R, F1 = bert_score(predictions, references, lang="en", rescale_with_baseline=True)
        bert_f1 = F1.mean().item()

    # Ortalama skorlar
    scores = {}
    if 'ROUGE-1' in metrics:
        scores['ROUGE-1'] = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    if 'ROUGE-2' in metrics:
        scores['ROUGE-2'] = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    if 'ROUGE-L' in metrics:
        scores['ROUGE-L'] = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    if 'METEOR' in metrics:
        scores['METEOR'] = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    if 'BERTScore' in metrics:
        scores['BERTScore'] = bert_f1

    return scores



def add_metrics_to_dataframe(metrics_df, model_name, references, predictions, metrics=None):
    """
    Başarı metriklerini hesaplayarak verilen DataFrame'e ekler.

    Args:
        metrics_df (pd.DataFrame): Başarı metriklerini saklayan mevcut DataFrame.
        model_name (str): Modelin adı.
        references (list): Gerçek özetlerin listesi.
        predictions (list): Model tarafından üretilen özetlerin listesi.

    Returns:
        pd.DataFrame: Güncellenmiş metrikler tablosu.
    """

    # Metrikleri hesapla
    scores = calculate_summary_metrics(references, predictions, metrics = metrics)

    # Yeni bir satır olarak metrikleri eklemek için model adını ekle
    scores["Model Name"] = model_name
    metrics_df = pd.concat([metrics_df, pd.DataFrame([scores])], ignore_index=True)

    return metrics_df

def save_excell(name, df, base_path='/content/drive/MyDrive/Bitirme/results/'):
    # Model adını dosya ismi için uygun hale getir
    name_safe = name.replace("/", "_")
    today = datetime.today().strftime('%d%m%Y')
    file_path = f'{base_path}{today}_results_{name_safe}.xlsx'
    df.to_excel(file_path, index=False)
    print(f"Sonuçlar Excel'e kaydedildi: {file_path}")


def remove_tldr(text):
    """
    Metindeki 'TLDR ' ifadesini kaldırır.

    Args:
        text (str): İşlenecek metin.

    Returns:
        str: 'TLDR ' ifadesinden arındırılmış metin.
    """
    return text.replace("TLDR ", "")


def remove_citizens_for_the_republic(text):
    """
    Metindeki 'Citizens for the Republic' ifadesini kaldırır.

    Args:
        text (str): İşlenecek metin.

    Returns:
        str: 'Citizens for the Republic' ifadesinden arındırılmış metin.
    """
    return text.replace("Citizens for the Republic", "")


def summarize_and_save_metrics_AutoModelForSeq2SeqLM(models, tokenizer_names, texts, references, metrics_df=None,
                                                     device="cpu", max_token_length=512, metrics = None):
    """
    Modellerle özetleme yapar, başarı metriklerini hesaplar ve DataFrame'e kaydeder.

    Args:
        models (list): Özetleme yapacak model isimlerinin listesi.
        tokenizer_names (list): Model isimlerine karşılık gelen tokenizer isimlerinin listesi.
        texts (list): Özetlenecek metinlerin listesi.
        references (list): Gerçek özetlerin listesi (texts ile aynı uzunlukta olmalı).
        metrics_df (pd.DataFrame, optional): Başarı metriklerini saklayan mevcut DataFrame.
                                             None ise yeni bir DataFrame oluşturulur.
        device (str): Modeli çalıştırmak için kullanılacak cihaz ('cpu' veya 'cuda').

    Returns:
        pd.DataFrame: Güncellenmiş metrikler tablosu.
    """
    # Eğer metrics_df None ise boş bir DataFrame oluştur
    if metrics_df is None or not isinstance(metrics_df, pd.DataFrame):
        metrics_df = pd.DataFrame(columns= ["Model Name"] + available_metrics)
        print("Yeni bir metrics_df oluşturuldu.")

    # texts ve references uzunluk kontrolü
    if len(texts) != len(references):
        raise ValueError("texts ve references aynı uzunlukta olmalıdır.")

    try:
        for model_name, tokenizer_name in zip(models, tokenizer_names):
            # Model ve tokenizer'ı yükle
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print(f"{model_name} yüklendi ve çıktılar alınıyor.")

            # Tahminleri oluştur
            predictions = []
            for text in texts:
                # Girişleri cihaz üzerine taşı
                inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)

                # Modelden çıktı al
                outputs = model.generate(inputs, max_length=max_token_length, num_beams=4, early_stopping=True)

                # Çıktıyı CPU'ya taşı ve decode et
                summary = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
                predictions.append(summary)

            # Başarı metriklerini DataFrame'e ekle
            metrics_df = add_metrics_to_dataframe(metrics_df, model_name, references, predictions, metrics = metrics)
            print(f"Model '{model_name}' için başarı metrikleri kaydedildi.")

    except Exception as e:
        print(f"Hata oluştu: {e}")
        print("Şu ana kadar oluşturulan metrics_df döndürülüyor.")

    return metrics_df


def summarize_and_save_to_excel_single_model(model_name, tokenizer_name, texts, references, save_to_excel=False,
                                             base_path='/content/drive/MyDrive/Bitirme/results/', device="cpu"):
    """
    Bir modelle özetleme yapar, DataFrame olarak döndürür ve isteğe bağlı olarak Excel'e kaydeder.

    Args:
        tokenizer_name (str): Tokenizer ismi (model ismiyle aynı kabul edilir).
        texts (list): Özetlenecek metinlerin listesi.
        references (list): Gerçek özetlerin listesi (texts ile aynı uzunlukta olmalı).
        save_to_excel (bool): Sonuçları Excel dosyasına kaydetmek için True yapın (varsayılan: False).
        base_path (str): Excel dosyasının kaydedileceği temel yol (varsayılan: '/content/drive/MyDrive/Bitirme/results/').
        device (str): Modeli çalıştırmak için kullanılacak cihaz ('cpu' veya 'cuda').

    Returns:
        pd.DataFrame: Özetler, olması gereken özetler ve modelin ürettiği çıktıları içeren bir DataFrame.
    """
    # texts ve references uzunluk kontrolü
    if len(texts) != len(references):
        raise ValueError("texts ve references aynı uzunlukta olmalıdır.")

    # Model ve tokenizer'ı yükle
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Sonuçları saklamak için bir DataFrame oluştur
    results_df = pd.DataFrame(columns=["Input Text", "Expected Summary", "Generated Summary"])

    # Tahminleri oluştur
    for text, reference in zip(texts, references):
        # Girişleri cihaz üzerine taşı
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        # Modelden çıktı al
        outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)

        # Çıktıyı CPU'ya taşı ve decode et
        summary = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)

        # DataFrame'e ekle
        results_df = pd.concat([
            results_df,
            pd.DataFrame({
                "Input Text": [text],
                "Expected Summary": [reference],
                "Generated Summary": [summary]
            })
        ], ignore_index=True)

    print(f"Model '{model_name}' için özetleme tamamlandı.")

    # Eğer Excel'e kaydetme seçeneği True ise
    if save_to_excel:
        # Model adını dosya ismi için uygun hale getir
        model_name_safe = model_name.replace("/", "_")
        today = datetime.today().strftime('%d%m%Y')
        file_path = f'{base_path}{today}_results_{model_name_safe}.xlsx'
        results_df.to_excel(file_path, index=False)
        print(f"Sonuçlar Excel'e kaydedildi: {file_path}")

    return results_df

def save_to_excel_single_model(model_name, tokenizer_name, texts, references, generated_texts, save_to_excel=False,
                                             base_path='/content/drive/MyDrive/Bitirme/results/', device="cpu"):
    """
    Bir modelle özetleme yapar, DataFrame olarak döndürür ve isteğe bağlı olarak Excel'e kaydeder.

    Args:
        model_name (str): Tokenizer ismi (model ismiyle aynı kabul edilir).
        texts (list): Özetlenecek metinlerin listesi.
        references (list): Gerçek özetlerin listesi (texts ile aynı uzunlukta olmalı).
        save_to_excel (bool): Sonuçları Excel dosyasına kaydetmek için True yapın (varsayılan: False).
        base_path (str): Excel dosyasının kaydedileceği temel yol (varsayılan: '/content/drive/MyDrive/Bitirme/results/').
        device (str): Modeli çalıştırmak için kullanılacak cihaz ('cpu' veya 'cuda').

    Returns:
        pd.DataFrame: Özetler, olması gereken özetler ve modelin ürettiği çıktıları içeren bir DataFrame.
    """
    # texts ve references uzunluk kontrolü
    if len(texts) != len(references) != len(generated_texts):
        raise ValueError("texts ve references aynı uzunlukta olmalıdır.")

        # DataFrame'e ekle
    results_df = pd.DataFrame({
        "Input Text": texts,
        "Expected Summary": references,
        "Generated Summary": generated_texts
    })

    print(f"Model '{model_name}' için özetleme tamamlandı.")

    # Eğer Excel'e kaydetme seçeneği True ise
    if save_to_excel:
        # Model adını dosya ismi için uygun hale getir
        model_name_safe = tokenizer_name.replace("/", "_")
        today = datetime.today().strftime('%d%m%Y')
        file_path = f'{base_path}{today}_results_{model_name_safe}.xlsx'
        results_df.to_excel(file_path, index=False)
        print(f"Sonuçlar Excel'e kaydedildi: {file_path}")

    return results_df


def balance_selected_categories(df, category_col, sample_size, selected_categories=None):
    """
    Belirli kategorilerden veya tüm kategorilerden örnekleme yaparak veri setini dengeler.

    Args:
        df (pd.DataFrame): Veri seti.
        category_col (str): Kategoriyi temsil eden sütun adı.
        sample_size (int): Her kategoriden alınacak maksimum örnek sayısı.
        selected_categories (list or None): Örnekleme yapılacak kategorilerin listesi.
                                            None ise tüm kategoriler hesaba katılır.

    Returns:
        pd.DataFrame: Dengelenmiş veri seti.
    """
    # Eğer selected_categories None ise tüm kategorileri kullan
    if selected_categories is None:
        selected_categories = df[category_col].unique()

    # Belirtilen kategorilere göre filtreleme
    filtered_df = df[df[category_col].isin(selected_categories)]

    # Kategorilere göre gruplandır ve örnekleme yap
    balanced_df = filtered_df.groupby(category_col, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), sample_size), random_state=42)
    )

    return balanced_df.reset_index(drop=True)


def upload_dataset_to_huggingface(dataset, repo_name, token):
    """
    Hugging Face Hub'a Dataset yükleme fonksiyonu.

    Args:
        dataset (Dataset): Hugging Face Dataset objesi.
        repo_name (str): Hugging Face repository ismi.
        token (str): Hugging Face erişim tokeni.
        private (bool): Repository'nin public/private ayarı. Varsayılan: False (public).

    Returns:
        str: Repository URL.
    """
    # Hugging Face Hub'a giriş yap
    login(token=token)

    # Dataset'i Hugging Face Hub'a push etmek
    dataset.push_to_hub(repo_id=repo_name)

    print(f"Dataset '{repo_name}' başarıyla Hugging Face Hub'a yüklendi!")
    return f"https://huggingface.co/datasets/{repo_name}"