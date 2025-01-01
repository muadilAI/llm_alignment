#measures the scores and save the responses to given google sheets for given ai models

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
# Tokenization için NLTK veri indirme
nltk.download('punkt')
nltk.download('punkt_tab')

import gc
import torch

import transformers
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
import os
import pandas as pd
import re
import inflect
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import single_meteor_score
from bert_score import score as bert_score
import pandas as pd
import numpy as np
import pickle
import gspread
pd.set_option('display.max_colwidth', None)

#verisetini dataframe olarak oku
df = pd.read_parquet("hf://datasets/Muadil/all_unique_cleaned_openai_summarize_comparisons_test/data/train-00000-of-00001.parquet")

def initialize_google_sheets():
    # Google Sheets API'ye erişim için gerekli izinleri ayarlıyoruz
    client = gspread.service_account(
        filename="/kaggle/input/service-account/august-oarlock-441317-t9-0a0e6dbc7072.json")

    # Google Sheet'i açıyoruz
    spreadsheet = client.open_by_url(
        'https://docs.google.com/spreadsheets/d/1in0Xvs-rFgF_ORVnfbKhu_LElumt1dKPeZ01kLtF4XY/edit?usp=sharing')

    # İlk sayfayı alıyoruz
    return spreadsheet.get_worksheet(0)


def sheets_to_df():
    worksheet = initialize_google_sheets()
    # Veriyi çekiyoruz ve pandas DataFrame'e çeviriyoruz
    data = worksheet.get_all_records()  # Bu, veriyi bir liste olarak alır
    df = pd.DataFrame(data)  # Veriyi DataFrame'e çeviriyoruz
    return df


def write_to_google_sheets(prompts, chosens, predictions, model_name):
    """
    Google Sheets'e verilen verileri yazar ve modelin sütun indeksini otomatik belirler.

    Args:
        prompts (list): Prompt verileri.
        chosens (list): Chosen verileri.
        predictions (list): Prediction verileri.
        model_name (str): Model adı (ilk satıra yazılır ve sütun indeksini belirler).
    """
    model_name = model_name.replace("-", "_").replace("/", "_")

    sheet_df = sheets_to_df()
    columns = sheet_df.columns.tolist()

    # Eksik sütunları ekle
    for col in ["Info", "Chosen", "Score_1"]:
        if col not in columns:
            sheet_df[col] = None

    # Yeni model sütununu ekle
    if model_name not in columns:
        print(model_name + "column initialized")
        sheet_df[model_name] = None
        # Model ile ilişkili skor sütununu ekle
        model_index = sheet_df.columns.get_loc(model_name)
        score_col = f"Score_{model_index}"
        sheet_df[score_col] = None

    # Satır satır ekleme işlemi
    for row_index, (prompt, chosen, prediction) in enumerate(zip(prompts, chosens, predictions), start=0):
        # Eğer satır yoksa yeni bir satır ekle
        if row_index >= len(sheet_df):
            sheet_df = pd.concat([sheet_df, pd.DataFrame([{col: None for col in sheet_df.columns}])], ignore_index=True)

        # Satırdaki değerleri kontrol et ve ekle
        if sheet_df.loc[row_index, "Info"] in [np.nan, None, ""]:
            sheet_df.loc[row_index, "Info"] = prompt

        if sheet_df.loc[row_index, "Chosen"] in [np.nan, None, ""]:
            sheet_df.loc[row_index, "Chosen"] = chosen

        if sheet_df.loc[row_index, model_name] in [np.nan, None, ""]:
            sheet_df.loc[row_index, model_name] = prediction
        # if pd.isna(sheet_df.loc[row_index, model_name]):
        #     sheet_df.loc[row_index, model_name] = prediction

    # Boş veya geçersiz verileri temizle
    sheet_df = sheet_df.replace([float("nan"), float("inf"), float("-inf")], None)

    # Google Sheets'e yaz
    headers = sheet_df.columns.tolist()
    updated_data = sheet_df.values.tolist()

    worksheet = initialize_google_sheets()
    worksheet.clear()
    worksheet.append_row(headers)
    worksheet.append_rows(updated_data)
    print("Veri başarıyla güncellendi.")


def calculate_summary_metrics(references, predictions):
    """
    Büyük dil modellerinde başarı metriklerini hesaplamak için fonksiyon.

    Args:
        references (list): Gerçek özetlerin listesi.
        predictions (list): Model tarafından üretilen özetlerin listesi.

    Returns:
        dict: Ortalama ROUGE, METEOR, BERTScore ve MoverScore metriklerini içeren bir sözlük.
    """
    if len(references) != len(predictions):
        raise ValueError("References ve predictions aynı uzunlukta olmalı!")

    # ROUGE skorları için scorer tanımlanıyor
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    meteor_scores = []

    # ROUGE ve METEOR hesaplamaları
    for ref, pred in zip(references, predictions):
        # ROUGE skorları
        rouge_scores = rouge_scorer_obj.score(ref, pred)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

        # METEOR skoru (cümleleri tokenize ediyoruz)
        tokenized_ref = nltk.word_tokenize(ref)
        tokenized_pred = nltk.word_tokenize(pred)
        meteor_scores.append(single_meteor_score(tokenized_ref, tokenized_pred))

    # BERTScore hesaplaması
    P, R, F1 = bert_score(predictions, references, lang="en", rescale_with_baseline=True)
    bert_f1 = F1.mean().item()

    # MoverScore hesaplaması
    # moverscores = get_moverscore(references, predictions)

    # Ortalama skorlar
    scores = {
        'ROUGE-1': sum(rouge1_scores) / len(rouge1_scores),
        'ROUGE-2': sum(rouge2_scores) / len(rouge2_scores),
        'ROUGE-L': sum(rougeL_scores) / len(rougeL_scores),
        'METEOR': sum(meteor_scores) / len(meteor_scores),
        'BERTScore': bert_f1,
        # 'MoverScore': sum(moverscores) / len(moverscores)
    }

    return scores


def add_metrics_to_dataframe(metrics_df, model_name, references, predictions):
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
    scores = calculate_summary_metrics(references, predictions)

    # Yeni bir satır olarak metrikleri eklemek için model adını ekle
    scores["Model Name"] = model_name
    metrics_df = pd.concat([metrics_df, pd.DataFrame([scores])], ignore_index=True)

    return metrics_df


def summarize_and_save_metrics_AutoModelForSeq2SeqLM(models, tokenizer_names, texts, references, metrics_df=None,
                                                     device="cpu", max_token_length=2048):
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
        metrics_df = pd.DataFrame(columns=["Model Name", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "BERTScore"])
        print("Yeni bir metrics_df oluşturuldu.")

    # texts ve references uzunluk kontrolü
    if len(texts) != len(references):
        raise ValueError("texts ve references aynı uzunlukta olmalıdır.")

    # Quantization için BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                             bnb_4bit_quant_type="nf4")
    # for model_idx, (model_name, tokenizer_name) in enumerate(zip(models, tokenizer_names), start=3):
    for model_name, tokenizer_name in zip(models, tokenizer_names):
        # Model ve tokenizer'ı yükle
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"  # GPU varsa otomatik cihaz ayarı
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"{model_name} yüklendi ve çıktılar alınıyor.")
        # Tahminleri oluştur
        predictions = []
        for text in texts:
            # # Girişleri cihaz üzerine taşı
            # inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_token_length).to(device)

            # # Modelden çıktı al
            # outputs = model.generate(inputs, max_length= max_token_length, num_beams=4, early_stopping=True)

            # # Çıktıyı CPU'ya taşı ve decode et
            # summary = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
            # predictions.append(summary)
            text = f"System: I want you to summarize this text\nDocument: {text}\nSummary:"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_token_length).to(device)
            output = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id, early_stopping=True, num_beams=2)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            if "Summary:" in prediction:
                prediction = prediction.split("Summary:")[1].strip()
            else:
                prediction = prediction  # Beklenmeyen durumlar için

            predictions.append(prediction)

            # Google Sheets'e yaz
        write_to_google_sheets(texts, references, predictions, model_name)

        # Başarı metriklerini DataFrame'e ekle
        metrics_df = add_metrics_to_dataframe(metrics_df, model_name, references, predictions)
        print(f"Model '{model_name}' için başarı metrikleri kaydedildi.")
        # Belleği temizle
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return metrics_df


# Modellerin listesi
models = [
    "Muadil/Llama-3.2-1B-Instruct_sum_DPO_1k_4_1ep",
    "Muadil/Llama-3.2-1B-Instruct_sum_DPO_10k_1_1ep",
]

# Tokenizer'ların listesi
tokenizer_names = [
    "Muadil/Llama-3.2-1B-Instruct_sum_DPO_1k_4_1ep",
    "Muadil/Llama-3.2-1B-Instruct_sum_DPO_10k_1_1ep",
]
# from google.colab import userdata
# secret_value_0 = userdata.get('HF_TOKEN')
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("HF_TOKEN")

from huggingface_hub import login

login(token=secret_value_0)

# Başarı metriklerini saklamak için boş bir DataFrame oluştur
metrics_df = pd.DataFrame(columns=["Model Name", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "BERTScore"])

# Fonksiyonu çalıştır ve metrikleri hesapla
metrics_df = summarize_and_save_metrics_AutoModelForSeq2SeqLM(
    models=models,
    tokenizer_names=tokenizer_names,
    texts=list(df.iloc[:10]["prompt"]),
    references=list(df.iloc[:10]["chosen"]),
    metrics_df=metrics_df,
    device="cuda",  # GPU kullanımı
)

# Sonuçları görüntüle
print(metrics_df)

from IPython.display import FileLink

# Kaggle'da dosyayı Pickle dosyası olarak kaydet
file_name = "output.pkl"
metrics_df.to_pickle(file_name)

# İndirme bağlantısını oluştur
display(FileLink(file_name))
