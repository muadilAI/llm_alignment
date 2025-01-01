# İşleyiş

### Bir arayüz tasarlanacak
    Girdi olarak dosya linkini alacak.
    Verilen linkten dosyayı indirecek.
    Dosyadaki son Score column name inden sonra bir model adı tanımlandıysa 
        onun sağına Score column u oluşturup her çıktısı için 1 den 10 a skor üretecek.
    Bu işlem için aşağıdaki formatta llm e request atacak:
    """
    Score the summary of the given context above :
    Context:{prompt}\nSummary:{summary}\n Score the summary between 1 and 10. 
    Here is your score:
    """
    Çıktı hücreye yazılacak ve tüm çıktılar tamamlanınca dosyayı indirilebilir bir şekilde çıktı olarak verecek.




### Bunların yapılması için:
    ** Dosya okuma, dosyaya yazma ve dosyayı output olarak indirilebilir 
    bir obje verecek olan sınıf tanımlanacak.
    ** llm e request atmak, llm in outputunu parserlamak için class yazılacak. 
    ** ana işleyişlerin yazılacağı main dosyası yazılacak.
    ** arayüz dosyası yazılacak.



