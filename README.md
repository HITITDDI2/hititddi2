<p align="center">
  <img src="https://github.com/HITITDDI2/hititddi2./blob/main/hititddi2-logo.png?raw=true" width="400" alt="logo">
</p>


# Yapay Zeka Destekli Metin Tespit Projesi
# Giriş:
### Takım Hakkında:

Bu proje, HİTİTDDİ2 tarafından geliştirilen Yapay zeka destekli metin espiti adlı bir uygulamadır. Ekibimiz, haber siteleri ve medya, akademik çalışmalar, ve sosyal medya platformlarında yanlış bilgilendirme ve manipülatif içerik tespitini sağlamak için bu projeyi oluşturmuştur.

### Takım Üyeleri
- *Danışman*: Dr. Öğr. Üyesi Emre DENİZ
- *Makine Öğrenimi Mühendisi*: Harun Emre Kıran
- *Yazılım Geliştirici ve Takım Kaptanı*: Rumeysa Gül Özdağ
- *Doğal Dil İşleme Uzmanı*: Sena ELif Telli 
- *Veri Mühendisi*: Senanur Okuducu


# Projenin Amacı:
Bu projenin amacı, yapay zeka tarafından üretilen makale özetlerini yüksek doğruluk oranı ile tespit etmektir. 
Giderek dijitalleşen dünyamızda, AI tarafından üretilen içeriklerin doğru bir şekilde tanınması, bilgi güvenilirliği ve doğruluğu açısından büyük önem taşımaktadır. 
Bu proje, AI tarafından yazılan makale özetlerini yüksek bir başarı oranı ile tespit etmeyi amaçlamaktadır. 
Giderek daha fazla içeriğin AI tarafından üretildiği günümüz dijital dünyasında, bu içeriklerin doğruluğunu 
ve güvenilirliğini değerlendirmek kritik bir hale gelmiştir. Proje, özellikle BERT, DistilBERT ve RoBERTa gibi 
ileri düzey doğal dil işleme (NLP) modellerinin, AI tarafından üretilen metinleri nasıl tespit edebileceğini 
incelemekte ve bu modellerin performanslarını karşılaştırmaktadır.

# Projenin Önemi
AI tarafından üretilen içeriklerin tespiti, gelecekte dijital medya ve yayıncılık alanında büyük bir rol oynayacaktır. 
Özellikle sosyal medya platformları, haber siteleri ve akademik yayınlar gibi bilgi dağıtım ağlarında, AI tarafından 
üretilen yanlış veya yanıltıcı bilgilerin hızla yayılmasını önlemek için güvenilir araçlara ihtiyaç duyulmaktadır. Bu 
proje, bu tür AI içeriklerini tespit etmek için kullanılan tekniklerin geliştirilmesine önemli bir katkı sağlamaktadır.

# Kullanılan Modeller ve Yaklaşımlar
BERT (dbmdz/bert-base-turkish-uncased): AI tarafından yazılmış metinleri tespit etmek için güçlü bir temel model olarak kullanılmıştır.
DistilBERT (distilbert-base-uncased): Daha hafif bir model olup, performans ve hız arasındaki dengeyi incelemek amacıyla değerlendirilmiştir.
RoBERTa (roberta-base): Güçlü bir transfer öğrenme modeli olarak, derinlemesine metin analizi için kullanılmıştır.

# Projenin Gelecekteki Etkisi
Bu proje, AI tarafından üretilen içeriklerin doğruluk ve güvenilirlik açısından nasıl değerlendirilebileceğine dair önemli bulgular sunmaktadır. 
Gelecekte, bu tür tespit yöntemleri, medya okuryazarlığı, bilgi doğrulama ve dijital güvenlik alanlarında kritik bir rol oynayacaktır. Ayrıca, 
akademik araştırmalarda AI içeriklerinin tespiti, intihal ve etik dışı davranışların önlenmesi için vazgeçilmez hale gelebilir.

# Kurulum ve Gereksinimler
Projenin çalışması için aşağıdaki gereksinimlerin karşılanması gerekmektedir:

Python 3.x
Gerekli kütüphanelerin kurulumu:
```
pip install -r requirements.txt
```

Projenin Çalıştırılması
Verilerin Yüklenmesi ve Ön İşleme:

Verilerin temizlenmesi ve modellerin eğitilmesi için gerekli olan ön işleme adımları gerçekleştirilir.
Modellerin Eğitilmesi ve Değerlendirilmesi:

BERT, DistilBERT ve RoBERTa modelleri eğitilir ve doğrulama veri setleri üzerinde değerlendirilir.
Sonuçlar, results/results.json dosyasında saklanır.
```
python src/train.py
```
Sonuçların Görselleştirilmesi:
```
Modellerin performansını karşılaştırmak ve sonuçları analiz etmek için grafikler oluşturulur ve results/ klasörüne kaydedilir.
```
python src/evaluate.py
```
#Acikhack2024TDDİ
