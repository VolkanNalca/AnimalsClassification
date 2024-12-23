## Kod Hücrelerinin Açıklamaları

Bu bölüm, `hayvan_siniflandirma.ipynb` dosyasındaki her hücrenin ne işe yaradığını ve nasıl çalıştığını açıklamaktadır.

### Hücre 1: Kütüphanelerin Yüklenmesi

**Açıklama:** Bu hücrede, proje boyunca kullanılacak olan Python kütüphaneleri yüklenir. Ayrıca, `utils.py` dosyasındaki `get_manipulated_images` (görüntü manipülasyonu için) ve `get_wb_images` (renk sabitliği için) fonksiyonları import edilir.

### Hücre 2: Sabit Değerlerin Tanımlanması

**Açıklama:** Bu hücrede, veri setinin bulunduğu ana dizin (`ANA_DIZIN`), kullanılacak hayvan sınıfları (`SECILEN_SINIFLAR`) ve resimlerin yeniden boyutlandırılacağı boyut (`RESIM_BOYUTU`) gibi sabit değerler tanımlanır. **Not:** `ANA_DIZIN` değişkenini kendi veri setinizin yolu ile güncellemeyi unutmayın.

### Hücre 3: Resim Yeniden Boyutlandırma ve Normalleştirme Fonksiyonu

**Açıklama:** Bu hücrede, `resmi_yeniden_boyutlandir_ve_normallestir` adlı bir fonksiyon tanımlanır. Bu fonksiyon, verilen bir resim yolundaki resmi okur, belirlenen boyuta yeniden boyutlandırır ve piksel değerlerini 0 ile 1 arasına normalize eder (min-max normalleştirme).

### Hücre 4: Seçilen Sınıflara Ait Resimlerin Toplanması

**Açıklama:** Bu hücrede, `ANA_DIZIN` altında bulunan ve `SECILEN_SINIFLAR` listesinde belirtilen sınıflara ait resimler, `resmi_yeniden_boyutlandir_ve_normallestir` fonksiyonu kullanılarak okunur, yeniden boyutlandırılır ve normalize edilir. Her sınıftan **ilk 650 resim** alınır. Resimler ve etiketler `veri` isimli bir sözlükte (dictionary) saklanır ve daha sonra bu sözlük bir `pandas` DataFrame'ine (`veri_cercevesi`) dönüştürülür.

### Hücre 5: Veri Setinin İncelenmesi

**Açıklama:** Bu hücrede, oluşturulan `veri_cercevesi` DataFrame'inin ilk 5 satırı ve her sınıfa ait resim sayısı (`value_counts()`) ekrana yazdırılır.

### Hücre 6: Veri Setinin Eğitim ve Test Olarak Ayrılması

**Açıklama:** Bu hücrede, veri seti eğitim (%70) ve test (%30) olmak üzere ikiye ayrılır.

*   `X` değişkeni resimleri (giriş verileri), `y` değişkeni ise etiketleri (çıkış verileri) tutar.
*   Etiketler (sınıf isimleri) önce `LabelEncoder` ile sayısal değerlere dönüştürülür, sonra `to_categorical` fonksiyonu ile one-hot encoding formatına çevrilir.
*   `train_test_split` fonksiyonu, veriyi eğitim ve test setlerine ayırırken `random_state=42` parametresi ile her çalıştırmada aynı şekilde bölünmesini sağlar. `stratify=y_one_hot` parametresi ise her iki sette de sınıf oranlarının korunmasını sağlar.

### Hücre 7: Veri Çoğaltma (Data Augmentation) Ayarlarının Yapılması ve Fonksiyonun Oluşturulması

**Açıklama:** Bu hücrede, eğitim veri setini genişletmek için kullanılacak veri çoğaltma (data augmentation) ayarları `ImageDataGenerator` nesnesi ile yapılır. Farklı dönüşümler (döndürme, kaydırma, yakınlaştırma, vb.) ve bunların parametreleri belirlenir. Ayrıca, `tf.data.Dataset` API'sini kullanarak, `apply_wb_and_augment` fonksiyonunu her resme uygulayan bir veri akışı oluşturan `custom_data_generator` fonksiyonu tanımlanır.

### Hücre 8: Veri Çoğaltma Örneklerinin Gösterilmesi (İsteğe Bağlı)

**Açıklama:** Bu hücrede, veri çoğaltma işleminin nasıl çalıştığını görselleştirmek için, eğitim setinden bir örnek alınır ve uygulanan dönüşümler gösterilir.

### Hücre 9: CNN Modelinin Oluşturulması ve Derlenmesi

**Açıklama:** Bu hücrede, Keras `Sequential` API'si kullanılarak bir Evrişimli Sinir Ağı (CNN) modeli oluşturulur. Model, 4 evrişim katmanı (sırasıyla 32, 64, 128 ve 256 filtreli), her evrişim katmanından sonra bir ReLU aktivasyon fonksiyonu, bir maksimum havuzlama (max pooling) katmanı ve bir dropout katmanı içerir. Evrişim katmanlarından sonra, özellik haritaları düzleştirilir (flatten) ve 2 tam bağlantılı (dense) katman (512 ve 10 nöronlu) eklenir. Son katman, 10 sınıflı sınıflandırma için softmax aktivasyon fonksiyonu kullanır. Model, `categorical_crossentropy` kayıp fonksiyonu, `Adam` optimizasyon algoritması ve `accuracy` (doğruluk) metriği ile derlenir. `model.summary()` ile modelin bir özeti ekrana yazdırılır. Ek olarak, `EarlyStopping` callback'i tanımlanır, bu, doğrulama kaybı belirli bir süre boyunca iyileşmediğinde eğitimi durdurur ve en iyi modeli geri yükler.

### Hücre 10: Modelin Eğitilmesi

**Açıklama:** Bu hücrede, oluşturulan CNN modeli, `custom_data_generator` fonksiyonu ile sağlanan veri çoğaltma uygulanmış veri akışı kullanılarak eğitilir. Eğitim `epochs=100` ile 100 epoch boyunca sürer, ancak `EarlyStopping` callback'i doğrulama kaybı 10 epoch boyunca iyileşmediğinde eğitimi erken durduracaktır. `steps_per_epoch`, her epoch'ta işlenecek batch sayısını belirtir ve veri çoğaltma da dikkate alınarak `(len(X_train) * 2) // 32` olarak hesaplanır. `model.save()` ile eğitilmiş model `hayvan_siniflandirma_modeli.keras` dosyasına kaydedilir.

### Hücre 11: Eğitim ve Doğrulama Sonuçlarının Görselleştirilmesi

**Açıklama:** Bu hücrede, eğitim ve doğrulama sürecindeki kayıp (loss) ve doğruluk (accuracy) değerleri grafikler halinde çizdirilir. Bu grafikler, modelin eğitim sürecini ve performansını değerlendirmek için kullanılır.

### Hücre 12: Modelin Test Seti Üzerinde Değerlendirilmesi

**Açıklama:** Bu hücrede, eğitilmiş model test seti üzerinde değerlendirilir. `model.evaluate()` fonksiyonu ile test kaybı ve doğruluğu hesaplanır. `model.predict()` ile test setindeki resimler için tahminler elde edilir. `classification_report` ile her sınıf için hassasiyet (precision), duyarlılık (recall), F1-skoru ve destek (support) değerleri içeren bir sınıflandırma raporu oluşturulur. `confusion_matrix` ile modelin tahminleri ve gerçek sınıflar arasındaki ilişkiyi gösteren bir karmaşıklık matrisi (confusion matrix) oluşturulur ve `sns.heatmap` ile görselleştirilir.

### Hücre 13: Test Resimlerinin Işık Manipülasyonlarının Yapılması

**Açıklama:** Bu hücrede, test setindeki resimlere `get_manipulated_images` fonksiyonu kullanılarak ışık manipülasyonları (gama düzeltmesi ve histogram eşitleme) uygulanır. Her resim için gama düzeltmesi uygulanmış (gamma=0.5 ve gamma=1.5) ve histogram eşitlemesi uygulanmış versiyonları oluşturulur. Dönüştürülmüş resimler `X_test_manipulated` listesine, etiketleri ise `y_test_manipulated` listesine eklenir.

### Hücre 14: Modelin Manipüle Edilmiş Test Setiyle Test Edilmesi

**Açıklama:** Bu hücrede, eğitilmiş model, ışık manipülasyonu uygulanmış test seti (`X_test_manipulated`) üzerinde değerlendirilir. `model.evaluate()` ile kayıp ve doğruluk değerleri hesaplanır. Sınıflandırma raporu ve karmaşıklık matrisi oluşturularak modelin performansı detaylı bir şekilde incelenir.

### Hücre 15: Renk Sabitliği Algoritmasının Uygulanması

**Açıklama:**  Bu hücrede, ışık manipülasyonu uygulanmış test setindeki resimlere `get_wb_images` fonksiyonu kullanılarak Simple White Balance renk sabitliği algoritması uygulanır. Elde edilen resimler `X_test_wb` listesine eklenir ve `wb_images_dir` klasörüne kaydedilir.

### Hücre 16: Modelin Renk Sabitliği Uygulanmış Test Setiyle Test Edilmesi

**Açıklama:** Bu hücrede, eğitilmiş model, renk sabitliği uygulanmış test seti (`X_test_wb`) üzerinde değerlendirilir. Kayıp ve doğruluk değerleri hesaplanır, sınıflandırma raporu ve karmaşıklık matrisi oluşturulur.

### Hücre 17: Sonuçların Karşılaştırılması ve Raporlama

**Açıklama:** Bu hücrede, üç farklı test seti (orijinal, ışık manipülasyonlu, renk sabitliği uygulanmış) üzerindeki model performansları bir `pandas` DataFrame'inde (`results`) özetlenir ve ekrana yazdırılır. Ayrıca, sonuçlar bir bar grafiği ile görselleştirilir. Son olarak, sonuçlar bir metin rapor halinde sunulur ve renk sabitliği uygulanmış test setindeki doğruluk değeri 0.6'dan düşükse olası çözüm yolları listelenir.

**Rapor şu bilgileri içerir:**

*   Her test seti için doğruluk (accuracy) ve kayıp (loss) değerleri.
*   Işık manipülasyonunun model performansına etkisi.
*   Renk sabitliği algoritmasının model performansına etkisi.
*   Renk sabitliği uygulanmış test setindeki doğruluk düşükse, olası çözüm önerileri.

Bu şekilde, her hücrenin işlevini ve projenin genel akışını daha net bir şekilde anlamış olacaksınız.
