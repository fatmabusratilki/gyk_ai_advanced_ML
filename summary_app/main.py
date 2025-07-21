import nltk
import numpy as np
import re
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

text = """
Giant Rubber Duck Causes Traffic Chaos in Downtown Blumbleton
In an unexpected turn of events, a giant rubber duck appeared in the middle of Main Street in downtown Blumbleton early Tuesday morning.
The inflatable duck, estimated to be over 30 feet tall, was first spotted by local jogger Tina Marsh at 6:42 AM.
“I thought I was hallucinating,” Marsh told reporters. “It just sat there, staring at me with its giant plastic eyes.”
Authorities were baffled by the duck’s sudden arrival, as no permits had been issued for any large-scale duck installations.
Traffic came to a standstill for nearly four hours as drivers attempted to navigate around the massive yellow obstruction.
Local police attempted to deflate the duck but were unsuccessful due to its reinforced vinyl exterior.
Mayor Gerald Pickle called an emergency press conference, stating, “Blumbleton will not be held hostage by poultry-shaped inflatables.”
Children from nearby schools were delighted, calling it “the best field trip ever.”
Social media exploded with hashtags like #DuckGate and #QuackAttack trending worldwide.
Some conspiracy theorists believe the duck is a message from extraterrestrials.
Others claim it’s a guerrilla marketing stunt for a new bath product line.
The company “Bubble Bliss” has denied any involvement, though their stock rose 12% following the incident.
By noon, a team of engineers arrived with a crane and began the delicate process of relocation.
The duck was eventually moved to Blumbleton Park, where it now serves as a temporary tourist attraction.
Local vendors have already begun selling duck-themed merchandise.
City council is considering making the duck a permanent fixture.
Meanwhile, residents are divided — some love the whimsy, others demand answers.
A petition titled “Let the Duck Stay” has already gathered over 5,000 signatures.
Mayor Pickle has promised a full investigation into the duck’s origins.
For now, Blumbleton remains the only city in the world with a traffic-stopping rubber duck.
"""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    return text

cleaned_text = clean_text(text)

sentences = sent_tokenize(cleaned_text)

print(sentences)

vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(sentences)

sentence_scores = {}

for i in range(len(sentences)):
    score = X[i].toarray().sum()
    sentence_scores[sentences[i]] = score


# Use heapq to get the top 3 sentences
summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)
print("Summary of your text:")
print(summary)

# Turkish Version
print("\nTurkish Version:")

text_tr = """Dev Plastik Ördek, Blumbleton Merkezinde Trafik Felaketine Neden Oldu
Beklenmedik bir olay sonucu, Salı sabahı erken saatlerde Blumbleton şehir merkezindeki Main Street’in ortasında dev bir plastik ördek belirdi.
Yaklaşık 9 metre boyunda olduğu tahmin edilen şişme ördek, ilk olarak sabah 6:42’de koşuya çıkan yerel halktan Tina Marsh tarafından fark edildi.
“Halüsinasyon gördüğümü sandım,” dedi Marsh gazetecilere. “Dev plastik gözleriyle bana öylece bakıyordu.”
Yetkililer, ördeğin aniden ortaya çıkmasından şaşkına döndü; zira böyle büyük çaplı bir ördek yerleştirmesi için hiçbir izin alınmamıştı.
Sürücüler dev sarı engele takılıp yön değiştirmeye çalışırken, trafik neredeyse dört saat boyunca durma noktasına geldi.
Yerel polis ördeği söndürmeye çalıştı ancak dayanıklı vinil yapısı sebebiyle başarılı olamadı.
Belediye Başkanı Gerald Pickle acil basın toplantısı düzenleyerek “Blumbleton, kanatlı şekilli şişmelerin esiri olmayacak!” açıklamasında bulundu.
Çevredeki okullardan gelen çocuklar ise olaya bayıldı: “Bu en güzel okul gezisiydi!”
Sosyal medya çılgına döndü; #DuckGate ve #QuackAttack gibi etiketler dünya çapında trend oldu.
Bazı komplo teorisyenleri, ördeğin uzaylılardan bir mesaj olduğunu iddia etti.
Bazı kişilerse bunun yeni bir banyo ürünü için gerilla pazarlama yöntemi olduğunu öne sürdü.
“Bubble Bliss” şirketi olayla hiçbir ilgisi olmadığını söylese de, hisse senetleri %12 oranında arttı.
Öğlene doğru bir vinçle gelen mühendis ekibi, ördeği dikkatli biçimde taşımaya başladı.
Ördek sonunda Blumbleton Park’a yerleştirildi ve geçici bir turist cazibesine dönüştü.
Yerel satıcılar ördek temalı ürünler satmaya başladı bile.
Şehir meclisi, ördeği kalıcı bir simge haline getirmeyi değerlendiriyor.
Bu arada, şehir halkı ikiye bölündü — kimileri bu neşeli görüntüyü severken, kimileri cevap istiyor.
“Ördek Kalsın” başlıklı bir dilekçe, şimdiden 5000’den fazla imza topladı.
Belediye Başkanı Pickle, ördeğin kökenine dair tam bir soruşturma yapılacağını söz verdi.
Şimdilik, Blumbleton dünyada trafiği durduran plastik bir ördeğe sahip tek şehir olma unvanını taşıyor.
"""

cleaned_text_tr = clean_text(text_tr)

sentences_tr = sent_tokenize(cleaned_text_tr)

print(sentences_tr)

vectorizer = TfidfVectorizer(stop_words=stopwords.words('turkish'))
X = vectorizer.fit_transform(sentences_tr)

sentence_scores_tr = {}

for i in range(len(sentences_tr)):
    score_tr = X[i].toarray().sum()
    sentence_scores_tr[sentences_tr[i]] = score_tr


# Use heapq to get the top 3 sentences
summary_sentences_tr = heapq.nlargest(3, sentence_scores_tr, key=sentence_scores_tr.get)

summary_tr = ' '.join(summary_sentences_tr)
print("Summary of your text:")
print(summary_tr)

# Run in the FastAPI
# do parametric
#Parameters 1:text, 2:sentence number, default 3
# Evulution of the code: ROUGE analysis