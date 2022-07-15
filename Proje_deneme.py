#Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
# Görev 1: Aşağıdaki Soruları Yanıtlayınız
#Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
import pandas as pd
df=pd.read_csv("https://raw.githubusercontent.com/brktlhylmz/Miuul-Yaz-Kampi/main/persona.csv")
df.info()
# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].value_counts()
# Soru 3: : Kaç unique PRICE vardır?
df["PRICE"].nunique()
# Soru 4:Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()
# Soru 5:Hangi ülkeden kaçar tane satış olmuş?
df.groupby("COUNTRY").agg({"PRICE":"count"})
# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY").agg({"PRICE":"sum"})
# Soru 7: SOURCE türlerine göre satış sayıları nedir?
df.groupby("SOURCE").agg({"PRICE":"count"})
# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE":"mean"})
# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE").agg({"PRICE":"mean"})
# Soru 10:  COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["SOURCE","COUNTRY"]).agg({"PRICE":"mean"})

# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).apply(lambda x:x.mean(),axis=1).head()

# Görev 3: Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız ve elde edilen çıktıyı agg_df olarak kaydediniz.

df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).sort_values("PRICE",ascending=False)

agg_dff=df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).sort_values("PRICE",ascending=False)

# Görev 4: Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir. Bu isimleri değişken isimlerine çeviriniz.

agg_dff=agg_dff.reset_index()

# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
# • Aralıkları ikna edici şekilde oluşturunuz.
# • Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'
agg_dff["AGE"]=agg_dff["AGE"].astype("category")
agg_dff["AGE_CAT"]=pd.cut(agg_dff["AGE"], [0,18,23,30,40,70], labels=["0_18","19_23","24_30","31_40","41_70"])
agg_dff.drop("AGE_CAT",axis=1,inplace=True)

# Görev 6:
#Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
#• Yeni eklenecek değişkenin adı: customers_level_based
#• Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.

# Hint: Dikkat! List comprehension ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18. Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

def kazanc(x):
    return x["COUNTRY"].upper()+"_"+x["SOURCE"].upper() + "_"+x["SEX"].upper() + "_"+x["AGE_CAT"]


agg_dff["customers_level_based"] = agg_dff.apply(kazanc,axis=1)
agg_dff.columns


agg_dff=agg_dff.groupby("customers_level_based").agg({"PRICE":"mean"}) # Customers_level_based değerlerine göre groupby() yapıp , her bir unique değerin price ortalamasını aldık ve agg_dff e kaydettik.
agg_dff=agg_dff.reset_index() # Groupby sonucunda indexler kaydığı için reset index yaptıkve bunu da agg_dff 'e kaydettik.


# Görev 7:Yeni müşterileri (personaları) segmentlere ayırınız.
#Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.

pd.qcut(agg_dff["PRICE"], 4 ,labels=["D", "C", "B", "A"])
#• Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.

agg_dff["SEGMENT"]=pd.qcut(agg_dff["PRICE"], 4 ,labels=["D", "C", "B", "A"])
#• Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).

agg_dff.groupby("SEGMENT").agg({"PRICE":["mean","max","sum"]})
# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
#33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_19_23"
# Gelir tahmini yapan bir fonksiyon yazalım:
def gelir_tahmin(x):
    return agg_dff[agg_dff["customers_level_based"] == x]
#• 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user1="TUR_ANDROİD_FEMALE_31_40"
gelir_tahmin(new_user1)

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user2="FRA_IOS_FEMALE_31_40"
gelir_tahmin(new_user2)



