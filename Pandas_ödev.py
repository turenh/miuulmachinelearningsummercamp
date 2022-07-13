""" Görev 1: Seaborn kütüphanesi içinden "Titanic" veri setini tanımlayınız """

import seaborn as sns
df=sns.load_dataset("titanic")
""" Görev 2: Titanic veri setindeki kadın ve erkek yolcu sayısını bulunuz"""

df["sex"].value_counts()
""" Görev 3: Herbir sütuna ait unique değerlerin sayısını bulunuz"""
for i in df.columns:
    df[i].nunique()
    print(f"{i}\nunique değer sayısı\n{df[i].nunique()}")
# Veya
df.nunique()
""" Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz. """

df["pclass"].nunique() or df["pclass"].value_counts()
""" Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz"""

df[["pclass","parch"]].value_counts()

""" Görev 6: "embarked deişkeninin veri tipine bakınız. Tipini kategori olarak değiştirip tekrar kontrol ediniz"""
df["embarked"]=pd
pd=pd.astype("category")
print(f"Veri tipi: {pd.dtype}")
df.info()


""" Görev 7: "embarked değeri "c" olanların tüm bilgilerini getir"""
df.loc[df["embarked"]=="C"]

# Görev 8: "embarked" değeri S olmayanların tüm bilgilerini getir.

df.loc[df["embarked"]!="S"]
# Görev 9: Yaşı 30'dan küçük ve kadın olan tüm yolcuların bilgilerini getir.

df.loc[(df["age"]<30) & (df["sex"]=="female")]

# Görev 10: Fare' i 500'den büyük veya yaşı 70'den büyük yolcuların bilgilerini gösteriniz.
df.loc[(df["fare"] > 500) | (df["age"]>70)]
# Görev 11: Her bir değişkendeki "null" boş değerlerin sayısını bulunuz.
df.isnull().sum()
# Görev 12: who değişkenini data frame'den çıkarınız.
df.drop("who", axis=1).head()
# Görev 13: deck değişkenindeki null değerleri deck değişkeninin en çok tekrar eden değişkeni ile doldur.

df["deck"].fillna(df["deck"].mode().iloc[0],inplace=True)# Birden fazla aynı frekansta mode değeri olduğu için bir seri dönüyor.
df["deck"].isna().sum()     # Boş değer sayısını gözlemlemek için yaptık görüldüğü gibi 0'a eşit olduğu görüldü çünkü hepsini doldurduk.
                            # Bu yüzden iloc[0] yaparak sıfırıncı indeksteki değeri seçmiş oluyoruz!!
""" Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz."""
df["age"].fillna(df["age"].median(), inplace=True)

""" Görev 15: survived değişkeninin pclass ve cinsiyet değişimi kırılımlarında,sum,count,mean değerlerini bulunuz."""

df.groupby(["sex","pclass"]).agg({"survived": ["mean","sum","count"]})

""" Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri
setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)."""

def yas_hesapla(yas):
    if yas <30:
        return 1
    else:
        return 0

df["age_flag"]=dfson["age"].apply(yas_hesapla)

""" Görev 17: Seaborn kütüphanesi içinde tips veri setini tanımlayınız"""
import seaborn as sns
dfnew=sns.load_dataset("tips")
""" Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerinin sum, min, max ve mean değerlerini bulunuz."""
dfnew.groupby("time").agg({"total_bill": ["sum","min","max","mean"]})
""" Görev 19: Day ve time’a göre total_bill değerlerinin sum, min, max ve mean değerlerini bulunuz"""
dfnew.groupby(["day","time"]).agg({"total_bill": ["sum","min","max","mean"]})
""" Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre sum, min, max ve mean değerlerini bulunuz."""

dfnew[["total_bill","tip","day"]].loc[(dfnew["time"]=="Lunch") & (dfnew["sex"]=="Female")].groupby("day").\
    agg({"total_bill":["sum","min","max","mean"],"tip":["sum","min","max","mean"]})
# VEYA
dfnew[(dfnew["time"]=="Lunch") & (dfnew["sex"]=="Female")].groupby("day").\
    agg({"total_bill":["sum","min","max","mean"],"tip":["sum","min","max","mean"]})


""" Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)"""
dfnew.loc[(dfnew["size"] < 3) & (dfnew["total_bill"]>10)].mean()
"""  Görev 22:  total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin. """
dfnew["total_bill_tip_sum"]=dfnew["total_bill"]+dfnew["tip"]

""" Görev 23: Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulunuz. Bulduğunuz ortalamaların altında olanlara 0, üstünde ve eşit
olanlara 1 verildiği yeni bir total_bill_flag değişkeni oluşturunuz.
Kadınlar için Female olanlarının ortalamaları, erkekler için ise Male olanların ortalamaları dikkate alınacktır. Parametre olarak cinsiyet ve total_bill
alan bir fonksiyon yazarak başlayınız. (If-else koşulları içerecek)
"""
dfnew["total_bill"].loc[dfnew["sex"] =="Female"].mean()
dfnew["total_bill"].loc[dfnew["sex"]=="Male"].mean()

def ortalama(x):
    a = dfnew["total_bill"].loc[dfnew["sex"] == "Female"].mean()
    b = dfnew["total_bill"].loc[dfnew["sex"] == "Male"].mean()
        if x["sex"] == "Female":
            if x["total_bill"] < a:
                return 0
            elif x["total_bill"] >= a:
                return 1
        else:
            if x["total_bill"] < b:
                return 0
            elif x["total_bill"] >= b:
                return 1

    def ortalama(x):
        a = dfnew["total_bill"].loc[dfnew["sex"] == "Female"].mean()
        b = dfnew["total_bill"].loc[dfnew["sex"] == "Male"].mean()
        if x["sex"] == "Female" and x["total_bill"] < a:
            return 0
        elif x["sex"] == "Female" and x["total_bill"] >= a:
            return 1
        elif x["sex"] == "Male" and x["total_bill"] < b:
            return 0
        elif x["sex"] == "Male" and x["total_bill"] >= b:
            return 1

dfnew ["total_bill_tip_sum1"] = dfnew.apply(ortalama, axis=1)
dfnew.head()

""" Görev 24:  total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyiniz"""
dfnew.loc[dfnew["total_bill_tip_sum1"==0].value_counts()
dfnew[["sex"]].loc[dfnew["total_bill_tip_sum1"]==0,].value_counts()



""" Görev 25:Veriyi total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız. """
dfnew["total_bill_tip_sum"].sort_values(ascending=False).head(30).head()
# Bu yalnızca seçilen sütunu sıralar oysa biz dataframe' i "total_bill_tip_sum" e göre sıralamak istediğimiz için aşağıdaki gibi ilerlemeliyiz.

dfnew.sort_values(by=["total_bill_tip_sum"],ascending=False).head(30).head()
# Veya
dfnew.sort_values("total_bill_tip_sum",ascending=False).head(30).head()


