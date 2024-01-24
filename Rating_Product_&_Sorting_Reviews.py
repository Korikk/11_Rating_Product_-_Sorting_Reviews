
import pandas as pd
import math
import scipy.stats as st
import datetime as dt

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_=pd.read_csv("datasets/df_sub.csv")
df=df_.copy()
df.head()
df.tail()
df.info()
df.shape

df.isnull().sum()

df["overall"].max()

#Raw Average of Ratings =4.587589013224822
df["overall"].mean()

df["reviewTime"]=pd.to_datetime(df['reviewTime'], dayfirst=True)
current_date = pd.to_datetime('2014-12-08 0:0:0')
df["day_diff"] = (current_date - df['reviewTime']).dt.days

a = df["day_diff"].quantile(0.25)
b = df["day_diff"].quantile(0.50)
c = df["day_diff"].quantile(0.75)

#Calculate weighted average based on time frames for the product
df.loc[df["day_diff"] <= a, "overall"].mean() * 28 / 100 + \
    df.loc[(df["day_diff"] > a) & (df["day_diff"] <= b), "overall"].mean() * 26 / 100 + \
    df.loc[(df["day_diff"] > b) & (df["day_diff"] <= c), "overall"].mean() * 24 / 100 + \
    df.loc[(df["day_diff"] > c), "overall"].mean() * 22 / 100

#Most useful 20 review


df["helpful_yes"] = df[["helpful"]].applymap(lambda x: x.split(",")[0].strip('[')).astype(int)
df["helpful_total_vote"] = df[["helpful"]].applymap(lambda x: x.split(",")[1].strip(']')).astype(int)
df["helpful_no"] = df["helpful_total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "helpful_total_vote", "reviewTime"]]

#positive and negative votes for each review.
def score_pos_neg_diff(pos, neg):
    return pos - neg


df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)


# ratio of positive votes for each review.
def score_average_rating(pos, neg):
    if pos - neg == 0:
        return 0
    return pos/(pos+neg)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

# Calculate wilson lower bound score

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)

df.sort_values("score_average_rating", ascending=False).head(20)


