[ ]
X_train = ["This was really awesome an awesome movie",
           "Great movie! Ilikes it a lot",
           "Happy Ending! Awesome Acting by hero",
           "loved it!",
           "Bad not upto the mark",
           "Could have been better",
           "really Dissapointed by the movie"]
# X_test = "it was really awesome and really disspntd"

y_train = ["positive","positive","positive","positive","negative","negative","negative"] # 1- Positive class, 0- negative class


[ ]
X_train # Reviews
account_circle
['This was awesome an awesome movie',
 'Great movie! Ilikes it a lot',
 'Happy Ending! Awesome Acting by hero',
 'loved it!',
 'Bad not upto the mark',
 'Could have been better',
 'Dissapointed by the movie']
Cleaning of the data
[ ]
# Tokenize
# "I am a python dev" -> ["I", "am", "a", "python", "dev"]
[ ]
from nltk.tokenize import RegexpTokenizer
# NLTK -> Tokenize -> RegexpTokenizer
[ ]
# Stemming
# "Playing" -> "Play"
# "Working" -> "Work"
[ ]
from nltk.stem.porter import PorterStemmer
# NLTK -> Stem -> Porter -> PorterStemmer

from nltk.corpus import stopwords
# NLTK -> Corpus -> stopwords
[ ]
# Downloading the stopwords
import nltk
nltk.download('stopwords')
account_circle
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
True
[ ]
tokenizer = RegexpTokenizer(r"\w+")
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()
[ ]
def getCleanedText(text):
  text = text.lower()

  # tokenizing
  tokens = tokenizer.tokenize(text)
  new_tokens = [token for token in tokens if token not in en_stopwords]
  stemmed_tokens = [ps.stem(tokens) for tokens in new_tokens]
  clean_text = " ".join(stemmed_tokens)
  return clean_text
Input from the user
[ ]
X_test = ["it was bad"]
[ ]
X_clean = [getCleanedText(i) for i in X_train]
xt_clean = [getCleanedText(i) for i in X_test]
[ ]
X_clean
account_circle
['awesom awesom movi',
 'great movi ilik lot',
 'happi end awesom act hero',
 'love',
 'bad upto mark',
 'could better',
 'dissapoint movi']
[ ]
# Data before cleaning
'''
X_train = ["This was awesome an awesome movie",
           "Great movie! Ilikes it a lot",
           "Happy Ending! Awesome Acting by hero",
           "loved it!",
           "Bad not upto the mark",
           "Could have been better",
           "Dissapointed by the movie"]
'''
account_circle

Vectorize
[ ]
from sklearn.feature_extraction.text import CountVectorizer
[ ]
cv = CountVectorizer(ngram_range = (1,2))
# "I am PyDev" -> "i am", "am Pydev"
[ ]
X_vec = cv.fit_transform(X_clean).toarray()
[ ]
X_vec
account_circle
array([[0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1,
        1, 0, 0, 1, 1, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 1, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0]])
[ ]
print(cv.get_feature_names())
account_circle
['act', 'act hero', 'awesom', 'awesom act', 'awesom awesom', 'awesom movi', 'bad', 'bad upto', 'better', 'could', 'could better', 'dissapoint', 'dissapoint movi', 'end', 'end awesom', 'great', 'great movi', 'happi', 'happi end', 'hero', 'ilik', 'ilik lot', 'lot', 'love', 'mark', 'movi', 'movi ilik', 'upto', 'upto mark']
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
[ ]
Xt_vect = cv.transform(xt_clean).toarray()
[ ]
Xt_vect
account_circle
array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0]])
Multinomial Naive Bayes
[ ]
from sklearn.naive_bayes import MultinomialNB
[ ]
mn = MultinomialNB()
[ ]
mn.fit(X_vec, y_train)
account_circle
MultinomialNB()
[ ]
y_pred = mn.predict(Xt_vect)
[ ]
y_pred
account_circle
array(['negative'], dtype='<U8')