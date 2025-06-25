import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
import numpy as np
from PIL import Image
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer 
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Open the text
text_file = open("new text_1.txt")
text = text_file.read()

# print(type(text))
# print("\n")

# print(text)
# print("\n")

# print(len(text))

# ---- Sentences tokenizing
sentences = sent_tokenize(text)
# print(len(sentences))
# print(sentences)

#  ---- Word tokenizing
words = word_tokenize(text)
# print(len(words))
# print(words)

#  ---- Find the frequency
# fdist = FreqDist(words)
# print(fdist.most_common(10))

#  ---- Plot the frequency graph
# fdist.plot(10)
# plt.show()

#  ---- Remove punctuation marks.

words_no_punc = []
# -- Removing punctuation marks :
# for w in words:
#     if w.isalpha():
#         words_no_punc.append(w.lower())

# print(words_no_punc)
# print("\n")
# print(len(words_no_punc))

fdist = FreqDist(words_no_punc)

# fdist.most_common(10)
# print(fdist)

# fdist.plot(10)
# plt.show()

#  --- List of stopwords
stopwords = stopwords.words("english")
# print(stopwords)

# wordcloud = WordCloud().generate(text)

# plt.figure(figsize = (6, 10))
# plt.imshow(wordcloud)

# plt.axis("off")
# plt.show()

char_mask = np.array(Image.open("circle.png"))

wordcloud = WordCloud(background_color="black", 
    mask=char_mask).generate(text)

# plt.figure(figsize = (8,8))
# plt.imshow(wordcloud)

# plt.axis("off")
# plt.show()

# ------ Stemming ---------

# porter  = PorterStemmer()
# word_list = ["Study", "Studying", "Studies", 
#              "Studied"]

# for w in word_list:
#     print(porter.stem(w))

# porter = PorterStemmer()
# word_list = ["studies", "leaves", "decreases", "plays"]
# for w in word_list:
#     print(porter.stem(w))

# print(SnowballStemmer.languages)

#  ----- Lemmatizer with default PoS value
lemma = WordNetLemmatizer()
# word_list = ["studies", "leaves", "decreases", "plays"]
# for w in word_list:
#     print(lemma.lemmatize(w))
    
# word_list = ["am", "is", "are", "was", "were"]
# for w in word_list:
#     print(lemma.lemmatize(w, pos="v"))

sentence = "A very beautiful young lady is walking on the beach"
tokenized_words = word_tokenize(sentence)
for words in tokenized_words:
    tagged_words = nltk.pos_tag(tokenized_words)
# print(tagged_words)

# grammar = "NP : {<DT>?<JJ>*<NN>}"
# parser = nltk.RegexpParser(grammar)

# output = parser.parse(tagged_words)
# print (output)
# output.draw()

# grammar = r""" NP: {<.*>+}

# }<JJ>+{"""

# parser = nltk.RegexpParser(grammar)
# output = parser.parse(tagged_words)
# print(output)

# output.draw()

# sentence = "Mr.Smith made a deal on a a beach of Switzerland near WHO"
# tokenized_words = word_tokenize(sentence)
# nltk.download('maxent_ne_chunker_tab')

# for w in tokenized_words:
#     tagged_words = nltk.pos_tag(tokenized_words)
# N_E_R = nltk.ne_chunk(tagged_words,  binary=False)
# print(N_E_R)
# N_E_R.draw()

# word = wordnet.synsets("Play")[0]
# print(word.name())
# print(word.definition())
# print(word.examples())

sentences = ["Jim and Pam travelled by the bus:",
             "The train was late",
             "The flight was full. Travelling by flight is expensive"]
cv = CountVectorizer()
B_O_W = cv.fit_transform(sentences).toarray()
print(cv.vocabulary_)
print("\n")

print(cv.get_feature_names_out())
print("\n")

print(B_O_W)