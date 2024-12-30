from sklearn.feature_extraction.text import CountVectorizer

# Convert the labels into binary values: spam = 1, ham = 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Create a CountVectorizer object to convert text into numerical vectors
vectorizer = CountVectorizer(stop_words='english')  # Removing common words (stopwords)

# Convert the messages into a matrix of token counts
X = vectorizer.fit_transform(df['message'])

# Labels (spam or ham)
y = df['label']
