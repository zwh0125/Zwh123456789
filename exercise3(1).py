import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt


def read_file():
    file_path = 'moby_dick.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def process_text(text):
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]

    tagged_tokens = nltk.pos_tag(filtered_tokens)

    return tagged_tokens


def count_pos(tagged_tokens):
    pos_counts = {}
    for token, pos in tagged_tokens:
        if pos in pos_counts:
            pos_counts[pos] += 1
        else:
            pos_counts[pos] = 1

    top_pos = FreqDist(pos_counts).most_common(5)

    return top_pos


def lemmatize_tokens(tagged_tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    tag_map = {
        'N': 'n',
        'V': 'v',
        'R': 'r',
        'J': 'a'
    }
    for token, pos in tagged_tokens:
        wn_pos = tag_map.get(pos[0])
        if wn_pos:
            lemma = lemmatizer.lemmatize(token, pos=wn_pos)
        else:
            lemma = lemmatizer.lemmatize(token)
        lemmas.append(lemma)

    top_lemmas = FreqDist(lemmas).most_common(20)

    return top_lemmas


def plot_frequency_distribution(tagged_tokens):
    pos_counts = {}
    for token, pos in tagged_tokens:
        if pos in pos_counts:
            pos_counts[pos] += 1
        else:
            pos_counts[pos] = 1

    fdist = FreqDist(pos_counts)
    fdist.plot(cumulative=False)
    plt.show()


def sentiment_analysis_with_textblob(text):
    blob = TextBlob(text)
    average_sentiment_score = blob.sentiment.polarity

    if average_sentiment_score > 0.05:
        sentiment = 'positive'
    elif average_sentiment_score < -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return average_sentiment_score, sentiment


def main():
    content = read_file()

    tagged_tokens = process_text(content)

    top_pos = count_pos(tagged_tokens)
    print("Top 5 parts of speech and their frequencies:")
    for pos, count in top_pos:
        print(f"{pos}: {count}")
    print()

    top_lemmas = lemmatize_tokens(tagged_tokens)
    print("Top 20 lemmas:")
    for lemma, count in top_lemmas:
        print(f"{lemma}: {count}")
    print()

    plot_frequency_distribution(tagged_tokens)

    average_sentiment_score, sentiment = sentiment_analysis_with_textblob(content)
    print(f"Average sentiment score using TextBlob: {average_sentiment_score}")
    print(f"Overall text sentiment using TextBlob: {sentiment}")


if __name__ == '__main__':
    main()