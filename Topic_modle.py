# import vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# import numpy for matrix operation
import numpy as np
import sys
import csv
import Vis


def vec(clean_corpus):
    # Converting text into numerical representation
    tf_idf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)

    # Converting text into numerical representation
    cv_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)

    tf_idf_arr = tf_idf_vectorizer.fit_transform(clean_corpus)
    #print(tf_idf_arr)
    # Array from Count Vectorizer
    cv_arr = cv_vectorizer.fit_transform(clean_corpus)

    vocab_tf_idf = tf_idf_vectorizer.get_feature_names()
    vocab_cv = cv_vectorizer.get_feature_names()

    return (cv_arr, vocab_cv,cv_vectorizer, tf_idf_arr, vocab_tf_idf)


def LDA(data, vocab_tf_idf,vector, n):

        # Implementation of LDA:

        # Create object for the LDA class
        # Inside this class LDA: define the components:
        lda_model = LatentDirichletAllocation(n_components=n, learning_decay = .5, max_iter=20, random_state=10)

        # fit transform on model on our count_vectorizer : running this will return our topics
        X_topics = lda_model.fit_transform(data)
        Vis.extract_lda_model(X_topics, data, n, vector, True, True)


        # .components_ gives us our topic distribution
        topic_words = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]


        #  Define the number of Words that we want to print in every topic : n_top_words
        n_top_words = 100

        for i, topic_dist in enumerate(topic_words):
            print(sum(topic_dist))
            # np.argsort to sorting an array or a list or the matrix acc to their values

            sorted_topic_dist = np.argsort(topic_dist)
            print(sorted_topic_dist)

            # Next, to view the actual words present in those indexes we can make the use of the vocab created earlier
            topic_words = np.array(vocab_tf_idf)[sorted_topic_dist]
            topic_words_prob = np.array(topic_dist)[sorted_topic_dist]
            # so using the sorted_topic_indexes we ar extracting the words from the vocabulary
            # obtaining topics + words
            # this topic_words variable contains the Topics  as well as the respective words present in those Topics
            topic_words = topic_words[:-n_top_words:-1]
            topic_words_prob = topic_words_prob[:-n_top_words:-1]
            normilized = [int(round(element, 5)*100000) for element in topic_words_prob]
            #normilized =  [element * 10 for element in topic_words_prob]
            filename = "topic" + str(i+1) + "temp.csv"
            print("Topic", str(i + 1), topic_words, topic_words_prob)
            with open(filename, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)

                writer.writerow(topic_words)
                writer.writerow((topic_words_prob))
                writer.writerow((normilized))


            # iterating over ever value till the end value
            movies = [[] for x in range(n)]
            non = []

        counts = [0]*n
        for m in range(X_topics.shape[0]):
            # argmax() gives maximum index value
            topic_doc = X_topics[m].argmax()
            # document is n+1
            counts[topic_doc] += 1
            movies[topic_doc].append(m)
            print("Document", m + 1, " -- Topic:",  topic_doc)
        print(counts)
        num = 0
        for j in range(len(counts)):
            if counts[j] < 20:
                num +=1
                non.extend(movies[j])
                X_topics = np.delete(X_topics, j-num, axis=1)
        ran = list(range(len(X_topics)))
        X_topics = [X_topics[i] for i in ran if i not in non]

        return X_topics, non
