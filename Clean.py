import spacy
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import enchant

dict = enchant.Dict("en_US")
nltk.download('wordnet')
import string

from names_dataset import NameDataset

nd = NameDataset()

# to suppress warnings
from warnings import filterwarnings

filterwarnings('ignore')

# import LDA from sklearn
nlp = spacy.load('en_core_web_sm')

# stop loss words
stop = set(stopwords.words('english'))

# punctuation
exclude = set(string.punctuation)
other = ['—', '“', '…', '’', '–', '·', '’', ]
for c in other:
    exclude.add(c)
# lemmatization
lemma = WordNetLemmatizer()
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()


def getDic():
    dictionary = {}
    with open("wordlist.txt") as file:
        for line in file:
            key = line.rstrip().split("\n")[0]

            dictionary[key] = 1

    return dictionary


def getMovieLines():
    filename = "cornell movie-dialogs corpus//movie_lines.txt"
    b = "m0"
    temp = []
    docs = []
    with open(filename) as file:
        for line in file:
            temp.append(line.rstrip().split("+++$+++"))
        # print(temp)
    lines = []
    for i in range(len(temp)):
        a = temp[i][2].strip().rstrip().lower()
        if (a != b):
            b = a
            doc = ' '.join(lines)
            docs.append(doc)
            lines = []
        lines.append(temp[i][4].strip().rstrip().lower())
    doc = ' '.join(lines)
    docs.append(doc)
    print(len(docs))
    return docs


# Apply Preprocessing on the Corpus
def clean(doc):
    # convert text into lower case + split into words
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])

    # remove any stop words present
    punc_free = ''.join([(ch if ch not in exclude else ' ') for ch in stop_free])

    port = [porter.stem(word) for word in punc_free.split()]

    #lan = [lancaster.stem(word) for word in port]

    # remove punctuations + normalize the text
    normalized = " ".join(lemma.lemmatize(word) for word in port)

    return normalized


def good_words():
    # Use a breakpoint in the code line below to debug your script.
    words = getDic()

    corpus = getMovieLines()

    clean_corpus = [clean(doc).split() for doc in corpus]
    bad = ['an', 'em', 'know', 'you', 'like', 'paul', 'erica', 'jean', 'teddy', 'tommy', 'seems', 'jacob', 'dreyfus',
           'catherine', 'luther', 'smith', 'seem', 'andrew', 'austin', 'that', 'ready', 'wilkins', 'shall', 'bye',
           'wow', 'it', 'jenny', 'me', 'dan', 'felix', 'have', 've', 'lancelot', 'perhaps', 'can', 'mr', 'he', 'it',
           'one', 'know', 'you', 'felice', 'felicia', 'felicity',
           're', 'say', 'it', 'you', 'brad', 'what', 'know', 'josh', 'that', 'he', 'tell', 'told', 'this', 'get',
           'ah', 'mister', 'and', 'call', 'sure', 'is', 'put', 'in',
           'john', 'david', 'mr', 'jason', 'james', 'leo', 'dewey', 'baldwin', 'of', 'johnson', 'as', 'give', 'ten',
           'yet', 'de', 'will', 'clarence', 'ray', 'heather', 'al', 'patrick', 'billy', 'martin', 'sure', 'jimmy',
           'where', 'time', 'harper', 'cooper', 'okay', 'on', 'see', 'la', 'casablanca', 'gotham', 'joe', 'mary',
           'paul', 'daniel', 'bruce', 'ni', 'domino', 'rob', 'you', 'four', 'thing', 'that', 'it', 'me', 've', 'want',
           'what', 'really', 'nick', 'yah', 'sal',
           'jerry', 'he', 're', 'here', 'jonah', 'sally', 'before', 'pm', 'tyler', 'terry', 'jordan', 'judy', 'make',
           'bob', 'rather', 'thou', 'ted', 'skipper', 'back', 'still', 'veronica', 'them', 'thy', 'guy', 'wilson',
           'something', 'two', 'romeo', 'thee', 'up', 'said', 'do', 'there', 'tom', 'never', 'take', 're', 'we',
           'george', 'bailey', 'potter', 'ah', 'fucking', 'fuck',
           'yeah', 'huh', 'the', 'shit', 'charley', 'lee', 'genesis', 'though', 'thomas', 'peter', 'lucy', 'nine',
           'norman', 'not', 'out', 'elizabeth', 'homer', 'maybe', 'need', 'even', 'ha', 'beckett', 'barry', 'edward',
           'vanessa', 'morpheus', 'ti', 'huh', 'heh', 'craig', 'frank', 'benjamin', 'kirk', 'ye', 'aye', 'mar', 'th',
           'either', 'penelope',
           'richard', 'buzz', 'yeh', 'ruth', 'sam', 'for', 'woody', 'more', 'scott', 'now', 'to', 'look', 'jack',
           'lot', 'ever', 'wan', 'claude', 'logan', 'robert', 'victor', 'good', 'alex', 'am', 'take', 'charles',
           'yeah', 'magneto' 'cool', 'well', 'it', 'that', 'know', 'oh', 'no', 'what', 'we', 'would', 'rick', 'going',
           'her', 'come', 'you', 'brian', 'let', 'ben', 'elaine', 'leda', 'simon', 'so', 'robinson', 'er', 'joseph',
           'st', 'henry',
           'betty', 'of', 'im', 'karen', 'yo', 'adam', 'at', 'my', 'glenn', 'dylan', 'soon', 'jones', 'if', 'juno',
           'jack', 'every', 'new', 'reynard', 'day', 'obi', 'first', 'hey', 'cannot', 'maya', 'ok', 'keep', 'big',
           'long', 'tony', 'don', 'kat', 'luke', 'lex', 'kent', 'thea', 'harry', 'rc', 'juliet', 'want', 'victoria',
           'll', 'yes', 'frank', 'could', 'they', 'think', 'way', 'right', 'sir', 'must', 'eve', 'miss', 'man', 'dr',
           'men', 'much', 'him', 'go', 'they', 'mean', 'sorry,' 'little', 'talk', 'anything', 'thought', 'year',
           'night',
           'people', 'life', 'great', 'home', 'feel', 'guess', 'nothing',
           'kid', 'last', 'everything', 'help', 'kind', 'money', 'mean', 'little', 'around', 'talk', 'anything',
           'year', 'people',
           'work', 'place', 'better', 'nothing', 'girl','penni', 'kid', 'car', 'three', 'who', 'job',
           'five', 'bad', 'old', 'night', 'listen', 'mean', 'sorry', 'little', 'thought', 'great', 'people', 'year',
           'feel', 'talk', 'anything', 'night', 'others', 'always', 'guess',
           'please', 'nice', 'dad', 'rand',  'argon', 'everything', 'last', 'god', 'wait', 'suppose', 'fine',
           'meredith', 'dixie', 'hi', 'remember', 'believe', 'thanks',
           'real', 'hell', 'talking', 'stop', 'along', 'sue', 'wanted', 'mom', 'getting', 'house', 'warren',
           'how', 'wrong', 'too', 'name', 'li', 'about', 'find', 'find', 'name', 'hell', 'away', 'li',
           'minute', 'hundred', 'might', 'believe', 'enough', 'done',
           'understand', 'thousand', 'run', 'mind', 'ask', 'else', 'wanted', 'why', 'again',
           'might', 'course', 'talking', 'real',
           'understand', 'trying', 'made', 'looking', 'problem', 'getting', 'thank', 'but','actual','your' ,
           'leave', 'stop', 'then', 'left', 'all', 'world', 'called', 'week', 'minute', 'behind', 'house', 'care',
           'hear', 'how', 'down', 'saw', 'coming', 'off', 'father'
                                                          'came', 'head', 'stay', 'next', 'try', 'start', 'did',
           'enough','cynthia',
           'another', 'pretty', 'room', 'heard', 'jesus', 'boy', 'mother', 'tonight', 'baby', 'mom', 'live',
           'morning', 'hello', 'idea', 'te',
           'town', 'matter', 'whole', 'used', 'seen', 'are', 'wife', 'forget',
           'crazy', 'she', 'together', 'since', 'probably', 'son', 'superman', 'hudson', 'otis', 'fort', 'second','other','mari',
           'drop', 'sidney', 'gale','whoa',
           'may', 'ho', 'hawk', 'air', 'many', 'read','franc',
           'fly', 'mayflower', 'le', 'el', 'van', 'family', 'hundred', 'side', 'twenty', 'month',
           'may', 'deal', 'paper',
           'done' 'tomorrow' 'bill' 'hour' 'show' 'somebody' 'six' 'york' 'buy'
           'without', 'word', 'anybody', 'phone', 'pay', 'show', 'jus', 'stuff', 'taylor', 'dude', 'while', 'mon',
           'actually', 'dear', 'rose', 'quite', 'saying', 'best', 'today', 'thinking',
           'party', 'hard', 'person', 'sound', 'doing', 'exactly',
           'part', 'tomorrow', 'everyone', 'supposed', 'parent', 'dream', 'play', 'ma', 'done', 'max',
           'toddy', 'collins', 'hand', 'also', 'thursday''everybody', 'somebody', 'brother', 'be', 'eat', 'daddy',
           'cause', 'meet', 'ago', 'alone', 'happen', 'child', 'hurt', 'watch', 'bed',
           'worry', 'wa', 'nobody', 'yourself', 'drink', 'abbe', 'hour', 'done', 'six', 'found', 'holly',
           'already', 'hand', 'turn', 'alexander', 'or', 'move', 'napoleon', 'ye', 'end', 'arthur', 'known', 'top',
           'vulcan', 'graham', 'albert', 'seventy', 'bobby', 'toby', 'beth',
           'question', 'check', 'use', 'chance', 'without', 'door', 'far', 'eye',
           'alone', 'line', 'damn', 'ago', 'point', 'dinner', 'true', 'till', 'later', 'young',
           'change', 'important', 'excuse', 'feeling', 'met', 'making', 'of', 'sort',
           'write', 'bring', 'gone', 'least', 'late', 'anymore', 'face', 'over', 'answer', 'stand', 'cool', 'tried',
           'inside', 'trouble', 'shut', 'sleep', 'walk', 'hit', 'easy', 'bet', 'open', 'couple',
            'sit', 'half', 'pick', 'hold', 'telling', 'sometimes',
           'whatever', 'dog', 'taking', 'alright', 'either', 'husband', 'working', 'trust', 'number', 'set', 'million',
           'bitch', 'asked', 'case', 'high', 'break', 'different', 'buy', 'just',
           'bullshit', 'cut', 'figure', 'almost', 'serious', 'with', 'street', 'thirty', 'fifty', 'christ',
           'eight', 'mine', 'plan', 'buck', 'lady', 'pull', 'myself',
           'somewhere', 'send', 'bit', 'rest', 'lost', 'piece']

    clean_corpus1 = []
    for item in clean_corpus:
        clean_corpus1.append([elem for elem in item if (len(elem) > 2 and
                                                        (elem in words) and (not (elem in bad)) and (
                                                                    True not in [char.isdigit() for char in elem]))])

    return clean_corpus1
