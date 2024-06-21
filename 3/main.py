from parsivar import Normalizer, Tokenizer, FindStems
import json
import dill
import string
from collections import defaultdict, Counter
import re

normalizer = Normalizer()
tokenizer = Tokenizer()
stemmer = FindStems()


def load_docs(docs_path):
    result = {}
    with open(docs_path) as f:
        docs = json.load(f)
        for docID, body in docs.items():
            result[docID] = {}
            result[docID]['title'] = body['title']
            result[docID]['content'] = body['content']
            result[docID]['url'] = body['url']
    return result


def remove_stop_words(words):
    persian_stopwords = ['و', 'در', 'به', 'با', 'از', 'که', 'این', 'را', 'برای', 'تا', 'بر', 'هم', 'نیز',
                         'یا', 'هر', 'پس', 'البته', 'هنوز', '؟', '،', 'زیرا']
    return [word for word in words if word not in persian_stopwords]


def preprocess(text):
    # remove all the punctuations(!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~), if there is any
    pure_text = text.translate(str.maketrans('', '', string.punctuation))
    # tokenizer
    tokens = tokenizer.tokenize_words(normalizer.normalize(pure_text))
    # stemming
    stemmed_tokens = list(map(stemmer.convert_to_stem, tokens))
    result = remove_stop_words(stemmed_tokens)
    return result


# print(preprocess('سلام خوبی؟ امیدوارم امروز حالت خوب باشه. تقدیم به تو می یابد!'))

# Load docs
docs_path = './IR_data_news_12k.json'
docs = load_docs(docs_path)


def process():
    # print(docs['0'])
    preprocessed_data = docs

    for _, body in preprocessed_data.items():
        # preprocessing on contents
        body['content'] = preprocess(body['content'])

    print(preprocessed_data['0'])


def create_index(preprocessed_data):
    # Create the inverted index
    index = defaultdict(lambda: {'num': 0, 'positions': defaultdict(lambda: [0, []])})
    for i, doc in preprocessed_data.items():
        if int(i) % 1000 == 0:
            print(f'{int(i)} have been processed.')
        content = doc['content']
        for j, token in enumerate(content):
            index[token]['num'] += 1
            index[token]['positions'][i][0] += 1
            index[token]['positions'][i][1].append(j)


# term = index['خبر']
# print(f"Total frequency: {term['num']}")
# print("Positions:")
# for doc_id, positions in term['positions'].items():
#     print(f"  Document {doc_id}: total_number: {positions[0]}, positions: {positions[1]}")
# print()


def save_index(ind, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        dill.dump(ind, outp)
        print(f'index saved in {filename}')


def load_index(filename):
    ind = None
    with open(filename, 'rb') as inp:
        ind = dill.load(inp)
    return ind


# save_index(index, "ind.pkl")
index = load_index("ind.pkl")


def query_search(query):
    # quatation words
    preprocessed_quotation_query = {}
    # words after !
    words_after_exclamation = re.findall(r'\!\s(\w+)', query)
    # words between quotations: both " " & ' '
    words_in_quotation = re.findall(r'["\']([^"\']*)["\']', query)

    # delete words_after_exclamation & words_in_quotation
    raw_tokens = re.sub(r'\!\s\w+', '', query)
    raw_tokens = re.sub(r'["\']([^"\']*)["\']', '', raw_tokens)

    # Convert index into set
    index_set = set(index)

    # preprocess each type of token in query
    # raw tokens
    preprocessed_raw_query = dict()
    if len(raw_tokens):
        preprocessed_raw_query = preprocess(raw_tokens)

        # Convert preprocessed_query into set
        preprocessed_raw_query_set = set(preprocessed_raw_query)
        # Use set intersection to find common elements
        filtered_raw_tokens = list(preprocessed_raw_query_set.intersection(index_set))
        preprocessed_raw_query = filtered_raw_tokens

    # !
    if len(words_after_exclamation):
        preprocessed_exclamation_query = preprocess(" ".join(words_after_exclamation))

        # Convert preprocessed_query into set
        preprocessed_exclamation_query_set = set(preprocessed_exclamation_query)
        # Use set intersection to find common elements
        filtered_exclamation_tokens = list(preprocessed_exclamation_query_set.intersection(index_set))
        words_after_exclamation = filtered_exclamation_tokens

    # ""
    if len(words_in_quotation):

        preprocessed_quotation_query = list()
        for phrase in words_in_quotation:
            preprocessed_quotation_query.append(preprocess(phrase))

    print("Query: ", query)
    print("Terms: ", preprocessed_raw_query)
    print("Forbidden Terms: ", words_after_exclamation)
    print("phase words: ", preprocessed_quotation_query)

    # docs with quotation terms
    max_diff = 1
    if preprocessed_quotation_query:
        for phase in preprocessed_quotation_query:
            # print("phase:  ", phase)
            fisrt_term = phase[0]
            if fisrt_term in index:
                for docID, positions in index[fisrt_term]['positions'].items():  # Get the list of documents containing the first term
                    # print("Doc:  ", docID)
                    for pos in positions[1]:  # check for all positions of the first term
                        # print("Pose:  ", pos)
                        for i in range(1, len(phase)):  # check for all words in that phase
                            if phase[i] == docs[f'{docID}']['content'].split()[pos+i+1]:
                                if docID not in scores:
                                    scores[docID] = (0, 0)  # Initialize relevance score for the document
                                if i == len(phase)-1:
                                    scores[docID] = (scores[docID][0] + 1, scores[docID][1] + positions[0])  # Increment the relevance score for the document based on term appearance
    # docs with accepted terms
    for term in preprocessed_raw_query:
        if term in index:
            for docID, positions in index[term]['positions'].items():  # Get the list of documents containing the term
                if docID not in scores:
                    scores[docID] = (0, 0)  # Initialize relevance score for the document
                scores[docID] = (scores[docID][0] + 1, scores[docID][1] + positions[0])  # Increment the relevance score for the document based on term appearance
    # docs with forbidden terms
    for term in words_after_exclamation:
        if term in index:
            for docID, _ in index[term]['positions'].items():  # Get the list of documents containing the term
                if docID in scores:
                    scores[docID] = (-1, 0)  # Initialize relevance score for the document

    # Sort the documents based on their relevance scores in descending order
    ranked_docs = sorted(scores.keys(), key=lambda x: (scores[x][0], scores[x][1]), reverse=True)

    return ranked_docs



scores = {}

q = "'لیگ برتر' ! چغر"
results = query_search(q)

if not results:
    print("No result for this query")

for doc in results:
    print("Document:", doc, "in Title of: ", docs[doc]['title'], "Relevance Score:", scores[doc])
