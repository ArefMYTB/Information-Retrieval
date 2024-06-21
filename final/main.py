from parsivar import Normalizer, Tokenizer, FindStems
import json
import dill
import string
from collections import defaultdict
import math

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

# Load docs
docs_path = './IR_data_news_12k.json'
docs = load_docs(docs_path)

def process():
    pre_data = docs
    for _, body in pre_data.items():
        # preprocessing on contents
        body['content'] = preprocess(body['content'])
    return pre_data

# print(preprocessed_data['0'])

def create_index(preprocessed_data):
    # Create the inverted index
    ind = defaultdict(lambda: {'num': 0, 'positions': defaultdict(lambda: [0, []])})
    for i, doc in preprocessed_data.items():
        if int(i) % 1000 == 0:
            print(f'{int(i)} have been processed.')
        content = doc['content']
        for j, token in enumerate(content):
            ind[token]['num'] += 1
            ind[token]['positions'][i][0] += 1
            ind[token]['positions'][i][1].append(j)

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
# save_index(preprocessed_data, "preprocessed_data.pkl")
index = load_index("ind.pkl")
preprocessed_data = load_index("preprocessed_data.pkl")

# for token, token_dict in index.items():
#     print("tok: ", token, " :-> ", token_dict)
#     break


# TFIDF Score
def calculate_tfidf(term_frequency, document_frequency, collection_size):
    if document_frequency != 0:
        return math.log(term_frequency + 1) * math.log(collection_size / document_frequency)
    else:
        return None

def tfidf(term, document, collection):
    # term frequency
    term_frequency = document['content'].count(term)
    # document frequency
    document_frequency = sum([1 for _, doc in collection.items() if term in doc['content']])

    # tdidf
    return calculate_tfidf(term_frequency, document_frequency, len(collection))

# print(tfidf('خبرگزاری', preprocessed_data['3'], preprocessed_data))

# index structure
# Token
#       num
#       positions
#               doc
#               total num in every doc
#               positions
#               tfidf
#       champion list

def update_index(ind, collection_size, champion_lists_ratio=0, champion_list_enable=True):
    for token, token_dict in ind.items():
        document_frequency = token_dict['num']
        champion_list_size = int(champion_lists_ratio * document_frequency)

        if champion_list_enable:
            champion_list = []

        for docID, positions in token_dict['positions'].items():
            term_frequency = positions[0]
            tfidf_score = calculate_tfidf(term_frequency, document_frequency, collection_size)
            positions.append(tfidf_score)
            # preprocessed_data[docID]['length'] += tfidf_score ** 2

            if champion_list_enable:
                champion_list.append((docID, tfidf_score))

        if champion_list_enable:
            champion_list.sort(key=lambda x: x[1], reverse=True)
            token_dict['champion list'] = champion_list[:champion_list_size]
    return ind

index = update_index(index, len(preprocessed_data), 0.5)

# Similarity Funcs

#       Cosine Similarity
def cosine_similarity(query, k, champion_list_enable=False):
    scores = {}

    for term in query:
        if champion_list_enable == True:
            postings = index[term]['champion list']
            document_frequency_query = index[term]['num']
        else:
            postings = index[term]['positions']
            document_frequency_query = index[term]['num']

        term_frequency_query = query.count(term)

        weight_term_query = calculate_tfidf(term_frequency_query, document_frequency_query, len(preprocessed_data))

        for doc in postings:
            if champion_list_enable == True:
                docID = doc[0]
                weight_term_doc = float(doc[1])
            else:
                docID = doc
                weight_term_doc = postings[doc][2]

            if docID in scores:
                scores[docID] += weight_term_query * weight_term_doc
            else:
                scores[docID] = weight_term_query * weight_term_doc

    # print(scores.keys())
    docID_list = list(scores.keys())

    for docID in docID_list:
        doc_length = len(preprocessed_data[f'{docID}']['content']) ** 0.5
        scores[docID] = scores[docID] / doc_length
    #
    scores_tuples = list(scores.items())
    scores_tuples.sort(key=lambda x: x[1], reverse=True)

    return scores_tuples[:k]

#       Jacard Similarity
def calculate_jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union)
    return similarity

def jaccard_similarity(query, k):
    similarities = {}
    query_token = set(query)
    docs_token = docs
    for docID, body in docs_token.items():
        body_tokens = set(preprocess(body['content']))
        similarity = calculate_jaccard_similarity(query_token, body_tokens)
        similarities[docID] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:k]


def run_query(query, similarity_type):

    print("query: ", query)
    print("similarity function: ", similarity_type)

    if similarity_type == "Cosine":
        result = cosine_similarity(preprocess(query), 5, False)
    else:
        result = jaccard_similarity(preprocess(query), 5)

    for r in result:
        print(f"Document ID is: {r[0]}")
        print(f"Document Score is: {r[1]}")
        print(preprocessed_data[f'{r[0]}']['title'])
        print(preprocessed_data[f'{r[0]}']['url'])

query = "اسناد"
similarity_type = "Jaccard"
run_query(query, similarity_type)

