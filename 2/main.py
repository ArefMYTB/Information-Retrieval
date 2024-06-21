
from parsivar import Normalizer, Tokenizer, FindStems
import json
import string
from collections import defaultdict

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
  persian_stopwords = ['و','در','به','با','از','که','این','را','برای','تا','بر','هم','نیز',
                       'یا','هر','پس','البته','هنوز','؟','،','زیرا']
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


docs_path = './IR_data_news_12k.json'
docs = load_docs(docs_path)
# print(docs['0'])
preprocessed_data = docs

for _, body in preprocessed_data.items():
        # preprocessing on contents
        body['content'] = preprocess(body['content'])
#
# print(preprocessed_data['0'])

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

term = index['خبر']
print(f"Total frequency: {term['num']}")
print("Positions:")
for doc_id, positions in term['positions'].items():
    print(f"  Document {doc_id}: total_number: {positions[0]}, positions: {positions[1]}")
print()
