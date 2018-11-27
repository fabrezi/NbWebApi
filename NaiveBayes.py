import pandas as pd
import numpy as np
from collections import defaultdict
import re


def preprocess_string(str_arg):
    cleaned_str = re.sub('[^a-z\s]+', ' ', str_arg, flags=re.IGNORECASE)
    cleaned_str = re.sub('(\s+)', ' ', cleaned_str)
    cleaned_str = cleaned_str.lower()
    return cleaned_str

#Common stop words from online
stop_words = [
"a", "about", "above", "across", "after", "afterwards", 
"again", "all", "almost", "alone", "along", "already", "also",    
"although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", 
    "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", 
    "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", 
    "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", 
    "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", 
    "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", 
    "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", 
    "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", 
    "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", 
    "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", 
    "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should",
    "since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take",
    "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
"this", "those", "though", "through", "throughout",
"thru", "thus", "to", "together", "too", "toward", "towards",
"under", "until", "up", "upon", "us",
"very", "was", "we", "well", "were", "what", "whatever", "when",
"whence", "whenever", "where", "whereafter", "whereas", "whereby",
"wherein", "whereupon", "wherever", "whether", "which", "while", 
"who", "whoever", "whom", "whose", "why", "will", "with",
"within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
]

class NaiveBayes:

    def __init__(self, unique_classes):

        self.classes = unique_classes

    def addToBow(self, example, dict_index):

        if isinstance(example, np.ndarray): example = example[0]

        for token_word in example.split():
            self.bow_dicts[dict_index][token_word] += 1

    def train(self, dataset, labels):

        self.examples = dataset
        self.labels = labels
        self.bow_dicts = np.array([defaultdict(lambda: 0) for index in range(self.classes.shape[0])])
        if not isinstance(self.examples, np.ndarray): self.examples = np.array(self.examples)
        if not isinstance(self.labels, np.ndarray): self.labels = np.array(self.labels)
        for cat_index, cat in enumerate(self.classes):
            all_cat_examples = self.examples[self.labels == cat]  # filter all examples of category == cat
            cleaned_examples = [preprocess_string(cat_example) for cat_example in all_cat_examples]
            cleaned_examples = pd.DataFrame(data=cleaned_examples)
            np.apply_along_axis(self.addToBow, 1, cleaned_examples, cat_index)
        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            prob_classes[cat_index] = np.sum(self.labels == cat) / float(self.labels.shape[0])
            count = list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index] = np.sum(
                np.array(list(self.bow_dicts[cat_index].values()))) + 1  # |v| is remaining to be added
            all_words += self.bow_dicts[cat_index].keys()

        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = self.vocab.shape[0]
        denoms = np.array(
            [cat_word_counts[cat_index] + self.vocab_length + 1 for cat_index, cat in enumerate(self.classes)])
        self.cats_info = [(self.bow_dicts[cat_index], prob_classes[cat_index], denoms[cat_index]) for cat_index, cat in
                          enumerate(self.classes)]
        self.cats_info = np.array(self.cats_info)

    def getExampleProb(self, test_example):
        likelihood_prob = np.zeros(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):

            for test_token in test_example.split():
                test_token_counts = self.cats_info[cat_index][0].get(test_token, 0) + 1
                test_token_prob = test_token_counts / float(self.cats_info[cat_index][2])
                likelihood_prob[cat_index] += np.log(test_token_prob)
        post_prob = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            post_prob[cat_index] = likelihood_prob[cat_index] + np.log(self.cats_info[cat_index][1])

        return post_prob

    def test(self, test_set):

        predictions = []  # to store prediction of each test example
        for example in test_set:
            cleaned_example = preprocess_string(example)
            post_prob = self.getExampleProb(cleaned_example)
            predictions.append(self.classes[np.argmax(post_prob)])

        return np.array(predictions)
