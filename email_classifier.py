"""created by Hugo Darwood, trained using the ACL lingspam dataset @ http://www.aueb.gr/users/ion/data/lingspam_public.tar.gz"""

"""
==Designed to be interacted with via shell===

Evaluation of work:
-Classifier seems reasonably accurate at detecting fairly obvious spammy messages.

-I never got around to including an algorithm to re-weight the model when an unknown key is encountered.
 I would achieve this by looking at the cumulative result of the message. If the message was spammy on the
 whole, for example, it should then be added into the spam word dictionary with a weighting equal to its frequency.

-In hindsight I should have had the tests compute a confusion matrix.

"""
import os
# list of generic english stopwords
from nltk.corpus import stopwords


class EmailClassifier:
    ham_word_set = set()
    ham_word_dict = {}
    ham_email_count = 0
    ham_email_proportion = 0

    spam_word_set = set()
    spam_word_dict = {}
    spam_email_count = 0
    spam_email_proportion = 0

    def __init__(self, ham_path=None, hamdirs=None, spam_path=None, spamdirs=None, **kwargs):
        # set up the spam/ham dictionaries
        self.ham_word_set, self.ham_word_dict, self.ham_email_count = self.parse_samples(hamdirs,
                                                                                         ham_path)

        self.spam_word_set, self.spam_word_dict, self.spam_email_count = self.parse_samples(spamdirs,
                                                                                            spam_path)
        # for bayes calculations
        self.spam_email_proportion = (self.spam_email_count / (self.ham_email_count + self.spam_email_count))
        self.ham_email_proportion = (self.ham_email_count / (self.ham_email_count + self.spam_email_count))

        # express all word counts as frequencies:
        for value in self.ham_word_dict:
            self.ham_word_dict[value] /= self.ham_email_count

        for value in self.spam_word_dict:
            self.spam_word_dict[value] /= self.spam_email_count

    # retrain the model if you add in new training data
    def retrain_model(self,ham_path=None, hamdirs=None, spam_path=None, spamdirs=None, **kwargs):
        # set up the spam/ham dictionaries
        self.ham_word_set, self.ham_word_dict, self.ham_email_count = self.parse_samples(hamdirs,
                                                                                         ham_path)

        self.spam_word_set, self.spam_word_dict, self.spam_email_count = self.parse_samples(spamdirs,
                                                                                            spam_path)
        # for bayes calculations
        self.spam_email_proportion = (self.spam_email_count / (self.ham_email_count + self.spam_email_count))
        self.ham_email_proportion = (self.ham_email_count / (self.ham_email_count + self.spam_email_count))

        # express all word counts as frequencies:
        for value in self.ham_word_dict:
            self.ham_word_dict[value] /= self.ham_email_count

        for value in self.spam_word_dict:
            self.spam_word_dict[value] /= self.spam_email_count

    # gather training data and fill the relevant dictionaries
    @staticmethod
    def parse_samples(directory, path):
        _word_set = set()
        line_list = []
        _word_dict = {}
        _sample_count = 0
        # iterate through each of the files in the chosen directory
        for a in directory:
            _sample_count += 1
            # print('parsing: ' + a)
            try:
                for line in open(path + a):
                    line_list = line.split(' ')
                    for word in line_list:
                        if word.lower() in _word_dict:
                            _word_dict[word] += 1
                            # extremely lazy way to filter out any html elements
                        elif len(word) < 10 and word.lower() not in stop_list:
                            _word_set.add(word)
                            _word_dict[word] = 1
                        else:
                            pass
                    line_list = []
            # deal with byte reading errors
            except Exception:
                pass
        print('parsed %s items from the path: %s. vocabulary size: %s words' % (len(directory), path, len(_word_set)))
        return _word_set, _word_dict, _sample_count

    # predict spamicity of a phrase using bayes method
    def prob_spam(self, word):
        try:
            return (self.spam_word_dict[word] * self.spam_email_proportion) / (
                (self.spam_word_dict[word] * self.spam_email_proportion + self.ham_word_dict[
                    word] * self.ham_email_proportion))
        except KeyError:
            print('key: \'%s\' not in spam_dict, use reweight() to handle new keys' % (word))
            return

    def prob_ham(self, word):
        try:
            return (self.ham_word_dict[word] * self.ham_email_proportion) / (
                (self.spam_word_dict[word] * self.spam_email_proportion + self.ham_word_dict[
                    word] * self.ham_email_proportion))
        except KeyError:
            print('key: \'%s\' not in ham_dict, use reweight() to handle new keys' % word)
            return

    @staticmethod
    def prod(iterable):
        i = 1
        for b in iterable:
            try:
                i = i * b
            except:
                pass
        return i

    @staticmethod
    def prodn(iterable):
        i = 1
        for b in iterable:
            try:
                i *= 1 - b
            except Exception as x:
                print(x)
        return i

    def predict_message(self, message):
        # probability that a message is spam - cumulative probability
        prob_array = []
        for i in message.split(' '):
            if i not in stop_list and self.spam_word_dict.__contains__(i):
                prob_array.append(self.prob_spam(i))
        sum_prod = self.prod(prob_array)
        prob_spam_msg = sum_prod / (sum_prod + self.prodn(prob_array))

        # probability that a message is ham
        prob_array = []
        for i in message.split(' '):
            if i not in stop_list and self.ham_word_dict.__contains__(i):
                prob_array.append(self.prob_ham(i))
        sum_prod = self.prod(prob_array)
        prob_ham_msg = sum_prod / (sum_prod + self.prodn(prob_array))

        return '----------\n' \
               'message: \"%s\" \n' \
               'probability of spam: %s , probability of ham: %s \n' \
               'confidence intervals [ham,spam]: %s \n----------' % (message,
                                                                     '%.2f'%prob_spam_msg, '%.2f'%prob_ham_msg,
                                                                     self.test_result(prob_spam_msg, prob_ham_msg))

    # conduct a z test under the alternative hypothesis p_spam =/= p_ham
    def test_result(self, p_spam=None, p_ham=None):
        # pooled standard deviation
        dev = (p_spam * (1 - p_spam) / self.spam_email_count + p_ham * (1 - p_ham) / self.ham_email_count) ** 0.5
        conf_interval_ham = [p_ham - 1.65 * dev, p_ham + 1.65 * dev]
        conf_interval_spam = [p_spam - 1.65 * dev, p_spam + 1.65 * dev]
        return conf_interval_spam, conf_interval_ham


stop_list = stopwords.words('english')

# directory of the spam email samples
spam_path = 'spams/'
spamdirs = os.listdir(spam_path)

# directory of the ham email samples
ham_path = 'hams/'
hamdirs = os.listdir(ham_path)

classifier = EmailClassifier(
        ham_path=ham_path,
        hamdirs=hamdirs,
        spam_path=spam_path,
        spamdirs=spamdirs)

# test a few messages:
print(classifier.predict_message('I have arranged a lunch with your supervisor for tomorrow'))
print(classifier.predict_message('How are we meant to buy the latest computer if we are under budget'))
print(classifier.predict_message('discover NEVER before seen deals at our latest holiday resort'))
print(classifier.predict_message('follow the link to see why thousands of people are changing cable company'))
print(classifier.predict_message('see the gains you could be making by switching to a new car brand'))
print(classifier.predict_message('wild girls in your area are desperate for sexy middle aged men!'))

"""
run output:

parsed 3134 items from the path: hams/. vocabulary size: 44685 words
parsed 981 items from the path: spams/. vocabulary size: 15868 words
----------
message: "I have arranged a lunch with your supervisor for tomorrow"
probability of spam: 0.00 , probability of ham: 1.00
confidence intervals [ham,spam]: ([-0.0008435437613887966, 0.003775896636222444], [0.9962241033637776, 1.000843543761389])
----------
message: "How are we meant to buy the latest computer if we are under budget"
probability of spam: 0.09 , probability of ham: 0.91
confidence intervals [ham,spam]: ([0.07533730086805218, 0.11037700073225842], [0.8896229992677418, 0.924662699131948])
----------
message: "discover NEVER before seen deals at our latest holiday resort"
probability of spam: 0.43 , probability of ham: 0.57
confidence intervals [ham,spam]: ([0.3985350893038612, 0.45827802141399726], [0.5417219785860028, 0.6014649106961389])
----------
message: "follow the link to see why thousands of people are changing cable company"
probability of spam: 1.00 , probability of ham: 0.00
confidence intervals [ham,spam]: ([0.9981492232172553, 1.0008496586654803], [-0.0008496586654801099, 0.001850776782744778])
----------
key: 'switching' not in ham_dict, use reweight() to handle new keys
unsupported operand type(s) for -: 'int' and 'NoneType'
----------
message: "see the gains you could be making by switching to a new car brand"
probability of spam: 0.53 , probability of ham: 0.47
confidence intervals [ham,spam]: ([0.49773430025120935, 0.5580054055572587], [0.44199459444274125, 0.5022656997487907])
----------
key: 'sexy' not in spam_dict, use reweight() to handle new keys
unsupported operand type(s) for -: 'int' and 'NoneType'
key: 'aged' not in ham_dict, use reweight() to handle new keys
unsupported operand type(s) for -: 'int' and 'NoneType'
----------
message: "wild girls in your area are desperate for sexy middle aged men!"
probability of spam: 0.16 , probability of ham: 0.84
confidence intervals [ham,spam]: ([0.13819504998933124, 0.18249389813451566], [0.8175061018654843, 0.8618049500106688])
----------
"""

