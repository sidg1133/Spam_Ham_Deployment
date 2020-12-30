from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import re

def clean_data( data):

    data = data.replace('\W', ' ')

    data = re.sub('[^a-zA-Z0-9 \n\.]', '', str(data).lower())
    data = data.replace('\n', " ")

    return data


def word_to_count_matrix(data,vocabulary):
    ''' It creates text mail to number of counts '''
    word_counts_per_mail = {unique_word: 0  for unique_word in vocabulary}
    data = clean_data(data)
    sentence = data.split()
    for word in sentence:
        try:
            word_counts_per_mail[word] += 1
        except:
            continue


    return list(word_counts_per_mail.values())



def classify_spam_ham(message,parameters):
    (p_spam,p_ham) = (0.26404704125443346, 0.7359529587455665)

    message = clean_data(message)
    message = message.split()
    #print(message)

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    #print(message)
    for word in message:
        try:
            if(parameters['parameters_spam'][word]==0 or parameters['parameters_ham'][word]==0 ):
                continue
            temp = parameters['parameters_spam'][word]+parameters['parameters_ham'][word]
            p_spam_given_message = p_spam_given_message*(parameters['parameters_spam'][word]/temp)
            p_ham_given_message *= (parameters['parameters_ham'][word]/temp)
        except:
            continue

    if p_ham_given_message > p_spam_given_message:
        return 'ham'

    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return ["spam","ham"][np.random.randint(0,2)]




app = Flask(__name__)
model_svm = pickle.load(open('SVM.pkl','rb'))


@app.route('/')
def maphtml():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    inp_features = str(request.form['mail'])

    try:
        model_svm = pickle.load(open('SVM.pkl','rb'))
        vocabulary = pd.read_feather('vocabulary.ftr')
        test_str = word_to_count_matrix(inp_features, list(dict(vocabulary).keys()))

        parameters = pd.read_feather('spam_ham_parameters.ftr')
        parameters.set_index('word', inplace=True)

        res_svm = ['ham','spam'][int(model_svm.predict(np.array(test_str).reshape(1,-1)))]
        res_NB = classify_spam_ham(inp_features,parameters)

        text = " Mail is ===>   \t {a} \t    <=== as per SVM Model  \n and \n Mail is ===>     \t {b} \t    <=== as per Naive Bayes Model".format(a=res_svm.upper(),b=res_NB.upper())
        text = text.split('\n')
        return render_template('home.html',prediction_text = text)
    except:
        return "Sorry for inconvenience. ReRun the model!!"


if __name__ == "__main__":
    app.run(debug=True)
