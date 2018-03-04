#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import pdb

''' ============ Description ============
Bag-of-words representation in text documents
 - convert all involved text documents to a sparse matrix representation, each row corresponding to a particular document and containing word/frequency mappings (count matrix).
 - use the CountVectorizer for this purpose
 - sparse matrix because each document generally uses only a small portion of the overall vocabulary spanning all documents
- Apply the TfidfTransformer to discount the importance of constantly repeating filler terms throughout documents
    - see http://scikit-learn.org/stable/modules/feature_extraction.html

=========== Prereqs ============
Natural Language Tool Kit: python-nltk
    - nltk.download() o download the tokenizers/punkt package
Word Cloud - https://github.com/amueller/word_cloud. 
    pip install wordcloud
'''

TEST_DATA_PERCENT = .2
RANDOM_STATE = None
CV_FOLDS = 3
RESULTS_NUM_PRECISION = 3

def filter_wordlist(wordlist):
    new_wordlist = ''
    for val in wordlist:
        text = val.lower()
        tokens = nltk.word_tokenize(text)
        #tokens = [word for word in tokens if word not in stopwords.words('english')]
        for words in tokens:
            new_wordlist += words + ' '
    return new_wordlist

# Generate word clouds from words corresponding to each label
def visualize_data(data_frame):
    spam = data_frame[data_frame.label_num == 1]
    ham = data_frame[data_frame.label_num == 0]
    ham_words = filter_wordlist(ham.text)
    spam_words = filter_wordlist(spam.text)
    spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
    ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)
    #Spam Word cloud
    plt.figure(figsize=(10,8), facecolor='k')
    plt.imshow(spam_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    #Ham word cloud
    plt.figure( figsize=(10,8), facecolor='k')
    plt.imshow(ham_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# Initiate the fixed and varied hyperparameters for all pipeline text feature transformers and ML classifiers
def build_cross_validator(pipeline):
    param_grid = {}
    for step in pipeline.steps[:-1]: # Iterate all transformers
        transform = step[-1]
        if isinstance(transform, CountVectorizer):
            transform.set_params(**{
                'strip_accents': None,
                'ngram_range': (1,1),
                'stop_words': None,
                'max_df': 0.7,
                'max_features': None,
            })
            param_grid.update({
                'vect__ngram_range': [(1,1), (1,2)] # 2-ngrams sometimes yields marginal improvement and sometimes not
            })
        elif isinstance(transform, TfidfTransformer):
            transform.set_params(**{
                'norm': 'l2',
                'use_idf': True,
                'smooth_idf': False,
            })
    clf = pipeline.steps[-1][-1] 
    if isinstance(clf, MultinomialNB):
        param_grid.update({
            'clf__alpha': [0.8, 1.0], # Smoothing parameter
        })
    elif isinstance(clf, LogisticRegression):
        clf.set_params(**{
            'random_state': RANDOM_STATE,
            'solver': 'liblinear',
            'multi_class': 'ovr', # one-vs-rest if > 2 classes
            'max_iter': 100,
            'verbose': 0,
        })
        param_grid.update({
            'clf__penalty': ['l1','l2'],
            'clf__C': [0.1,0.5,1.0], # Inverse of regularization strength
        })
    elif isinstance(clf, KNeighborsClassifier):
        clf.set_params(**{
            'algorithm': 'auto',
            'metric': 'minkowski',
            'p': 2, # power param of minkowski distance
        })
        param_grid.update({
            'clf__n_neighbors': [3,5,7,9,11],
            'clf__weights': ['uniform', 'distance']
        })
    elif isinstance(clf, RandomForestClassifier):
        clf.set_params(**{
            'bootstrap': True, # if True, draw samples w/ replacement
            'random_state': RANDOM_STATE,
            'criterion': 'entropy',
            'max_leaf_nodes': None,
            'verbose': 0,
        })
        param_grid.update({
            'clf__max_features': [None, 'log2'],
            'clf__max_depth': [9,13], # Yet to overfit w/ 13
            #'min_samples_split': [32,16,8,4,2],
            'clf__n_estimators': [30] # best w/ 30 (even w/ 50 present)
        })
    elif isinstance(clf, AdaBoostClassifier):
        # Algorithm: AdaBoost-SAMME
        clf_weak = DecisionTreeClassifier(criterion='entropy', 
                max_depth=1, min_samples_split=10)
        clf.set_params(**{
            'base_estimator': clf_weak,
            'random_state': RANDOM_STATE,
        })
        param_grid.update({
            'clf__base_estimator__max_depth': [1,3,5,7], # overfit after 3 w/ only 1 CountVectorizer ngrams, but yet to overfit with 2
            'clf__learning_rate': [1], 
            'clf__n_estimators': [30,50] 
        })
    cv_clf = GridSearchCV(estimator=pipeline, 
        param_grid=param_grid, cv=CV_FOLDS, 
        return_train_score=True, verbose=1, n_jobs=1)
    return cv_clf

# Output and optional plot the learning curve for classifier clf and data given by X and y
def output_plot_learning_curve(model_name, clf, X, y, scoring="accuracy", enable_plot=False):
    train_sizes = np.linspace(.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=CV_FOLDS, n_jobs=-1, 
        scoring=scoring, verbose=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    df = pd.DataFrame(data=zip(train_sizes, train_scores_mean, 
        train_scores_std, test_scores_mean, test_scores_std),
        columns=['train-size', 'CV-train-mean', '+/-', 
                'CV-test-mean', '+/-']).round(RESULTS_NUM_PRECISION)
    print df
    if not enable_plot:
        return
    plt.figure()
    plt.title("Learning Curves: %s" % (model_name))
    plt.xlabel("Training examples")
    plt.ylabel("%s score" % scoring)
    plt.grid()
    plt.fill_between(train_sizes, 
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1, color="r")
    plt.fill_between(train_sizes, 
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
        label="CV train score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
        label="CV test score")
    plt.legend(loc='best')
    plt.show()
    plt.close()

# Output and optionally render the confusion matrix cm with given classes
def output_plot_confusion_matrix(model_name, cm, classes,
        normalize=False, title='Confusion matrix', 
        cmap=plt.cm.Blues, enable_graphical=False):
    np.set_printoptions(precision=2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print "Confusion matrix:"
    print cm
    if classes is None:
        classes = range(len(cm))
    if not enable_graphical:
        return
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    idx_prod = [(i,j) for i in range(cm.shape[0]) for j in range(cm.shape[1])]
    for i, j in idx_prod:
        plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()    

# Generate the command line parameters and help text
def generate_cmd_line_opts(op, models):
    model_options = [model.__class__.__name__ for model in models]
    op.add_option("-m", "--model", dest="model_idx", type="int",
        help="select 0 - %d one of the following models: %s" % (len(models) - 1, model_options))
    op.add_option("-V", "--visualize", action="store_true", dest="visualize_data_only", default=False, help="Visualize data only. No ML.")
    op.add_option("-l", "--learning_curve", action="store_true", dest="compute_learning_curve", default=False, help="Compute and output the learning curve table.")
    op.add_option("--plot_lc", action="store_true", dest="plot_lc", default=False, help="Plot the learning curve.")
    op.add_option("-t", "--use_test_data", action="store_true", dest="use_test_data", default=False, help="Evaluate model against test data.")
    op.add_option("--cm_graphical", action="store_true", dest="cm_graphical", default=False, help="Display the graphical confusion matrix.")

def main(argv):
    models = [MultinomialNB(), LogisticRegression(), 
            KNeighborsClassifier(), RandomForestClassifier(), 
            AdaBoostClassifier()]
    # ========== Process command line parameter ==========
    op = OptionParser()
    generate_cmd_line_opts(op, models)
    argv = argv[1:]
    (opts, args) = op.parse_args(argv)
    if len(argv) < 1:
        op.print_help()
        sys.exit()
    if opts.model_idx in range(len(models)):
        models = [models[opts.model_idx]]

    # ========== Import data ==========
    data = pd.read_csv("spam_2col.csv", encoding='latin-1')
    # ========== Preprocess ==========
    if len(data.columns) > 2:
        data = data.drop(data.columns[2:], axis=1)
        data.to_csv("spam_2col.csv", index=False, encoding='latin-1')
    data = data.rename(columns={"v1":"label", "v2":"text"})
    # Count observations in each label
    print data.label.value_counts()
    # Numerically encode the labels (and place in new column)
    data['label_num'] = data.label.map({'ham':0, 'spam':1})
    if opts.visualize_data_only:
        visualize_data(data)
        sys.exit()
    # ========== Split test/train data ==========
    X, y = data["text"], data["label_num"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=TEST_DATA_PERCENT, stratify=y, 
        random_state=RANDOM_STATE)
    # ========== Vectorize the text features ==========
    # Convert all text sample to a bag-of-words sparse matrix representation, w/ optional Tfidf Transformation
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        #('tfidf', TfidfTransformer()),
        ('clf', None), # Provide the classifier further below
    ])
    # ======= Train models via Cross Validation ========
    for model in models:
        pipeline.steps[-1] = ('clf', model)
        cv_clf = build_cross_validator(pipeline)
        print "\nTraining %s model. \n" % model
        cv_clf.fit(X_train, y_train)
        print "\nBest parameters: %s" % cv_clf.best_params_
        print "\nBest (mean) train score: %0.3f" % max(cv_clf.cv_results_['mean_train_score'])
        print "Best (mean) validation score: %0.3f" % cv_clf.best_score_
        best_clf = cv_clf.best_estimator_
        model_name = type(model).__name__
        if opts.compute_learning_curve:
            output_plot_learning_curve(model_name, best_clf, 
                X_train, y_train, enable_plot=opts.plot_lc)
        if opts.use_test_data:
            test_score = cv_clf.score(X_test, y_test)
            print "\nTest score: %0.3f" % test_score
            y_pred = cv_clf.predict(X_test)
            # A few scoring metrics
            print(classification_report(y_test, y_pred, 
                    target_names = ["Ham", "Spam"]))
            conf_mat = confusion_matrix(y_test, y_pred)
            output_plot_confusion_matrix(model_name, conf_mat, 
                    classes=["Ham", "Spam"], normalize=False,
                    enable_graphical=opts.cm_graphical)
            #pd.set_option('display.max_colwidth', -1)
            print "\nTest data misclassified as spam:"
            print X_test[y_test < y_pred]
            print "\nTest data misclassified as ham:"
            print X_test[y_test > y_pred]

if __name__ == "__main__":
    main(sys.argv)
