from sklearn.metrics import f1_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np

class ArgumentDetector(): 

    # Initialise argument detector
    def __init__(self, modelPath, vectorizerPath):
        self.modelPath= modelPath
        self.vectorizerPath= vectorizerPath
        
    def trainClassifier(self, df):
        print("Training Argument Detector")
        # Split dataset into training and evaluation, 80-20
        trainingSize= int(np.rint(df.shape[0]*0.8))
        train_df= df[:trainingSize]
        eval_df= df[trainingSize:]

        # Fit tokeniser on training dataset
        self.vectorizer = CountVectorizer(min_df=4)
        self.vectorizer.fit(train_df["tweet_text"])

        # Tokenise training and evaluation datasets
        X_train = self.vectorizer.transform(train_df["tweet_text"]).toarray()
        X_eval = self.vectorizer.transform(eval_df["tweet_text"]).toarray()

        # #Visualize word frequency
        # #Code from https://towardsdatascience.com/cross-topic-argument-mining-learning-how-to-classify-texts-1d9e5c00c4cc
        # from yellowbrick.text import FreqDistVisualizer
        # features   = self.vectorizer.get_feature_names()
        # visualizer = FreqDistVisualizer(features=features, size=(850, 1000))
        # visualizer.fit(X_train)
        # visualizer.finalize()
        # visualizer.show()

        # Create LR model and fit on tokenised training data
        self.model = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, train_df["Argumentative"])

        # Evaluate model and calculate F1 score and loss
        eval_predicted= self.model.predict(X_eval)
        f1= f1_score(eval_df["Argumentative"], eval_predicted)
        eval_probs_predicted= self.model.predict_proba(X_eval)[:,1]
        loss= log_loss(eval_df["Argumentative"], eval_probs_predicted)

        # Save trained model and vectoriser
        pickle.dump(self.model, open(self.modelPath, 'wb'))
        pickle.dump(self.vectorizer, open(self.vectorizerPath, "wb"))

        print('Argument detector evaluation F1 Score=', f1, "Loss=", loss)
        # F1= 0.887005649717514 Loss= 0.4762292406646977

    def classifyArguments(self, dataset):
        print("Identifying argumentative tweets")

        # Load the pretrained model and vectoriser 
        loadedVectorizer = pickle.load(open(self.vectorizerPath, "rb"))
        loaded_model = pickle.load(open(self.modelPath, 'rb'))

        # Tokenise training dataset
        X_data = loadedVectorizer.transform(dataset["tweet_text_preprocessed"]).toarray()
        
        # Use LR model to classify tweets argumentative tweets, store result in new column
        result = loaded_model.predict(X_data)
        dataset['Argumentative']= result

        # Return updated dataset
        return dataset

