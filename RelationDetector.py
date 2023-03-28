from sklearn.metrics import f1_score, log_loss
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV

class RelationDetector(): 

    # Initialise relation detector
    def __init__(self, modelPath, vectorizerPath):
        self.modelPath= modelPath
        self.vectorizerPath= vectorizerPath
        
    def train(self, df):
        print("Training Relation Detector")

        # Split dataset into training and evaluation, 80-20
        trainingSize=int(np.rint(df.shape[0]*0.8))
        df.dropna(inplace=True)
        train_df= df[:trainingSize]
        eval_df= df[trainingSize:]

        # Fit tokeniser on training dataset
        trainingArguments= train_df["Argument1"].append(train_df["Argument2"])
        vectorizer = CountVectorizer(min_df=4)
        vectorizer.fit(trainingArguments)

        # Tokenise argument pairs in training and evaluation datasets
        A1_train= vectorizer.transform(train_df["Argument1"]).toarray()
        A2_train= vectorizer.transform(train_df["Argument2"]).toarray()

        A1_eval= vectorizer.transform(eval_df["Argument1"]).toarray()
        A2_eval= vectorizer.transform(eval_df["Argument2"]).toarray()

        # Combine tweet pair vectors
        X_train= np.column_stack((A1_train, A2_train))
        X_eval= np.column_stack((A1_eval,A2_eval))

        Y_train= train_df["Label"]
        Y_eval= eval_df["Label"]

        # Create SVN model and fit on tokenised training data
        model=svm.SVC(kernel='linear', C=0.1)
        model.fit(X_train, Y_train)

        # Create calibrated classifier to get probabilitic predictions
        clf= CalibratedClassifierCV(model).fit(X_train, Y_train)

        # Evaluate SVM Model
        eval_probs_predicted = clf.predict_proba(X_eval)[:,1]
        eval_predicted=model.predict(X_eval)
        
        f1= f1_score(Y_eval, eval_predicted)
        Y_eval= Y_eval.to_numpy(dtype=np.float64)
        loss= log_loss(Y_eval, eval_probs_predicted)

        print('Relation detector evaluation F1=', f1, ', Loss=', loss)
        # F1= 0.7234042553191489, Loss= 0.7140861130332254

        # Save trained model and vectoriser
        pickle.dump(model, open(self.modelPath, 'wb'))
        pickle.dump(vectorizer, open(self.vectorizerPath, "wb"))

    def predict(self, dataset):
        print("Identify relations between tweet pairs")

        # Load the pretrained model and vectoriser 
        loadedVectorizer = pickle.load(open(self.vectorizerPath, "rb"))
        loaded_model = pickle.load(open(self.modelPath, 'rb'))

        # Tokenise training dataset
        X1= loadedVectorizer.transform(dataset["Argument1_preprocessed"]).toarray()
        X2= loadedVectorizer.transform(dataset["Argument2_preprocessed"]).toarray()
        X= np.column_stack((X1, X2))
        
        # Use LR model to classify tweets argumentative tweets, store result in new column
        result = loaded_model.predict(X)
        dataset['Relation']= result

        return dataset
