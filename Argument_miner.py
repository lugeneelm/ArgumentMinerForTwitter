from ArgumentDetector import ArgumentDetector
from RelationDetector import RelationDetector
from Preprocessing import preprocess
import pandas as pd
from ArgumentGraph import plotGraph
import random
from os.path import exists

class ArgumentMiner():

    # File paths for saved models and vectorisers
    argumentDetectionModel= 'Models/argumentDetectionModel.pickel'
    argumentDetectionVectorizer= "Models/argumentDetectionVectorizer.pickel"
    relationDetectionModel= 'Models/relationDetectionModel.pickel'
    relationDetectionVectorizer= "Models/relationDetectionVectorizer.pickel"

    # File paths for training datasets
    argumentDetectorTainingDataset= '../Datasets/Argument_Annotated_Tweets.csv'
    relationDetectorTainingDataset= "../Datasets/PAIRED-ANNOTATED.csv"

    # Initialise argument miner
    def __init__(self):
        print("Initialising argument miner")
        # Initiallise argument detector and relation detector
        # Specify file paths to save models and vectorisers
        self.argumentDetector = ArgumentDetector(self.argumentDetectionModel, self.argumentDetectionVectorizer)
        self.relationDetector= RelationDetector(self.relationDetectionModel, self.relationDetectionVectorizer)

    # Check whether pretrained models and vectorisers exist at file paths. 
    def modelsAndVectorisersExist(self):
        if (exists(self.argumentDetectionModel) and exists(self.argumentDetectionVectorizer) and exists(self.relationDetectionModel) and exists(self.relationDetectionVectorizer)):
            return True 
        else:
            return False

    # Pair argumentative tweets to pass to relation detector
    def pairArguments(self, dataset):
        # Sort arguments in ascending order by timecode
        dataset.sort_values(by=['created_at'], inplace=True, ascending=False)
        dataset=dataset.reset_index(drop=True)

        # Create all unique pairs of tweet indexes
        pairsIndex=[]
        for i in range(dataset.shape[0]-1):
            pairsIndex.append((i, i+random.randrange(1, dataset.shape[0]-i) ))
            pairsIndex.append((i, i+random.randrange(1, dataset.shape[0]-i) ))
        pairsIndex= list(set(pairsIndex))

        # Use indexes to create tweet pair tuples
        arg1Index = [a_tuple[0] for a_tuple in pairsIndex]
        arg2Index = [a_tuple[1] for a_tuple in pairsIndex]

        # Add preprocessed and raw argument pairs
        argument1= []
        argument2= []
        argument1_preprocessed= []
        argument2_preprocessed= []
        for i in range(len(arg1Index)):
            argument1.append(dataset.tweet_text[arg1Index[i]])
            argument2.append(dataset.tweet_text[arg2Index[i]])
            argument1_preprocessed.append(dataset.tweet_text_preprocessed[arg1Index[i]])
            argument2_preprocessed.append(dataset.tweet_text_preprocessed[arg2Index[i]])

        # Create new dataframe with argument pairs
        pair_dict = {
            'Argument1': argument1,
            'Argument2': argument2,
            'Argument1_preprocessed': argument1_preprocessed,
            'Argument2_preprocessed': argument2_preprocessed,
        }
        argumentPairs = pd.DataFrame(pair_dict)

        return argumentPairs

    # Training the argument miner
    def trainMiner(self):
        # Extracting dataset from CSV file containing annotated tweets
        df= pd.read_csv(self.argumentDetectorTainingDataset)
        # Preprocess tweets 
        df["tweet_text"]= df["tweet_text"].apply(preprocess)
        # Training the argument detector
        self.argumentDetector.trainClassifier(df)

        # Extracting dataset from CSV file containing annotated argument pairs
        pairs_df= pd.read_csv(self.relationDetectorTainingDataset)
        # Preprocess tweets 
        pairs_df["Argument1"]= pairs_df["Argument1"].apply(preprocess)
        pairs_df["Argument2"]= pairs_df["Argument2"].apply(preprocess)
        # Training the relation detector
        self.relationDetector.train(pairs_df)

    # Run the argument miner with a CSV file of tweets on Insulate Britain
    def mineArguments(self, filename, pngFilename):
        # Extract dataset from file
        df= pd.read_csv(filename)
        if(df.shape[0]<2): 
            print("The dataset provided does not contain enough tweets to mine.")
            return

        # Preprocess dataset
        df["tweet_text_preprocessed"]= df["tweet_text"].apply(preprocess)

        # Classify tweets into argumentative and non argumentative
        tweetsClassified= self.argumentDetector.classifyArguments(df)

        # Filter to obtain only argumentative tweets
        arguments=tweetsClassified[tweetsClassified["Argumentative"]==1].reset_index(drop=True)

        if(arguments.shape[0]<2): 
            print("Mining could not be complete as not enough arguments were detected.")
            return

        # Pair argumentative tweets
        pairs_df= self.pairArguments(arguments)

        # Detect argument pair relations 
        relationsPredicted= self.relationDetector.predict(pairs_df)

        # Create argument visualisation using argument pair relations
        plotGraph(relationsPredicted, pngFilename)

if __name__ == '__main__':
    miner= ArgumentMiner()
    # Specify file name for dataset to mine from and file name for visualisation
    visualisationFileName= 'Graphs/ArgumentVisualisation.png'
    inputDatasetFilename= '../Datasets/Evaluation_tweets.csv'
    # If saved models and vectorisers exist then run miner, else return train miner then run.
    # If a dataset does not exist at the path provided return error message
    if (exists(inputDatasetFilename)):
        if miner.modelsAndVectorisersExist():
            miner.mineArguments(inputDatasetFilename, visualisationFileName)
        else:
            miner.trainMiner()
            miner.mineArguments(inputDatasetFilename, visualisationFileName)
    else: print("The file specified cannot be found.")
