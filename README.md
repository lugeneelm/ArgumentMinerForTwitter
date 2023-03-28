# ArgumentMinerForTwitter

This project contains code for an Argument Miner for Twitter. The argument miner takes as input training datasets and a dataset to mine from. Running the miner returns an argument visualisation of the argument structures identified within the provided dataset. The argument miner is best suited for training and use on a controversial topic which can't be divided into smaller subtopics, e.g. Insulate Britain. Guidance on how to use the miner on the hashtag InsulateBritain annotated corpus is provided below alongside how to retrain the model on a new topic.
Code can also be found for the relation detection experiment conducted to identify the best relation detection model for the miner. Guidance is provided on how to replicate the relation detection experiment below.
Python3 was used to develop all of the code in this project.

Training the argument miner:
In order to train the argument miner, 2 datasets must be provided for training. For the Insulate Britain corpus these datasets have already been provided, "Argument_Annotated_Tweets.csv" and "PAIRED-ANNOTATED.csv". 
To train the miner on new datasets then the constants defined at the top of the ”Argument_miner.py” file can be modified. 2 datasets must be provided to the miner for training. The first dataset is a dataset of annotated tweets. The dataset should include the 2 columns ”tweet text” containing the tweets and ”Argumentative” containing labels. The ”1” label represents argumentative tweets and
the ”0” label represents non-argumentative tweets. The ”Scraper.py” file can be used to obtain a dataset for annotation. The second dataset is a dataset of labelled argument pairs. It should include 3 columns ”Argument1”, ”Argument2” and ”Label”. The first 2 columns consist of tweets and the final is a label of the relation between the pair. The label ”0” represents conflicting tweets while the label ”1” represents supporting ones. A pair dataset can be obtained for annotation using the "ArgumentPairer.py” file by providing a dataset of argumentative tweets.

Mining arguments using the argument miner:
To mine arguments using the miner it must first have access to training datasets for training.
As well as the training datasets a dataset of scraped tweets must be passed to the argument miner to mine from. A dataset has been provided to test the Insulate Britain miner, "Evaluation_tweets.csv". The filename for this dataset can be specified at the bottom of the ”Argument_miner.py” file The dataset must contain the columns ”tweet_text” and ”created_at”, containing tweet text and time-codes respectively. At least 2 tweets must be present in the dataset but more may be needed to produce the argument visualisation. An error message will appear if this is the case. In order to run the argument miner on an Insulate Britain tweet dataset the ”Argument miner.py” file can simply be called from the command line using the command "python Argument_miner.py".
When the miner is run for the first time, this command trains the miner and saves the models and vectorizers trained to avoid retraining again when the miner is used. The miner will then use the tweet dataset provided to mine arguments from produce the argument visualisation. The desired filename for the visualisation PNG file can be specified alongside the dataset to mine from in the main method of the ”Argument_miner.py” file.  The "Scraper.py" file can be used to obtain a tweet dataset to mine from by running it in the command line using the command "python Scraper.py" and providing the name of the scraped dataset to the miner. 

The NLTK stopwords library is used during preprocessing. 
The following command may need to be called prior to running the miner for the first time.
    "import nltk
    nltk.download('stopwords')"

Reproducing the relation detection experiment:
Each of the 6 models tested in the experiment were created individually in their respective files. 
All of the models can be tested from the command line with the command: "python FileName.py" where FileName can be replaced by whichever model file being tested.

The LSTM model makes use of the GloVe text vectorizer. The glove.6B.100d.txt file used by the model is of a significant size. It can be downloaded from https://nlp.stanford.edu/projects/glove/ and saved at the file path "/Datasets/glove-2/glove.6B.100d.txt".

