import pandas as pd
import random

# Create argument pairs CSV file for annotation for relation detection
def pairArguments( inputFilename, outputFilename):
    # Extract argument dataset from CSV
    dataset= pd.read_csv(inputFilename)
    # Sort tweets by date
    dataset.sort_values(by=['created_at'], inplace=True, ascending=False)
    dataset=dataset.reset_index(drop=True)

    # Create all unique pairs of indixes
    pairsIndex=[]

    for i in range(dataset.shape[0]-1):
        pairsIndex.append((i, i+random.randrange(1, dataset.shape[0]-i) ))
        pairsIndex.append((i, i+random.randrange(1, dataset.shape[0]-i) ))

    pairsIndex= list(set(pairsIndex))

    arg1Index = [a_tuple[0] for a_tuple in pairsIndex]
    arg2Index = [a_tuple[1] for a_tuple in pairsIndex]

    # Add arguments pairs to arrays representing dataset columns
    argument1= []
    argument2= []
    for i in range(len(arg1Index)):
        argument1.append(dataset.tweet_text[arg1Index[i]])
        argument2.append(dataset.tweet_text[arg2Index[i]])

    pair_dict = {
        'Argument1': argument1,
        'Argument2': argument2,
    }
    # Create the dataset using the argument arrays
    argumentPairs = pd.DataFrame(pair_dict)
    argumentPairs.to_csv(outputFilename)
    print("Argument pair dataset created")

if __name__ == '__main__':
    # Specify the file containing the arguments to be paired.
    # Specify CSV file name
    pairArguments("../Datasets/Argument_Annotated_Tweets.csv", '../Datasets/ArgumentPairsForAnnotation.csv')
    
