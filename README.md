# Ratings-from-review--Yelp-data
#The libraries required are numpy, pandas, scikit learn, json, re, sys, gensim, logging, keras

#Read 1:

The Yelp data set came with following data:

1. Restaurant information
2. Customer information who rated these restaurant
3. The rating information with text reviews

The most interesting problem I found was to predict the rating of a restaurant by a customer given her review comments. This is a NLP problem. To solve this 'The rating information with text reviews' file is enough but it is a very large file, around 2.5 GB.





#Read 2:

Approach - These scripts have to be run in the given order. 

Step 1. Read the business information, review information in pandas data frame

   - The first script to be run is called 'Reading Data .ipynb'. The input to this script is 'review.json'. The output of this file is 'review1', 'review2' ... 'review23' in pickle format. The input file was 2.5 GB so I had to chunk it into smaller pieces to be able to work with the data on my 4 GB memory.
   
Step 2. Generated the vector representation of the words in the review using word2vec model. 

We do not necessarily have to train  our own model. Instead we can use a global library. However, I preferred to do so as it gives more insight into the problem as to what is going on.

The second script to be run is called 'Word2vec model.ipynb'. The input to this script was 'review1', 'review2', 'review3', 'review4', 'review5'. I  did not use any more chunked review files because I had already around 1/2 million reviews. The output was a model called '100features_10minwords_10context'. 
    
I tested the accuracy of my vector representation by asking the model several questions. One of them is mentioned at the end of the script as to which is the most dissimilar between food, water, dinner and italy. The answer was italy.

Step 3. Mapped the individual reviews to the word vectors and stored them in the pandas dataframe. 

The third script to be run is called 'text to vector conversion as per trained word2vec.ipynb'. The input to this script was model generated in the previous script called '100features_10minwords_10context'. The output is a pickle file called 'business_data' with vectorised words for one of the chunked review files -'review1'.

Step 4.  Prepared the input and output data for answering the modelling question. 

The fourth script is is called - 'data preparation.ipynb'. The input is 'business_data' and the output are two files 'X.npy' and 'y.npy'. 

Step 5. Trained a neural net based  model in Keras to predict ratings - a multinomial classification problem. 

The relevant file is 'neural net(LSTM) model for rating.ipynb'. The input are two files 'X.npy' and 'y.npy'.  The output is the prediction for the test set. I treated 1-5 rating as multinomial classification problem. I used LSTM version of deep neural net to accomodate NLP based input which could be of varied length and complex. I measured my accuracy using AUC which is explained in the last part(Read 4)






#Read 3:

Key challenges:

1. My local system did not have enough RAM to support the operations on available data size.
2. I chunked data into small pieces and used first of these chunks
3. Even the review text beyond 30 words was not handled with my memory so I truncated the text after 30 words.





#Read 4:

For AUC calculation on multinomial classification, I chose the highest scores among scores for all 5 classes of rating. For the true class, I used 1 if the predicted rating matched the true rating. Otherwised I used 0.  The AUC was .71 which is quite reasonable. I found that ratings 4 or 5 were easily predicted, even without using the entire data set (due to memory limitations).
