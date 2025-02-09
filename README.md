# Introduction
In this project will be dealt problem of review binary sentiment classification. Firstly will be performed exploratory data analysis of given data. Next will be performed data preparation, data cleaning and feature engineering. After that models will be built and evaluated. Then the best model will be chosen and used for predicting sentiments of unseen during training reviews. All steps of the project will be commented and explained. All steps required from reader to launch the project will also be explained in details.

## Project Structure:
This project has a modular structure, where each folder has a specific duty.

```
FINALPROJECT_EPAM
├── data                                    # Data files
│   ├── processed                           # Folder for storing cleaned, prepared, preprocessed data
│   └── raw                                 # Folder for storing raw, uncleaned data
│
├── notebooks                               # Folder for storing .ipynb notebook of the project
│   └── sol_notebook.ipynb 
│
├── outputs                                 # Output files
│   ├── predictions                         # Folder for storing outputs(predictions) of inference script
│   └── models                              # Folder for storing trained models
│
├── src                                     # Main project files
│   ├── train                               # Scripts and Dockerfile used for training
│   │   ├── Dockerfile
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   └── data_engineering.py
│   │
│   ├── inference                           # Scripts and Dockerfile used for inference
│   │   ├── Dockerfile
│   │   ├── requirements_inference.txt
│   │   └── run_inference.py
│   ├── create_directories.py               # Script used for creating necessary directories  
│   └── data_loader.py                      # Script used for downloading raw data in /data/raw/ folder
│
├── README.md                               # Description of the project, instruction how to launch a project 
└── requirements.txt                        # File listing all necessary libraries used for training      
```

# Data Science Part
## Conlusions and Insights from EDA
*Data Set Format*   
The dataset was provided in two separate .csv files. The first one contains data needed for training. The second one contains testing data.
Each of the datasets contains two columns. First column is a text review to a film. Second column is a target feature - review sentiment value, that takes on values: "positive" or "negative", depending on the mood expressed in corresponding review.

*General Statistics About Train and Test Sets*    
- There are 40000 of reviews in training dataset. In total there were used 9254510 non-unique tokens in corpus. The shortest review consists of 4 tokens. The largest review consists of 2470 tokens. On average a review in train dataset has 231 tokens. The standard deviation of reviews lengths in training dataset is 171 tokens.
Reviews with negative sentiments and with positive sentiments on average have the same number of tokens in them , around 230 words for both.

- There are 10000 of reviews in testing dataset. In total there were used 2303337 non-unique tokens in testing corpus. The shortest review consists of 6 tokens. The largest review consists of 2108 tokens. On average a review in test dataset has 230 tokens(similiar, almost the same value was for training set). The standard deviation of reviews lengths in testing dataset is 172 tokens(similiar, almost the same value was for training set).
Reviews with negative sentiments and with positive sentiments on average have the same number of tokens in them , around 230 tokens for both(the same how was for training set).

- Main statistics for training and testing sets are very similiar, almost the same.

*Distribution of Reviews Lengths*  
Distribution of random variable "Length of review" resembles exponential distribution (the same results for training and testing sets).
Reviews with length more than 1000 tokens could be considered as outliers, unusually large reviews. But for this work such reviews will not be deleted from dataset, as they still might convey very important information during modeling.

*Missing values*  
There are no missing values in both training and testing datasets.

*Distribution of sentiment feature*  
In both training and testing sets, half of all reviews is marked to have negative sentiment and other half is of positive sentiment. In this sense both datasets are balanced. That also means that accuracy metric could be succesfully used for models perfomance evaluating.

*Unique tokens in corpuses*  
Number of unique tokens used in training corpus(380518 tokens) is twice(approximately) as big as number of unique words used in test corpus(158916 tokens).

*Most common words in datasets*  
In both train and test sets most common, popular are in general stop-words. Evidently, they are supposed to be deleted. Can be considered as noise.
Expression "/><br" is also very frequently appearing expression in corpus - also definitely should be deleted. Can be considered as noise.

Other most popular non stop-words for negative and positive reviews (in training set) are in general the same. Exceptions are words: "bad" - only distinctive popular word for negative reviews; and "great", "love", "first" - only distinctive popular words of positive reviews.   
The same situation with testing set, most popular non stop-words are the same for positive and negative reviews, though again there are some exceptions:  
distinctive popular words of positive reviews: "great", "love", "first"; distinctive popular word of negative reviews is only: "bad".

Train and test sets have almost the same most popular non stop-words. So can be very roughly said that these sets are coming from the same distribution, the training set corresponds to testing(or in other side).

Distinctive pairs of words for positive reviews(TRAIN SET): "one best", "highly recommend", "great movie", "must see", "great film", "like movie", "love movie";  
Distinctive pairs of words for negative reviews(TRAIN SET): "bad movie", "waste time", "low budget", "bad film", "one bad", "horror movie", "movie bad", "bad act", "horror film", "bad guy";  

Can be noticed that popular two words phrases for positive reviews contain positive, "nice" words, that express positive emotions. Opposite fact holds for popular phrases of negative reviews. That was expectable.  

Such pairs seem to be reasonable factors at separating positive review from negative.

Distinctive pairs of words for positive reviews(TEST SET): "one best", "highly recommend", "great movie", "must see", "great film", "movie one", "like movie", "love movie";  
Distinctive pairs of words for negative reviews(TEST SET): "bad movie", "waste time", "low budget", "one bad", "bad film", "horror movie", "movie bad", "horror film", "bad guy";

Overall, test and train sets have common most popular two words phrases for negative and positive reviews. That again proves that both sets are related, are coming from the same distribution.

*Duplicates*  
There are duplicate reviews in training and testing sets. Number of them is very small: for training set is 272 and for testing set is 13. Duplicate reviews should be deleted, so only one stays (out of all duplicates).

*Mutual reviews in test and train sets*  
There are same, mutual reviews (133) in test and train sets. Such reviews should be deleted from one of the sets.

## Description of Feature Engineering and Data Preparation
*Dropping duplicates (all kinds of)*    
Will be dropped dublicates that exist separately in test and train sets.
Mutual reviews (for train and test sets) will be deleted from train set. It is not acceptable for train and test sets to intersect.

*Deleting unnecessary characters*    
Reviews could contain: URL links, numbers (11, 0, 34, 2005, etc), punctuation and special characters (,^&*;/$) - all such characters/expressions should be deleted from each review. These characters, parts of review texts do not contain information related, helpful for modeling. That is why they are being deleted. Converting all words in each review to lower case will then be applied(so to make text uniform; for example to make words: "Book", "book", "BOOK", "BooK" - represent the same word "book", but not different words).

*Tokenization*    
Will be applied tokenization to every review text - meaning that every review will be divided by words of which it consists. This will make it easier to preprocess text in next steps.

*Stop-words filtering*    
Stop-words like: "the", "a", "is", "they", "me", ... - should be deleted. Because they are ubiquitous across all reviews , do not contain helpful, vital information that could be used during modeling. Such words could be considered as noise.

*Stemming vs Lemmatization*    
In the project will be tried two techniques: stemming and lemmatization. Then they will be compared (after modeling) in order to discover which of them works better, gives better results (with which of technique models achieve higher accuracy). Brief description of them: these two techniques are used to truncate words, reduce their form to the smallest possible. Use of them would make vocabulary of all reviews much smaller, would reduce feature space, which is beneficial. Stemming is based on deleting ending of the word. Lemmatization is based on finding the smallest original correct lexical form of the word. Lemmatization is much complicated and precise technique.

Stemming gives very rough results, for example it transforms words: accident -> accid; revival -> reviv; theatre -> theatr; movies -> movi; amusing -> amus. Which are not properly spelled english words. Basically it comes up with short, unmeaningfull words.  

From other side, we get almost perfect truncation of words after applying lemmatization, they are correctly transformed to their correct shortest forms. Though, sometimes, technique might not shorten word when it is applicable, possible.

*Deleting tokens of length less or equal to 2*  
We need do delete symbol "br", which was postulated earlier in EDA. Overall, we do not need any tokens of length <= 2, most likely they do not explain anything, are not valuable during modeling.

*Vectorization*  
Two different techniques for vectorization  will be used: Count Vectorizer and TF-IDF. After training models, the best vectorization technique will be chosen. Difference between Count Vectorizer and TF-IDF:   
- Count Vectorizer is a technique of converting text to vector, that relies solely on counting frequencies of each word appearing in peace of text(in short);
- TF-IDF Vectorizer still converts text to vector, but it not only counts word frequency in text, but also reduces importance of words that are too common across the whole corpus. It is more sophisticated technique of text vectorization(comparing to previous one). It is expected for it to give better results during modeling, as it conveys more information to model, comparing to Count Vectorizer.

*Preparing target variable*    
Target feature needs to be extracted from train and test datasets(column "sentiment"). After that its values will mapped: "positive" -> 1 and "negative" -> 0.

## Reasonings on Model Selection  
For baseline model was chosen Naive Bayes model, as it is very simple, easy and fast classification algorithm. After that was tried Logistic Regression, as it is also pretty fast, easy-interpretable and light algorithm, suitable for classification. More advanced algorithms: Random Forest Classifier and XGBoost classifier were tried at the end, to investigate if they could reach better accuracy than previous two algortihms.
The best model can be considered such that is as simple as possible, and that simultaneously has the highest accuracy score on test set.

## Conclusions of Modeling and Overall Perfomance Evaluation  
Models that were using lemmatized dataset were slightly better than those that were trained on stemmatized data. Though mostly the diference in perfomance is very small, could be considered insignificant.

Naive Bayes and Logistic Regression models that were trained on data vectorized with TF-IDF technique showed slightly higher accuracy scores(compared to same type models trained on vectorized data with Count technique). The opposite holds for Random Forest Clasifiers and XGBoost Classifiers(which had higher accuracy scores with data vectorized using Count technique), though in this case the differences is very small, again can be considered insignificant.

Overall, all models(Naive Bayes, Logistic Regression, Random Forest Classifier and XGBoost Classifier) regarded (in this project, notebook) have shown accuracy equal to or higher than 85%.

Naive Bayes and Logistic Regression that were trained on lemmatized, tf-idf vectorized dataset had the highest accuracy scores(86% and 89% respectfully) among same type of models. 

Random Forest Classifier and XGBoost Classifier that were trained on lemmatized dataset vectorized with count technique had the highest accuracy scores(85% and 86% respectfully) among same type of models.

Afterall, the best model was Logistic Regression trained on lemmatized, tf-idf vectorized dataset with accuracy 89%. To achieve this reuslt was also applied hyperparameters tuning(found optimal hyperparameters: C=2.15; penalty=l2; solver=liblinear). (Meaning of hyperparameters: C - specifies regularization strength; penalty - specifies regularization type; solver - specifies the algorithm that is used during optimization(for such high dimensional datasets , as in our case, it is recommended to choose: liblinear or saga))  

## Potential business applications and value for business
Sentiment analysis, for example, could help in:   
- unraveling general attitude of customers towards comany's particular product, its new feature, etc. That could help to determine, adjust tactic of product development;  
- identifying customer demandings, needs;
- pointing the company whether direction of its development was chosen correctly;

In this specific case, a film streaming service could analyze users comments towards each particular film and make predictions whether to suggest this film to a bigger audience or not. Or, a filmmaking company could analyze commentings of produced films and gather, develop some knowledge about what upcoming films should contain in order to be liked by viewers.

## Notebook
I encourage you to look at notebook(located in "/notebooks" directory), where you could find all above explanations from this README, but also complemented with code, visualizations for better understanding.

# MLE Part: Launching project scrpits
## Data Downloading
Start with data downloading. To download data you need to run "data_loader.py" file, which is located in directory "./src".
Just open Command Prompt in folder, where "data_loader.py" exists and use this command: 
```bash
python data_loader.py
```
After that, test and train files will appear in directory "./data/raw".

## Preparing project directories
Directories "data/processed", "outputs/predictions", "outputs/models" should be created at the beginning. Those are necessary folders that will be used in next steps during training and inference stages. 
Just open Command Prompt in folder, where "create_directories.py" exists and use this command:
```bash
python create_directories.py
```

## Running training scripts
1. In order to prepare data, run models and investigate their perfomance, first of all you need to create image of a container(in which all of these procedures will be flowing). Therefore, you need to run this command(use Command Prompt from root directory of project):
```bash
docker build -f src/train/Dockerfile -t training_image .
```

2. Next, you need to launch the docker container using built image from previous step(it will take around 30 minutes). For this purpose, use command:
```bash
docker run training_image
```
If everything ran succesfully, you would be able to see different comments(and, at the end, logs of trained models accuracy scores calculated on test set).

NOTE: All Logistic Regression models defined in train.py have different hyperparameters, comparing to those trained in notebook. Those defined in train.py file, have hyperparameters set up to values found after hyperparameters tuning in notebook.

NOTE: XGBoost classifiers were commented(in file train.py), because it was taking too long to train them all(around 20 minutes for each, and there are four XGBoosts). If you want you can uncomment and check their perfomance also. Though, all four XGBoosts were trained in notebook, you can also check them there.

3. After that, please run next two commands(in order to transfer trained models, and prepared data from container to your local machine):
```bash
docker cp <container_id>:/app/models/. outputs/models/
```
```bash
docker cp <container_id>:/app/data/processed/. data/processed/
```
Replace <container_id> - with your real built container's id.

You need to do this in order to make any predictions, inferences defined in next steps.

NOTE: In case you do not want to use Docker, you may run training script on your local machine. To do this, you need: 
1. Comment current version of variable DIR, and uncomment second version: 
```bash
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
```
in all three files: data_engineering.py, train.py, preprocess.py

2. Comment lines: "82. subprocess.run(["python", "/app/train/preprocess.py"], check=True)" and "86. subprocess.run(["python", "/app/train/data_engineering.py"], check=True)" in train.py

3. Comment current version of variable defining where to save models, and uncomment second version: 
```bash
model_save_path = os.path.join(BASE_DIR, "outputs", "models")
```
in train.py

4. Open Command Prompt in folder, where data_engineering.py, train.py, preprocess.py exist and run following command:
```bash
python preprocess.py; python data_engineering.py; python train.py 
```

## Running inference scripts
1. In this step, will be used best model out of all trained for predicting sentiments of unseen during training reviews.
First of all, image of a container, in which inferencing will be running, needs to be created, it can be done with command(use Command Prompt from root directory of project): 
```bash
docker build -f src/inference/Dockerfile -t inference_image .
```

2. Next, you need to launch the docker container using built image from previous step. For this purpose, use command:
```bash
docker run inference_image
```
After this set with predictions and raw reviews will be saved inside container.

3. In order to transfer dataset with inferences from container to your local machine use following command:
```bash
docker cp <container_id>:/app/outputs/predictions/. outputs/predictions/
```
Replace <container_id> - with your real built container's id(used for inferencing).
Data from this container will be saved to directory /outputs/predictions. 

NOTE: Output data with predictions will have two columns: review and sentiment. Sentiment column could take on values 1 and 0(1 for positive and 0 for negative sentiments).  

NOTE: In case you do not want to use Docker, you may run inference script on your local machine. To achieve this, you will need: 
1. Comment current version of variable DIR, and uncomment second version: 
```bash
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
```
in run_inference.py file

2. Open Command Prompt in folder, where run_inference.py exists and run following command:
```bash
python run_inference.py
```