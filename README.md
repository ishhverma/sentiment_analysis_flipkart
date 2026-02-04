Sentiment Analysis of Flipkart Product Reviews

📌 About This Repository
This repository contains a complete sentiment analysis project using real-time Flipkart product reviews.
The dataset includes 8,518 reviews of the product “YONEX MAVIS 350 Nylon Shuttle”.
The objective is to classify reviews into Positive or Negative sentiments and understand customer pain points from negative feedback.

sentiment_analysis_flipkart/
│
├── dataset/
│   ├── data.csv
│   └── credits.txt
│
├── gpu.ipynb
│
└── README.md

Project Overview
📌 Objective
Predict the sentiment of a review (Positive / Negative)
Identify customer pain points from negative reviews

📌 Dataset
8,518 reviews from Flipkart
Features include:
Reviewer Name
Rating
Review Title
Review Text
Place of Review
Date of Review
Up Votes / Down Votes

🧹 Data Preprocessing
Steps performed on review text:
Remove special characters & punctuation
Remove stopwords
Lowercase normalization
Lemmatization / Stemming

🔍 Feature Engineering
Multiple embedding techniques were explored:
Bag of Words (BoW)
TF-IDF
Word2Vec
BERT Embeddings


You said:
Sentiment Analysis of Real-time Flipkart Product Reviews  Objective The objective of this project is to classify customer reviews as positive or negative and understand the pain points of customers who write negative reviews. By analyzing the sentiment of reviews, we aim to gain insights into product features that contribute to customer satisfaction or dissatisfaction.  Dataset A team of Data Engineers have already scraped real-time data from Flipkart website. They followed the instructions given in this documentation for extracting user reviews. Don’t scrape your own data. Use the dataset given below to solve the problem statement. Click Here to download the data.  The dataset consists of 8,518 reviews for the "YONEX MAVIS 350 Nylon Shuttle" product from Flipkart. Each review includes features such as Reviewer Name, Rating, Review Title, Review Text, Place of Review, Date of Review, Up Votes, and Down Votes.  Data Preprocessing 1. Text Cleaning: Remove special characters, punctuation, and stopwords from the review text. 2. Text Normalization: Perform lemmatization or stemming to reduce words to their base forms. 3. Numerical Feature Extraction: Apply techniques like Bag-of-Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), Word2Vec (W2V), and BERT models for feature extraction.  Modeling Approach 1. Model Selection: Train and evaluate various machine learning and deep learning models using the embedded text data. 2. Evaluation Metric: Use the F1-Score as the evaluation metric to assess the performance of the models in classifying sentiment.   Model Deployment 1. Flask or Streamlit App Development: Develop a Flask or Streamlit web application that takes user input in the form of a review and generates the sentiment (positive or negative) of the review. 2. Model Integration: Integrate the trained sentiment classification model into the Flask or Streamlit app for real-time inference. 3. Deployment: Deploy the Flask or Streamlit app on an AWS EC2 instance to make it accessible over the internet.  Workflow 1. Data Loading and Analysis: Gain insights into product features that contribute to customer satisfaction or dissatisfaction. 2. Data Cleaning: Preprocess the review text by removing noise and normalizing the text. 3. Text Embedding: Experiment with different text embedding techniques to represent the review text as numerical vectors. 4. Model Training: Train machine learning and deep learning models on the embedded text data to classify sentiment. 5. Model Evaluation: Evaluate the performance of the trained models using the F1-Score metric. 6. Flask or Streamlit App Development: Develop a Flask or Streamlit web application for sentiment analysis of user-provided reviews. 7. Model Deployment: Deploy the trained sentiment classification model along with the Flask or Streamlit app on an AWS EC2 instance. 8. Testing and Monitoring: Test the deployed application and monitor its performance for any issues or errors.   Submission Click here to submit your work.    write readme for this
ChatGPT said:
Below is a professional, well-structured README for your project. You can copy and paste it into a README.md file and adjust any details (like links or file names) as needed.

📌 Sentiment Analysis of Real-time Flipkart Product Reviews
🔍 Project Overview
This project aims to classify Flipkart customer reviews for the product “YONEX MAVIS 350 Nylon Shuttle” into Positive or Negative sentiments. The goal is to identify customer satisfaction drivers and uncover pain points from negative reviews.

The dataset contains 8,518 real-time reviews scraped by a data engineering team. This project focuses on building a robust sentiment analysis model and deploying it as a web application.

📦 Dataset
The dataset contains the following columns:

Column Name	Description
Reviewer Name	Name of the reviewer
Rating	Product rating given by the reviewer
Review Title	Title of the review
Review Text	Full review text
Place of Review	Location of reviewer
Date of Review	Date when review was posted
Up Votes	Upvotes received
Down Votes	Downvotes received
➡️ Dataset Source: Provided by the project dataset link (already scraped).
➡️ Note: Do not scrape your own data. Use only the provided dataset.

🧹 Data Preprocessing
1. Text Cleaning
Remove special characters

Remove punctuation

Remove stopwords

2. Text Normalization
Lemmatization or Stemming

Lowercasing

3. Feature Extraction
We experimented with multiple text representation techniques:

Bag of Words (BoW)

TF-IDF

Word2Vec

BERT embeddings

🧠 Modeling Approach
🔍 Model Selection
Various models were trained and evaluated, including:

Traditional ML Models
Logistic Regression

Naive Bayes

Random Forest

Support Vector Machine (SVM)

Deep Learning Models
LSTM

CNN

BERT-based Classifier

🏆 Evaluation Metric
F1-Score was used as the primary evaluation metric because it balances precision and recall and is suitable for imbalanced datasets.

🚀 Model Deployment
🖥️ Web Application
A Streamlit / Flask web app was developed that:

Takes user review input

Predicts sentiment (Positive / Negative)

Displays the result in real time

☁️ Deployment
The app is deployed on an AWS EC2 instance to make it accessible online.

🧭 Workflow Summary
Data Loading & Analysis

Data Cleaning & Preprocessing

Text Embedding (BoW, TF-IDF, Word2Vec, BERT)

Model Training & Selection

Model Evaluation (F1-Score)

Web App Development (Streamlit/Flask)

Deployment on AWS EC2

Testing & Monitoring

📌 Project Structure
├── data/
│   └── flipkart_reviews.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   └── modeling.ipynb
│
├── models/
│   └── sentiment_model.pkl
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── README.md
└── requirements.txt
🧾 How to Run Locally
1. Clone the repository
git clone <repo-url>
cd <repo-name>
2. Install dependencies
pip install -r requirements.txt
3. Run the Streamlit app
streamlit run app/app.py
🌐 Deployment
The web app is deployed on an AWS EC2 instance.
The application can be accessed through the public EC2 endpoint.

📌 Future Improvements
Implement sentiment classification into 3 classes (Positive, Neutral, Negative)

Improve accuracy using ensemble models

Add topic modeling to identify major customer concerns

Add dashboard for visualization of trends

🧑‍💻 Author
Your Name
Data Science Enthusiast

📌 License
This project is for academic purposes only.

If you want, I can also help you generate a project report, presentation slides, or deploy the app step-by-step.

You said:
Skip to content
ishhverma
sentiment_analysis_flipkart
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Owner avatar
sentiment_analysis_flipkart
Public
ishhverma/sentiment_analysis_flipkart
Go to file
t
Name		
ishhverma
ishhverma
dataset
da44357
 · 
2 minutes ago
credits.txt
dataset
2 minutes ago
data.csv
dataset
2 minutes ago
gpu.ipynb
Add files via upload
3 minutes ago
Repository files navigation
README
Add a README
Help people interested in this repository understand your project.

About
No description, website, or topics provided.
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Languages
Jupyter Notebook
100.0%
Footer
© 2026 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information

Explain change this 
ChatGPT said:
It looks like you're showing the GitHub repository interface and asking to explain and change the README or the repository description to make it more informative.

Here’s a clean, professional README and repository description you can use to replace the default “Add a README” section and improve the overall project presentation.

✅ Updated README for sentiment_analysis_flipkart
🎯 Project Title
Sentiment Analysis of Flipkart Product Reviews
📌 About This Repository
This repository contains a complete sentiment analysis project using real-time Flipkart product reviews.
The dataset includes 8,518 reviews of the product “YONEX MAVIS 350 Nylon Shuttle”.

The objective is to classify reviews into Positive or Negative sentiments and understand customer pain points from negative feedback.

📁 Repository Structure
sentiment_analysis_flipkart/
│
├── dataset/
│   ├── data.csv
│   └── credits.txt
│
├── gpu.ipynb
│
└── README.md
🧠 Project Overview
📌 Objective
Predict the sentiment of a review (Positive / Negative)

Identify customer pain points from negative reviews

📌 Dataset
8,518 reviews from Flipkart

Features include:

Reviewer Name

Rating

Review Title

Review Text

Place of Review

Date of Review

Up Votes / Down Votes

🧹 Data Preprocessing
Steps performed on review text:

Remove special characters & punctuation

Remove stopwords

Lowercase normalization

Lemmatization / Stemming

🔍 Feature Engineering
Multiple embedding techniques were explored:

Bag of Words (BoW)

TF-IDF

Word2Vec

BERT Embeddings

🧠 Modeling
Multiple ML and DL models were trained and evaluated:
Traditional Models
Logistic Regression
Naive Bayes
Random Forest
Support Vector Machine
Deep Learning Models
LSTM
CNN
BERT

Evaluation Metric
The models were evaluated using:
✅ F1-Score (Primary metric)

🌐 Web App Deployment
A web application was developed using:
Streamlit / Flask
Deployed on AWS EC2
Provides real-time sentiment prediction based on user input review

How to Run Locally
Install Requirements
pip install -r requirements.txt

Run Streamlit App
streamlit run app.py

🏷️ Repository Topics / Tags (Suggested)
You can add these topics to your repository:
sentiment-analysis
nlp
machine-learning
deep-learning
flipkart
streamlit
flask
aws-ec2

✅ Summary
This repository provides:
End-to-end sentiment analysis pipeline
Data preprocessing and feature engineering
Multiple modeling approaches
Deployment-ready web app
