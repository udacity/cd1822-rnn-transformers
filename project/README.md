### Project Overview:

The objective of this project is to analyze the sentiment of movie reviews in three different languages - English, French, and Spanish. We have been given 30 movies, 10 in each language, along with their reviews and synopses in separate CSV files named `movie_reviews_eng.csv`, `movie_reviews_fr.csv`, and `movie_reviews_sp.csv`.

- The first step of this project is to read data from all the .csv files and create a single pandas dataframe. This dataframe should have the following columns - **Title**, **Year**, **Synopsis**, **Review**, and **Original Language**.

- The next step is to convert the French and Spanish reviews and synopses into English. This will allow us to analyze the sentiment of all reviews in the same language. We will be using pre-trained transformers from HuggingFace to achieve this task.

- Finally, we will use pretrained transformers from HuggingFace to analyze the sentiment of each review. The sentiment analysis results (**Positive** or **Negative**) will be added to the dataframe in a new column called **Sentiment**.

The output of the project will be a CSV file with a header row that includes column names such as **Title**, **Year**, **Synopsis**, **Review**, **Sentiment**, and **Original Language**. The **Original Language** column will indicate the language of the review and synopsis (*en/fr/sp*) before translation. The dataframe will consist of 30 rows, with each row corresponding to a movie.

### Tools used:

- Pandas: for data manipulation and analysis
- HuggingFace Transformers: for natural language processing tasks, such as translation and sentiment analysis
- PyTorch: for building and training machine learning models

### Skills learned:

- Data cleaning and manipulation using Pandas
- Natural language processing techniques, such as translation and sentiment analysis, using HuggingFace Transformers
- Building and training machine learning models using PyTorch
Integration of multiple tools and libraries to solve a complex problem

Overall, this project combines data manipulation, natural language processing, and machine learning techniques to perform multiple tasks on a dataset.




