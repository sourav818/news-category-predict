# Libraries and Techniques Used in News Classification Project

## Core Libraries

### Data Manipulation and Analysis
- **pandas (v1.5.3)**
  - Primary data manipulation library for structured data processing
  - Key functionalities used:
    ```python
    # DataFrame operations for data handling
    # Reading CSV file with news articles and their categories
    data = pd.read_csv("dataset/BBC_News_Train.csv")  
    
    # Converting categorical labels to numerical IDs (e.g., 'business' -> 0)
    # factorize() creates unique integer codes for each category
    data['CategoryId'] = data.Category.factorize()[0]  
    
    # Handling missing values in text data by replacing NaN with empty string
    # This ensures no null values that could break the preprocessing pipeline
    data['Text'] = data['Text'].fillna('')  
    
    # Analyzing category distribution to check dataset balance
    # Returns count of articles in each category
    category_distribution = data.groupby('Category').count()  
    ```
  - Benefits: 
    - Efficient handling of large datasets
    - Built-in data analysis tools
    - Easy data manipulation and cleaning functions
    - Seamless integration with other data science libraries

- **numpy (v1.24.3)**
  - Foundation for numerical computing in Python
  - Key functionalities:
    ```python
    # Creating feature matrices for machine learning
    # Initialize empty matrix for storing TF-IDF features
    # n_samples: number of documents
    # n_features: size of vocabulary
    feature_matrix = np.zeros((n_samples, n_features))  
    
    # Statistical operations for data analysis
    # Calculate mean TF-IDF scores across documents
    mean_tfidf = np.mean(feature_matrix, axis=0)
    # Calculate standard deviation for feature scaling
    std_tfidf = np.std(feature_matrix, axis=0)
    
    # Array operations for model predictions
    # Convert predicted probabilities to class labels
    predicted_classes = np.argmax(prediction_probabilities, axis=1)
    ```
  - Used for: 
    - Efficient array operations
    - Mathematical computations
    - Statistical analysis
    - Memory-efficient data structures

### Natural Language Processing (NLP)
- **nltk (v3.8.1)**
  - Comprehensive NLP toolkit for text processing
  - Components used:
    1. **nltk.corpus.stopwords**
       ```python
       # Import and setup stopwords for text cleaning
       from nltk.corpus import stopwords
       
       # Download required NLTK data (run once)
       nltk.download('stopwords')
       
       # Create set of English stopwords for efficient lookup
       stop_words = set(stopwords.words('english'))
       
       # Example of stopwords removal
       text = "This is a sample news article about business"
       words = text.split()
       # Remove common words that don't carry significant meaning
       filtered_words = [word for word in words if word.lower() not in stop_words]
       # Result: ['sample', 'news', 'article', 'business']
       ```
       - Purpose: Remove common words that don't contribute to article classification
       - Impact: Reduces noise and improves model performance
    
    2. **nltk.tokenize**
       ```python
       from nltk.tokenize import word_tokenize
       
       # Download tokenizer data (run once)
       nltk.download('punkt')
       
       # Example of text tokenization
       text = "Breaking news: Tech company launches new AI product!"
       words = word_tokenize(text)
       # Result: ['Breaking', 'news', ':', 'Tech', 'company', 'launches', 
       #          'new', 'AI', 'product', '!']
       
       # Advanced tokenization with sentence splitting
       from nltk.tokenize import sent_tokenize
       sentences = sent_tokenize(text)
       # Useful for analyzing sentence structure and context
       ```
       - Purpose: Break text into individual tokens (words and punctuation)
       - Benefits: 
         - Preserves important punctuation
         - Handles contractions properly
         - Maintains sentence boundaries
    
    3. **nltk.stem.PorterStemmer**
       ```python
       from nltk.stem import PorterStemmer
       stemmer = PorterStemmer()
       
       # Example of word stemming
       words = ['running', 'runs', 'runner', 'ran']
       stemmed_words = [stemmer.stem(word) for word in words]
       # Result: ['run', 'run', 'runner', 'ran']
       
       # Real-world example with news text
       text = "Companies are investing in artificial intelligence technologies"
       tokens = word_tokenize(text)
       stemmed_text = ' '.join([stemmer.stem(token) for token in tokens])
       # Result: "compani are invest in artifici intellig technolog"
       ```
       - Purpose: Reduce words to their root form
       - Benefits:
         - Reduces vocabulary size
         - Groups similar words together
         - Improves feature extraction efficiency

- **wordcloud (v1.9.2)**
  - Advanced text visualization library
  - Usage:
    ```python
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    # Create and configure word cloud generator
    wordcloud = WordCloud(
        width=800,           # Width of the canvas
        height=400,          # Height of the canvas
        background_color='white',
        min_font_size=10,    # Minimum font size for words
        max_font_size=150,   # Maximum font size for words
        max_words=200,       # Maximum number of words to display
        collocations=False   # Avoid repeating word pairs
    ).generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide axes
    plt.title('Most Common Words in News Articles')
    plt.show()
    
    # Save the word cloud
    wordcloud.to_file('news_wordcloud.png')
    ```
  - Features:
    - Size of words proportional to their frequency
    - Customizable colors and layouts
    - Support for custom masks and shapes
  - Applications:
    - Visualize most common terms in each news category
    - Identify key topics and themes
    - Create engaging visual representations of text data

### Machine Learning (scikit-learn v1.3.0)
- **Feature Extraction**
  - **TfidfVectorizer**
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create and configure TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,    # Limit vocabulary to top 5000 terms
        min_df=5,             # Ignore terms appearing in < 5 documents
        max_df=0.7,           # Ignore terms appearing in > 70% of docs
        stop_words='english', # Remove English stop words
        ngram_range=(1, 2),   # Include both unigrams and bigrams
        strip_accents='unicode', # Remove accents
        lowercase=True        # Convert all text to lowercase
    )
    
    # Transform text data to TF-IDF features
    X = vectorizer.fit_transform(texts)
    
    # Get feature names (words) for interpretation
    feature_names = vectorizer.get_feature_names_out()
    
    # Example: Get top terms for a document
    def get_top_terms(document_index, n_terms=5):
        tfidf_scores = X[document_index].toarray()[0]
        top_indices = tfidf_scores.argsort()[-n_terms:][::-1]
        return [(feature_names[i], tfidf_scores[i]) for i in top_indices]
    ```
    - Parameters explained:
      - max_features: Controls vocabulary size to prevent memory issues
      - min_df: Removes rare terms that might be typos or not meaningful
      - max_df: Removes too common terms that don't help in classification
      - ngram_range: Captures word combinations for better context
    - Benefits:
      - Accounts for both term frequency and importance
      - Reduces impact of common words
      - Creates sparse matrices for efficient processing

- **Classification Models**
  1. **LogisticRegression**
     ```python
     from sklearn.linear_model import LogisticRegression
     
     # Create and configure logistic regression model
     lr_model = LogisticRegression(
         C=1.0,               # Inverse of regularization strength
         max_iter=1000,       # Maximum iterations for convergence
         multi_class='ovr',   # One-vs-rest strategy for multiclass
         class_weight='balanced', # Handle class imbalance
         random_state=42      # For reproducibility
     )
     
     # Train the model
     lr_model.fit(X_train, y_train)
     
     # Get feature importance
     feature_importance = pd.DataFrame({
         'feature': feature_names,
         'importance': abs(lr_model.coef_[0])
     }).sort_values('importance', ascending=False)
     ```
     - Advantages:
       - Fast training and prediction
       - Probabilistic output for confidence scores
       - Good for linearly separable text data
       - Easy to interpret feature importance

  2. **Support Vector Machine (SVM)**
     ```python
     from sklearn.svm import SVC
     from sklearn.preprocessing import StandardScaler
     
     # Scale features for better SVM performance
     scaler = StandardScaler(with_mean=False)  # Sparse matrix handling
     X_scaled = scaler.fit_transform(X)
     
     # Create and configure SVM model
     svm_model = SVC(
         kernel='rbf',        # Radial basis function kernel
         C=1.0,               # Regularization parameter
         probability=True,    # Enable probability estimates
         class_weight='balanced', # Handle class imbalance
         random_state=42
     )
     
     # Train with scaled features
     svm_model.fit(X_scaled, y)
     
     # Grid search for optimal parameters
     from sklearn.model_selection import GridSearchCV
     param_grid = {
         'C': [0.1, 1, 10],
         'kernel': ['rbf', 'linear']
     }
     grid_search = GridSearchCV(SVC(), param_grid, cv=5)
     grid_search.fit(X_scaled, y)
     ```
     - Benefits:
       - Effective in high-dimensional spaces (like text data)
       - Memory efficient with kernel trick
       - Robust against overfitting
       - Handles non-linear relationships

  3. **Naive Bayes**
     ```python
     from sklearn.naive_bayes import MultinomialNB
     
     # Create and configure Naive Bayes model
     nb_model = MultinomialNB(
         alpha=1.0,           # Laplace/Lidstone smoothing
         fit_prior=True,      # Learn class prior probabilities
         class_prior=None     # Custom class priors if needed
     )
     
     # Train the model
     nb_model.fit(X_train, y_train)
     
     # Get feature probabilities per class
     feature_probs = pd.DataFrame(
         nb_model.feature_log_prob_,
         columns=feature_names
     )
     
     # Predict with probability estimates
     probabilities = nb_model.predict_proba(X_test)
     ```
     - Advantages:
       - Very fast training and prediction
       - Works well with high-dimensional data
       - Particularly effective for text classification
       - Requires less training data
     - Considerations:
       - Assumes feature independence
       - Works well with discrete features

- **Model Evaluation Tools**
  ```python
  from sklearn.metrics import accuracy_score, confusion_matrix
  from sklearn.metrics import classification_report
  from sklearn.model_selection import cross_val_score
  import seaborn as sns
  
  # Comprehensive model evaluation
  def evaluate_model(model, X_test, y_test, model_name="Model"):
      # Make predictions
      y_pred = model.predict(X_test)
      
      # Calculate accuracy
      accuracy = accuracy_score(y_test, y_pred)
      print(f"{model_name} Accuracy: {accuracy:.4f}")
      
      # Detailed classification metrics
      print("\nClassification Report:")
      print(classification_report(y_test, y_pred))
      
      # Create confusion matrix visualization
      plt.figure(figsize=(10, 8))
      cm = confusion_matrix(y_test, y_pred)
      sns.heatmap(cm,
                  annot=True,      # Show numbers in cells
                  fmt='d',         # Use integer format
                  cmap='Blues',    # Use blue color palette
                  xticklabels=model.classes_,
                  yticklabels=model.classes_)
      plt.title(f'Confusion Matrix - {model_name}')
      plt.xlabel('Predicted')
      plt.ylabel('True')
      plt.show()
      
      # Cross-validation scores
      cv_scores = cross_val_score(model, X, y, cv=5)
      print(f"\nCross-validation scores: {cv_scores}")
      print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
      
      return {
          'accuracy': accuracy,
          'cv_scores': cv_scores,
          'confusion_matrix': cm
      }
  ```
  - Features:
    - Comprehensive evaluation metrics
    - Visual representation of results
    - Cross-validation for robust performance estimation
    - Detailed per-class performance analysis

### Visualization Libraries
- **matplotlib (v3.7.1)**
  ```python
  import matplotlib.pyplot as plt
  
  # Create detailed category distribution visualization
  def plot_category_distribution(data):
      # Set figure size and style
      plt.figure(figsize=(12, 6))
      plt.style.use('seaborn')
      
      # Create bar plot
      category_counts = data['Category'].value_counts()
      bars = plt.bar(category_counts.index, category_counts.values)
      
      # Customize appearance
      plt.title('Distribution of News Articles Across Categories',
               fontsize=14, pad=20)
      plt.xlabel('News Category', fontsize=12)
      plt.ylabel('Number of Articles', fontsize=12)
      
      # Add value labels on top of bars
      for bar in bars:
          height = bar.get_height()
          plt.text(bar.get_x() + bar.get_width()/2., height,
                  f'{int(height)}',
                  ha='center', va='bottom')
      
      # Rotate x-labels for better readability
      plt.xticks(rotation=45)
      
      # Add grid for better readability
      plt.grid(True, axis='y', linestyle='--', alpha=0.7)
      
      # Adjust layout to prevent label cutoff
      plt.tight_layout()
      
      # Save the plot
      plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
      plt.show()
  ```

- **seaborn (v0.12.2)**
  ```python
  import seaborn as sns
  
  # Create detailed confusion matrix visualization
  def plot_confusion_matrix(y_true, y_pred, labels):
      # Create confusion matrix
      cm = confusion_matrix(y_true, y_pred)
      
      # Set up the matplotlib figure
      plt.figure(figsize=(10, 8))
      
      # Create heatmap with seaborn
      sns.heatmap(cm,
                  annot=True,        # Show numbers in cells
                  fmt='d',           # Use integer format
                  cmap='Blues',      # Use blue color palette
                  square=True,       # Make cells square
                  xticklabels=labels,
                  yticklabels=labels)
      
      # Add title and labels
      plt.title('Confusion Matrix for News Classification',
               pad=20, fontsize=14)
      plt.xlabel('Predicted Category', fontsize=12)
      plt.ylabel('True Category', fontsize=12)
      
      # Rotate x-labels for better readability
      plt.xticks(rotation=45)
      plt.yticks(rotation=45)
      
      # Add colorbar label
      cbar = plt.gca().collections[0].colorbar
      cbar.set_label('Number of Articles', rotation=270, labelpad=15)
      
      # Adjust layout
      plt.tight_layout()
      
      # Save the visualization
      plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
      plt.show()
      
      # Print additional metrics
      print("\nPer-category Accuracy:")
      for i, category in enumerate(labels):
          category_accuracy = cm[i,i] / cm[i,:].sum()
          print(f"{category}: {category_accuracy:.2%}")
  ```

## Text Preprocessing Pipeline
```python
def preprocess_text(text):
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. Tokenization
    tokens = word_tokenize(text)
    
    # 3. Remove punctuation and numbers
    tokens = [token for token in tokens if token.isalpha()]
    
    # 4. Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # 5. Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)
```

## Model Training and Evaluation Pipeline
```python
# 1. Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Model training
model.fit(X_train, y_train)

# 3. Prediction
y_pred = model.predict(X_test)

# 4. Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Dataset Details
- **BBC News Dataset**
  - Size: 2225 documents
  - Categories: 5 (business, entertainment, politics, sport, tech)
  - Features: 
    - Text content (raw news articles)
    - Category labels
  - Split: 80% training, 20% testing

## Performance Metrics
- Accuracy: ~96% (with LogisticRegression)
- F1-Score: ~95% (weighted average)
- Cross-validation scores: 5-fold CV with mean accuracy of ~94%

## Best Practices Implemented
1. **Data Preprocessing**
   - Handling missing values
   - Text normalization
   - Removing noise (stopwords, punctuation)

2. **Feature Engineering**
   - TF-IDF vectorization with optimal parameters
   - Feature selection based on document frequency

3. **Model Selection**
   - Multiple models comparison
   - Hyperparameter tuning
   - Cross-validation for robust evaluation

4. **Error Analysis**
   - Confusion matrix visualization
   - Misclassification analysis
   - Feature importance analysis
