Developed end-to-end sentiment analysis pipeline analyzing 500+ tweets from USA users during COVID-19 lockdown to extract public emotional responses and opinion trends across different lockdown phases
Implemented NLP preprocessing workflow using NLTK and Sastrawi libraries, performing tokenization with RegexpTokenizer, stop-word removal, Porter Stemming, and text normalization to clean unstructured social media data
Built and deployed Multinomial Naive Bayes classifier for sentiment classification (positive/negative/neutral), utilizing CountVectorizer with n-gram features (unigrams and bigrams) for feature extraction from tweet text
Engineered sentiment scoring system using VADER (Valence Aware Dictionary and sEntiment Reasoner) with compound polarity scores, achieving threshold-based classification (≥0.05 positive, ≤-0.05 negative)
Created interactive visualization dashboard using Streamlit, Plotly, and Altair to display real-time sentiment distributions, temporal trends, and geographic patterns of public discourse during pandemic
Performed statistical correlation analysis between news sentiment and social media sentiment using Pearson and Kendall-Tau correlation coefficients, with rolling window and exponential weighted moving average (EWM) techniques
Integrated multi-source data pipeline combining Twitter API (Twython), Google News API (GNews), and market data (yFinance) to analyze sentiment alignment across news media, social media, and financial markets
Deployed production-ready web application with user authentication, multi-page navigation, and real-time data processing capabilities handling 2000+ tweets per search query

Technical Stack: Python, scikit-learn, NLTK, Pandas, NumPy, Streamlit, Plotly, Seaborn, Matplotlib, VADER Sentiment, Twython (Twitter API), SciPy, Sastrawi, RegexpTokenizer

Key Achievements to Highlight:
✅ 500+ tweet dataset with temporal analysis
✅ Multi-model comparison (Naive Bayes primary model)
✅ Advanced NLP techniques (stemming, n-grams, TF-IDF)
✅ Statistical validation (correlation analysis, p-values)
✅ Full-stack deployment (Streamlit dashboard with authentication)
✅ Real-time processing capability
✅ Multi-source integration (Twitter + News + Financial data)
