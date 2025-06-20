import numpy as np
import pandas as pd
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import time

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Configuration
MODEL_FILE = "amazon_review_model.joblib"
SAMPLE_SIZE = 50000

# Enhanced preprocessing with negation handling
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags and special characters
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def train_model():
    st.info("Training Random Forest model...")
    
    # Enhanced training data with clearer positive examples
    positive_reviews = [
        ("very nice", "positive"),
        ("ok ok product", "neutral"),
        ("best budget 2 fit cooler nice cooling", "positive"),
        ("very bad product its a only a fan", "negative"),
        ("very good product", "positive"),
        ("the quality is good but the power of air is decent", "neutral"),
        ("great cooler excellent air flow and for this price its so amazing and unbelievablejust love it", "positive"),
        ("the cooler is really fantastic and provides good air flow highly recommended", "positive"),
        ("very good", "positive"),
        ("very bad cooler", "negative"),
        ("This product is absolutely amazing, I love everything about it!", "positive"),
        ("Excellent value for the money, works perfectly as described", "positive"),
        ("Highly recommend this product, the quality is outstanding", "positive"),
        ("Perfect in every way, far exceeded my expectations", "positive"),
        ("Best purchase I've made all year, worth every penny", "positive"),
        ("Outstanding performance and durability, very impressed", "positive"),
        ("Extremely satisfied with this high-quality product", "positive"),
        ("Works exactly as advertised, couldn't be happier", "positive"),
        ("Impressive quality for the price point, great value", "positive"),
        ("Would definitely buy again, fantastic product", "positive"),
        ("Superb craftsmanship and attention to detail", "positive"),
        ("Exactly what I needed, performs flawlessly", "positive"),
        ("Top-notch quality, better than I expected", "positive"),
        ("Very pleased with this purchase, excellent product", "positive"),
        ("Great investment, has improved my daily routine", "positive"),
        ("The cooling performance is exceptional, keeps items cold for days", "positive"),
        ("Energy efficient and quiet operation, perfect for my needs", "positive"),
        ("Setup was a breeze and the instructions were crystal clear", "positive"),
        ("After 3 months of heavy use, still works like new", "positive"),
        ("Customer service was prompt and resolved my issue immediately", "positive"),
        ("The design is both stylish and functional, a rare combination", "positive"),
        ("I was skeptical at first but this product proved me wrong", "positive"),
        ("Five stars without hesitation, would purchase again in a heartbeat", "positive"),
        ("This product has all the features I need and more", "positive"),
        ("The perfect balance between price and performance", "positive"),
        ("I've tried many alternatives but this one stands above the rest", "positive"),
        ("My family loves it and uses it every single day", "positive"),
        ("The manufacturer clearly put thought into the user experience", "positive"),
        ("Even better than the photos and description suggest", "positive"),
        ("I'm recommending this to all my friends and colleagues", "positive"),
        ("The premium materials are evident from the first touch", "positive"),
        ("It's rare to find such quality at this price point", "positive"),
        ("Exceeded my expectations in every possible way", "positive"),
        ("I can't imagine my life without this product now", "positive"),
        ("The attention to detail is remarkable for the price", "positive"),
        ("Worth every penny and then some", "positive"),
        ("I was pleasantly surprised by how good this product is", "positive"),
        ("The perfect solution to a problem I've had for years", "positive"),
        ("I've never written a review before but this deserves praise", "positive"),
        ("This product has saved me so much time and effort", "positive")
    ] * (SAMPLE_SIZE // 50)
    
    negative_reviews = [
        ("Terrible product, broke the first time I used it", "negative"),
        ("Complete waste of money, poor quality materials", "negative"),
        ("Did not work as described, very disappointed", "negative"),
        ("Battery died within hours of first use", "negative"),
        ("Extremely disappointed with this low-quality item", "negative"),
        ("Not worth even half the asking price", "negative"),
        ("Quality is much worse than expected", "negative"),
        ("Feels incredibly cheap and poorly made", "negative"),
        ("Misleading description, product is defective", "negative"),
        ("Arrived damaged and was late shipping", "negative"),
        ("Stopped working after just two days", "negative"),
        ("Flimsy materials that broke immediately", "negative"),
        ("Clearly used item sold as new", "negative"),
        ("Worst purchase decision I've ever made", "negative"),
        ("Total scam, avoid this product at all costs", "negative"),
        ("The cooling function doesn't work at all, completely useless", "negative"),
        ("Loud buzzing noise makes it impossible to use at night", "negative"),
        ("After one week of use, the motor burned out", "negative"),
        ("Customer service ignored all my emails and calls", "negative"),
        ("The product arrived with visible damage and missing parts", "negative"),
        ("The advertised capacity is completely inaccurate", "negative"),
        ("This was my third replacement and still doesn't work", "negative"),
        ("The design flaw makes it dangerous to use", "negative"),
        ("I've requested a refund but haven't heard back in weeks", "negative"),
        ("The product smells like chemicals even after multiple cleanings", "negative"),
        ("The buttons stopped responding after just a few uses", "negative"),
        ("The worst customer experience I've ever had", "negative"),
        ("False advertising - nothing like the product description", "negative"),
        ("The materials are so thin they tear easily", "negative"),
        ("I regret not reading the negative reviews before purchasing", "negative"),
        ("This product is a fire hazard - stay away", "negative"),
        ("The manufacturer clearly cut corners to save money", "negative"),
        ("I wouldn't take this product if it was free", "negative"),
        ("The product arrived in generic packaging with no instructions", "negative"),
        ("After 2 days of use, it started leaking everywhere", "negative"),
        ("The product is much smaller than advertised", "negative"),
        ("The mobile app required for setup doesn't work", "negative"),
        ("The warranty is worthless - company refuses to honor it", "negative"),
        ("This product has been nothing but trouble", "negative"),
        ("Save yourself the headache and buy from a reputable brand", "negative")
    ] * (SAMPLE_SIZE // 50)
    
    neutral_reviews = [
        ("It's okay, does the job but nothing special", "neutral"),
        ("Average product, meets basic requirements", "neutral"),
        ("Not bad, but not particularly good either", "neutral"),
        ("Meets minimum expectations but no more", "neutral"),
        ("Neither impressed nor disappointed", "neutral"),
        ("Adequate for the price but could be better", "neutral"),
        ("Standard quality, exactly as expected", "neutral"),
        ("Functional but nothing to get excited about", "neutral"),
        ("No major complaints but no praises either", "neutral"),
        ("Does the basic job it's supposed to do", "neutral"),
        ("The cooling works but not as well as I hoped", "neutral"),
        ("It gets the job done but the design could be improved", "neutral"),
        ("Performance is average for this price range", "neutral"),
        ("Works fine but the noise level is higher than expected", "neutral"),
        ("Does what it says but the setup was confusing", "neutral"),
        ("Not terrible but not great either - middle of the road", "neutral"),
        ("I have mixed feelings - some good aspects, some bad", "neutral"),
        ("It's fine for occasional use but not heavy duty", "neutral"),
        ("The product is acceptable but the packaging was excessive", "neutral"),
        ("Works as described but the materials feel cheap", "neutral"),
        ("I expected better based on the price but it's acceptable", "neutral"),
        ("The product is good but the shipping took too long", "neutral"),
        ("Functional but the user interface could be more intuitive", "neutral"),
        ("Average performance with no standout features", "neutral"),
        ("It serves its purpose but won't win any awards", "neutral"),
        ("The product is decent but the instructions were unclear", "neutral"),
        ("Not the best I've used but certainly not the worst", "neutral"),
        ("Does what it needs to do but lacks polish", "neutral"),
        ("I'm neither satisfied nor dissatisfied with this purchase", "neutral"),
        ("An adequate solution but there are better options available", "neutral"),
        ("The product is okay but the customer service was lacking", "neutral"),
        ("Works well enough but the build quality could be better", "neutral"),
        ("Gets the job done but I expected more for the price", "neutral"),
        ("A reasonable purchase but not exceptional in any way", "neutral"),
        ("I don't love it but I don't hate it either", "neutral"),
        ("The product is fine but the color was different than pictured", "neutral"),
        ("Average in every way - neither exceeded nor fell short", "neutral"),
        ("It's what you'd expect for the price - nothing more", "neutral"),
        ("Does its job but won't impress anyone", "neutral"),
        ("A practical choice but not particularly exciting", "neutral")
    ] * (SAMPLE_SIZE // 50)
    
    reviews = positive_reviews + negative_reviews + neutral_reviews
    df = pd.DataFrame(reviews, columns=["Review", "Sentiment"])
    
    # Check class distribution
    st.write("Class distribution in training data:")
    st.write(df['Sentiment'].value_counts())
    
    # Preprocess
    df['Processed'] = df['Review'].apply(preprocess_text)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['Processed'], df['Sentiment'], test_size=0.2, random_state=42, stratify=df['Sentiment']
    )
    
    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    # Optimized Random Forest pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,  # Increased from 8000
            ngram_range=(1, 3),
            min_df=3,  # Reduced from 5 to catch more phrases
            max_df=0.8,
            stop_words='english'
        )),
        ('clf', RandomForestClassifier(
            n_estimators=250,  # Increased from 200
            max_depth=35,  # Increased from 30
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Show detailed classification report
    st.write("Detailed Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    # Save model
    model_data = {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'classification_report': report
    }
    joblib.dump(model_data, MODEL_FILE)
    
    st.success(f"Corrected model trained successfully! Accuracy: {accuracy:.1%}")
    return model_data

def load_model():
    try:
        model_data = joblib.load(MODEL_FILE)
        st.success(f"Model loaded successfully! Accuracy: {model_data['accuracy']:.1%}")
        
        # Show performance metrics
        with st.expander("Model Performance Details"):
            st.write("Classification Report:")
            st.dataframe(pd.DataFrame(model_data['classification_report']).transpose())
        
        return model_data
    except:
        return train_model()

def analyze_sentiment(text, model):
    processed = preprocess_text(text)
    prediction = model['pipeline'].predict([processed])[0]
    proba = model['pipeline'].predict_proba([processed])[0]
    confidence = max(proba)
    
    # Additional checks to prevent misclassification
    proba_dict = dict(zip(model['pipeline'].classes_, proba))
    
    # Enhanced strong positive words that should override neutral predictions
    strong_positive = [
        'very nice', 'nice', 'good', 'very good', 'great', 'excellent', 'perfect', 
        'amazing', 'outstanding', 'fantastic', 'awesome', 'superb', 'impressive', 
        'highly recommend', 'delighted', 'thrilled', 'wonderful', 'great quality', 
        'top-notch', 'game changer', 'worth every penny', 'five stars', 'very satisfied',
        'exceeded expectations', 'premium quality', 'best purchase', 'flawless',
        'incredible', 'marvelous', 'satisfied', 'remarkable', 'trusted brand',
        'well made', 'high quality', 'works perfectly', 'durable', 'value for money',
        'pleasantly surprised', 'crystal clear', 'easy to use', 'user friendly',
        'beautiful design', 'modern look', 'smooth operation', 'loved it',
        'recommend to everyone', 'happy with purchase', 'top performance',
        'does the job well', 'met expectations', 'easy installation', 'no complaints',
        'great experience', 'reliable product', 'quick delivery', 'great support',
        'affordable and quality', 'built to last', 'fits perfectly', 'must buy',
        'exceptional', 'brilliant', 'phenomenal', 'stellar', 'first-rate', 'premium',
        'gold standard', 'industry leading', 'cutting edge', 'innovative', 'revolutionary',
        'best in class', 'unparalleled', 'superior', 'masterpiece', 'gem', 'treasure',
        'perfectly', 'exquisitely', 'magnificent', 'splendid', 'extraordinary',
        'breathtaking', 'awe-inspiring', 'second to none', 'peerless', 'matchless',
        'unrivaled', 'best ever', 'life-changing', 'transformative', 'heavenly',
        'divine', 'perfect fit', 'exactly what I wanted', 'better than expected',
        'very pleased', 'no regrets', 'excellent investment', 'worth the wait',
        'highly impressed', 'tick all the boxes', 'checks all the boxes',
        'above and beyond', 'blown away', 'can\'t recommend enough', 'standout product',
        'head and shoulders above', 'in a league of its own', 'sets the standard',
        'benchmark', 'pinnacle', 'apex', 'creme de la creme', 'best of the best',
        'top of the line', 'premier', 'elite', 'supreme', 'ultimate', 'perfect score',
        '10/10', 'five out of five', 'A+', 'gold star', 'blue ribbon', 'editor\'s choice',
        'award-winning', 'best seller', 'customer favorite', 'highly praised',
        'universally acclaimed', 'critically acclaimed', 'rave reviews', 'glowing reviews',
        'universally loved', 'crowd favorite', 'fan favorite', 'beloved', 'cherished',
        'prized possession', 'go-to product', 'daily driver', 'essential', 'indispensable',
        'can\'t live without', 'would buy again in a heartbeat', 'repeat customer',
        'loyal customer', 'brand loyal', 'converted me', 'made me a believer',
        'surpassed all expectations', 'beyond compare', 'nothing compares',
        'worth its weight in gold', 'money well spent', 'pays for itself',
        'return on investment', 'best decision ever', 'smart purchase', 'wise investment',
        'happy camper', 'over the moon', 'on cloud nine', 'i love it', 'i like it',
        'wonderful', 'awesome', 'fantastic'
    ]
    
    # Enhanced strong negative words that should override neutral predictions
    strong_negative = [
        'very bad', 'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'scam',
        'worst', 'broken', 'defective', 'waste', 'avoid', 'poor quality', 
        'does not work', 'very disappointed', 'not worth it', 'unusable', 'garbage', 
        'crap', 'refund', 'fake', 'misleading', 'frustrating', 'low quality', 
        'returning it', 'cheaply made', 'broke after', 'not as described', 'worthless',
        'never again', 'total loss', 'utter failure', 'waste of money',
        'stopped working', 'damaged', 'no response', 'overpriced',
        'annoying', 'malfunctioned', 'useless', 'disappointment',
        'not recommended', 'poor performance', 'late delivery',
        'pathetic', 'not functioning', 'flimsy', 'no value', 'looks cheap', 
        'bad experience', 'hate it', 'problematic', 'too noisy', 'leaking', 
        'ineffective', 'technical issue', 'returned it', 'poor packaging',
        'bad quality', 'not reliable', 'defective item', 'junk', 'trash',
        'ripoff', 'con', 'sham', 'fraudulent', 'deceptive', 'inferior',
        'substandard', 'unacceptable', 'unbearable', 'intolerable', 'dreadful',
        'appalling', 'atrocious', 'abysmal', 'lousy', 'rubbish', 'second-rate',
        'third-rate', 'shoddy', 'tacky', 'cheesy', 'unimpressive', 'mediocre',
        'underwhelming', 'displeasing', 'unsatisfactory', 'unpleasant', 'distasteful',
        'repulsive', 'revolting', 'vile', 'odious', 'contemptible', 'despicable',
        'detestable', 'execrable', 'abominable', 'godawful', 'egregious', 'flagrant',
        'glaring', 'gross', 'rank', 'rotten', 'putrid', 'foul', 'nasty', 'hideous',
        'monstrous', 'ghastly', 'gruesome', 'grating', 'irksome', 'vexing', 'maddening',
        'infuriating', 'exasperating', 'aggravating', 'trying', 'tiresome', 'tedious',
        'boring', 'dull', 'uninspiring', 'unexciting', 'uninteresting', 'lackluster',
        'forgettable', 'disheartening', 'discouraging', 'demoralizing', 'depressing',
        'dispiriting', 'daunting', 'disenchanting', 'disillusioning', 'disappointing',
        'unsatisfying', 'unfulfilling', 'letdown', 'anticlimactic', 'underwhelming',
        'unrewarding', 'thankless', 'unappreciated', 'unvalued', 'unwanted', 'rejected',
        'abandoned', 'neglected', 'ignored', 'overlooked', 'forgotten', 'disregarded',
        'dismissed', 'snubbed', 'slighted', 'insulted', 'offended', 'affronted',
        'humiliated', 'embarrassed', 'mortified', 'chagrined', 'discomfited', 'abashed',
        'sheepish', 'hangdog', 'shamefaced', 'contrite', 'remorseful', 'penitent',
        'repentant', 'rueful', 'apologetic', 'regretful', 'sorry', 'conscience-stricken',
        'guilt-ridden', 'compunctious', 'self-reproachful', 'self-condemning'
    ]
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count strong words
    pos_count = sum(1 for phrase in strong_positive if phrase in text_lower)
    neg_count = sum(1 for phrase in strong_negative if phrase in text_lower)
    
    # Override rules with more aggressive thresholds
    if pos_count > 0:
        if proba_dict.get('positive', 0) > 0.3:  # Lowered threshold from 0.4
            prediction = 'positive'
            confidence = max(confidence, proba_dict.get('positive', 0))
    elif neg_count > 0:
        if proba_dict.get('negative', 0) > 0.3:  # Lowered threshold from 0.4
            prediction = 'negative'
            confidence = max(confidence, proba_dict.get('negative', 0))
    
    # Special case for short positive/negative reviews
    words = text_lower.split()
    if len(words) <= 4:
        if any(word in ['nice', 'good', 'great', 'awesome', 'fine', 'ok'] for word in words):
            if not any(neg in words for neg in ['not', 'no', 'never']):
                prediction = 'positive'
                confidence = max(confidence, 0.85)  # Higher confidence for clear cases
        elif any(word in ['bad', 'poor', 'worst', 'terrible'] for word in words):
            prediction = 'negative'
            confidence = max(confidence, 0.85)
    
    return prediction, confidence

def detect_aspects(text):
    aspects = []
    text = text.lower()
    
    aspect_keywords = {
        'price': [
            'price', 'cost', 'value', 'expensive', 'affordable', 'cheap', 'overpriced',
            'bargain', 'deal', 'worth', 'reasonable', 'inexpensive', 'pricing', 'premium',
            'discount', 'budget', 'economical', 'money', 'bang for buck', 'marked up',
            'steal', 'pricy', 'underpriced', 'rs.', '₹', 'rupees', 'dollar', 'costly',
            'value for money', 'pocket-friendly', 'rate', 'valuation', 'worth the price',
            'pricey', 'low-cost', 'high-end', 'luxury', 'budget', 'investment'
        ],
        'quality': [
            'quality', 'durable', 'material', 'build', 'sturdy', 'flimsy', 'durability',
            'construction', 'well-made', 'solid', 'craftsmanship', 'finish', 'long-lasting',
            'poorly made', 'fragile', 'robust', 'reliable', 'wear and tear', 'cheaply made',
            'superior quality', 'premium feel', 'strong', 'weak', 'tough', 'breakable',
            'resistant', 'fragile', 'sturdiness', 'build quality', 'material quality',
            'finishing', 'polish', 'texture', 'feel', 'look', 'appearance', 'design',
            'plastic', 'metal', 'fabric', 'component', 'part', 'assembly', 'structure'
        ],
        'delivery': [
            'delivery', 'shipping', 'arrived', 'time', 'late', 'fast', 'slow', 'package', 'packaging',
            'on time', 'delay', 'courier', 'tracking', 'schedule', 'box', 'damage during delivery',
            'next day', 'prime delivery', 'timely', 'intact', 'unsealed'
        ],
        'service': [
            'service', 'customer', 'support', 'help', 'return', 'warranty', 'refund',
            'complaint', 'assistance', 'response', 'staff', 'representative', 'exchange',
            'helpline', 'policy', 'communication', 'rude', 'friendly', 'prompt',
            'unresponsive', 'satisfaction', 'care', 'delivery', 'shipping', 'packaging',
            'install', 'setup', 'assembly', 'manual', 'guide', 'instruction', 'contact',
            'email', 'phone', 'chat', 'resolution', 'issue', 'problem', 'fix', 'repair',
            'replacement', 'customer care', 'after sales', 'technical support'
        ],
        'performance': [
            'performance', 'speed', 'power', 'efficient', 'function', 'feature', 'works',
            'operation', 'lag', 'smooth', 'responsive', 'cooling', 'air flow', 'breeze',
            'output', 'crash', 'glitch', 'load', 'multitask', 'capacity', 'stability',
            'freeze', 'fluid', 'execute', 'hardware', 'software', 'noise', 'sound',
            'quiet', 'loud', 'volume', 'decibel', 'rpm', 'speed', 'power consumption',
            'energy efficient', 'cooling capacity', 'airflow', 'ventilation', 'fan speed',
            'temperature', 'cooling effect', 'chill', 'ice', 'water', 'tank', 'capacity',
            'cooling power', 'cooling performance', 'cooling efficiency'
        ]
    }
    
    # Check for multi-word phrases first
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if ' ' in keyword and keyword in text:
                aspects.append(aspect)
                break  # No need to check other keywords for this aspect
    
    # Then check for single words if no aspects found yet
    if not aspects:
        words = set(text.split())
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in words for keyword in keywords if ' ' not in keyword):
                aspects.append(aspect)
    
    return aspects if aspects else ['general']

def detect_emotion(text):
    text = text.lower()
    
    # Enhanced emotion detection with more nuanced word lists
    happy_words = [
        'love', 'great', 'excellent', 'perfect', 'happy', 'amazing', 'wonderful', 'fantastic', 
        'awesome', 'satisfied', 'delighted', 'pleased', 'thrilled', 'enjoyed', 'fabulous', 
        'superb', 'incredible', 'brilliant', 'cheerful', 'blissful', 'ecstatic', 'glad', 
        'content', 'joyful', 'grateful', 'impressed', 'smiling', 'sunny', 'positive vibes',
        'happy with', 'love it', 'highly recommend', 'exceeded expectations', 'perfectly',
        'exactly what i wanted', 'better than expected', 'very pleased', 'no complaints',
        'works great', 'perfect fit', 'excellent quality', 'very happy', 'great value',
        'would buy again', 'excellent product', 'excellent service', 'fast delivery',
        'good quality', 'nice product', 'well made', 'good value', 'happy customer',
        'satisfied customer', 'great purchase', 'good deal', 'nice price', 'good service',
        'fast shipping', 'good experience', 'nice experience', 'happy with purchase',
        'love this product', 'love this item', 'love this', 'very good', 'very nice',
        'very happy with', 'very satisfied with', 'very pleased with', 'very impressed with',
        'very good quality', 'very nice product', 'very good product', 'very nice item',
        'very good item', 'very happy customer', 'very satisfied customer'
    ]
    
    angry_words = [
        'terrible', 'awful', 'horrible', 'angry', 'disgusted', 'dissatisfied', 'unhappy', 
        'sad', 'irritated', 'let down', 'discouraged', 'displeased', 'resentful', 'unimpressed',
        'offended', 'depressed', 'grumpy', 'heartbroken', 'disappointed', 'frustrated', 
        'upset', 'annoyed', 'disgusted', 'hate', 'despise', 'loathe', 'regret', 'scam', 
        'fraud', 'lied', 'deceived', 'unacceptable', 'untrustworthy', 'disgusting', 
        'worthless', 'fake', 'manipulated', 'tricked', 'betrayed', 'scammed', 'con', 
        'terrified', 'furious', 'raging', 'vengeful', 'never again', 'waste of money',
        'poor quality', 'not as described', 'not working', 'not worth it', 'not happy',
        'not satisfied', 'not pleased', 'not impressed', 'not good', 'not nice',
        'not happy with', 'not satisfied with', 'not pleased with', 'not impressed with',
        'not good quality', 'not nice product', 'not good product', 'not nice item',
        'not good item', 'not happy customer', 'not satisfied customer'
    ]
    
    # Count matches for each emotion
    happy_count = sum(1 for word in happy_words if word in text)
    angry_count = sum(1 for word in angry_words if word in text)
    
    # Enhanced emotion determination with intensity scoring
    if angry_count >= 2 or any(word in text for word in ['hate', 'scam', 'fraud', 'terrible', 'awful']):
        return 'angry'
    elif happy_count >= 2 or any(word in text for word in ['love', 'excellent', 'perfect', 'amazing']):
        return 'happy'
    else:
        return 'neutral'

def main():
    st.set_page_config(
        page_title="Amazon Reviews Analyzer",
        page_icon="📊",
        layout="centered"
    )
    
    st.title("SentiScan")
    st.markdown("Analyze product reviews with accurate positive/negative classification")
    
    model = load_model()
    
    tab1, tab2 = st.tabs(["Single Review", "Batch Analysis"])
    
    with tab1:
        review = st.text_area("Enter review:", "This product is absolutely amazing, I love everything about it!")
        
        if st.button("Analyze"):
            if review.strip():
                sentiment, confidence = analyze_sentiment(review, model)
                aspects = detect_aspects(review)
                emotion = detect_emotion(review)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if sentiment == "positive":
                        st.success(f"Sentiment: {sentiment.capitalize()} 😊")
                    elif sentiment == "negative":
                        st.error(f"Sentiment: {sentiment.capitalize()} 😞")
                    else:
                        st.info(f"Sentiment: {sentiment.capitalize()} 😐")
                    
                    st.write(f"Confidence: {confidence:.1%}")
                    st.progress(confidence)
                
                with col2:
                    st.subheader("Analysis")
                    st.write(f"**Emotion:** {emotion.capitalize()}")
                    st.write("**Main Aspects:**")
                    for aspect in aspects:
                        st.write(f"- {aspect.capitalize()}")
                
                # Show explanation if positive
                if sentiment == "positive":
                    st.success("""
                    **Positive Review Detected:**  
                    Our model has identified strong positive sentiment in this review. 
                    Key indicators include:  
                    - Positive words and phrases  
                    - Expressions of satisfaction  
                    - Praise for product quality  
                    """)
                elif sentiment == "negative":
                    st.error("""
                    **Negative Review Detected:**  
                    Our model has identified negative sentiment in this review.
                    Key indicators include:
                    - Negative words and phrases
                    - Expressions of dissatisfaction
                    - Complaints about product quality
                    """)
                else:
                    st.info("""
                    **Neutral Review Detected:**  
                    The review appears balanced or factual without strong positive or negative language.
                    """)
            else:
                st.warning("Please enter a review to analyze")
    
    with tab2:
        from batch_analyzer import batch_analysis_tab
        batch_analysis_tab(model)

if __name__ == "__main__":
    main()
