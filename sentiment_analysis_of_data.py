import pandas as pd
import numpy as np
import sys 
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

class LSTMStressSentimentAnalyzer:
    def __init__(self, max_words=100000, max_len=150, embedding_dim=128, platform_name=""):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.platform_name = platform_name
        self.min_samples_per_class = 10  # Increased minimum samples
        
    def create_sentiment_labels(self, df):
        """Create more nuanced sentiment labels using TextBlob."""
        print(f"Creating sentiment labels for {self.platform_name} dataset...")
        
        def analyze_sentiment(text):
            analysis = TextBlob(str(text))
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            
            # More nuanced classification using both polarity and subjectivity
            if polarity > 0.1:
                return 0  # Positive
            elif polarity < -0.1:
                return 2  # Negative
            else:
                # Use subjectivity to help determine neutral vs negative cases
                if subjectivity < 0.5 and polarity <= 0:
                    return 2  # Likely negative
                return 1  # Neutral
        
        df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)
        
        # Check class distribution
        class_counts = Counter(df['sentiment'])
        print(f"\nInitial class distribution for {self.platform_name}:")
        for label, count in class_counts.items():
            print(f"Class {label}: {count} samples")
            
        return df['sentiment']

    def balance_dataset(self, df, sentiment_labels):
        """Balance the dataset using undersampling for majority classes and augmentation for minority."""
        print(f"\nBalancing dataset for {self.platform_name}...")
        
        df['sentiment'] = sentiment_labels
        class_counts = Counter(sentiment_labels)
        
        # Find median class size as target
        target_samples = int(np.median([count for count in class_counts.values()]))
        target_samples = max(target_samples, self.min_samples_per_class)
        
        balanced_dfs = []
        for sentiment_class in [0, 1, 2]:  # Ensure we handle all three classes
            class_df = df[df['sentiment'] == sentiment_class]
            if len(class_df) == 0:
                continue
                
            if len(class_df) > target_samples:
                # Undersample majority class
                class_df = class_df.sample(target_samples, random_state=42)
            elif len(class_df) < target_samples:
                # Simple augmentation for minority class
                multiplier = int(np.ceil(target_samples / len(class_df)))
                augmented_df = pd.concat([class_df] * multiplier)
                class_df = augmented_df.sample(target_samples, replace=False, random_state=42)
            
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Verify final distribution
        final_counts = Counter(balanced_df['sentiment'])
        print("\nFinal class distribution:")
        for label, count in final_counts.items():
            print(f"Class {label}: {count} samples")
        
        return balanced_df

    def load_and_prepare_data(self, cleaned_files):
        """Load and combine platform-specific datasets with improved error handling."""
        print(f"Loading and preparing data for {self.platform_name}...")
        
        if not cleaned_files:
            raise ValueError(f"No files provided for {self.platform_name}")

        dfs = []
        for file in cleaned_files:
            try:
                df = pd.read_csv(file)
                if 'cleaned_text' not in df.columns:
                    raise ValueError(f"Missing 'cleaned_text' column in {file}")
                # Remove empty or very short texts
                df = df[df['cleaned_text'].str.len() > 10]
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue

        if not dfs:
            raise ValueError(f"No valid data files loaded for {self.platform_name}")

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.dropna(subset=['cleaned_text'])
        
        # Remove duplicates and very similar texts
        combined_df = combined_df.drop_duplicates(subset=['cleaned_text'])
        
        print(f"Total samples after preprocessing: {len(combined_df)}")
        return combined_df

    def preprocess_texts(self, texts):
        """Tokenize and pad sequences with improved text preprocessing."""
        # Fit tokenizer on training texts
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        return padded_sequences

    def build_model(self):
        """Build an improved LSTM model with additional layers and regularization."""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.max_words, 
                                    output_dim=self.embedding_dim, 
                                    input_length=self.max_len),
            tf.keras.layers.SpatialDropout1D(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     metrics=['accuracy'])
        return model

    def analyze_sentiment(self, cleaned_files, test_size=0.2):
        """Run the improved sentiment analysis pipeline"""
        try:
            # Load and prepare data
            combined_df = self.load_and_prepare_data(cleaned_files)
            
            # Create initial sentiment labels with improved method
            sentiment_labels = self.create_sentiment_labels(combined_df)
            
            # Balance dataset with improved method
            balanced_df = self.balance_dataset(combined_df, sentiment_labels)
            
            # Prepare features and labels
            X = self.preprocess_texts(balanced_df['cleaned_text'])
            y = self.label_encoder.fit_transform(balanced_df['sentiment'])
            y = tf.keras.utils.to_categorical(y)

            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Build and train model
            self.model = self.build_model()
            
            # Improved callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            )
            
            print(f"\nTraining model for {self.platform_name}...")
            history = self.model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr],
                class_weight={0: 1.0, 1: 1.0, 2: 1.2}
            )

            # Evaluate model
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
            print(f"\n{self.platform_name} Test accuracy: {test_accuracy:.4f}")
            
            # Print detailed classification report
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            print("\nClassification Report:")
            print(classification_report(y_test_classes, y_pred_classes, 
                                     target_names=['Positive', 'Neutral', 'Negative']))

            return test_accuracy, balanced_df

        except Exception as e:
            print(f"Error in sentiment analysis for {self.platform_name}: {str(e)}")
            raise

    def predict_sentiment(self, texts):
        """Predict sentiment for new texts with confidence scores."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        processed_texts = self.preprocess_texts(texts)
        predictions = self.model.predict(processed_texts)
        predicted_labels = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Add prediction probabilities for each class
        class_probabilities = {
            'positive_prob': predictions[:, 0],
            'neutral_prob': predictions[:, 1],
            'negative_prob': predictions[:, 2]
        }
        
        return predicted_labels, confidence_scores, class_probabilities

def analyze_platform_data(platform_files, platform_name):
    """Analyze data for a specific platform with improved error handling and logging."""
    analyzer = LSTMStressSentimentAnalyzer(platform_name=platform_name)
    
    try:
        accuracy, combined_df = analyzer.analyze_sentiment(platform_files)
        all_texts = combined_df['cleaned_text'].tolist()
        
        predictions, confidence_scores, class_probs = analyzer.predict_sentiment(all_texts)
        
        # Create results DataFrame with all probabilities
        results_df = pd.DataFrame({
            "Text": all_texts,
            "Predicted_Sentiment": predictions,
            "Confidence_Score": confidence_scores,
            "Positive_Probability": class_probs['positive_prob'],
            "Neutral_Probability": class_probs['neutral_prob'],
            "Negative_Probability": class_probs['negative_prob']
        })
        
        # Export all predictions
        output_file = f"sentiment_predictions_lstm_{platform_name.lower()}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nAll {platform_name} predictions saved to {output_file}")
        
        # Export predictions for each class
        for sentiment, label in [(2, 'negative')]:
            class_df = results_df[results_df["Predicted_Sentiment"] == sentiment]
            class_output_file = f"{label}_sentiment_predictions_lstm_{platform_name.lower()}.csv"
            class_df.to_csv(class_output_file, index=False)
            print(f"\n{label.capitalize()} {platform_name} sentiment predictions saved to {class_output_file}")
        
        return accuracy
        
    except Exception as e:
        print(f"\nError in {platform_name} analysis:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return None

def main():
    """Main function to run separate Reddit and Twitter sentiment analysis"""
    print("Starting Separate Platform Sentiment Analysis using LSTM Model...")

    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sentiment_analysis.log'),
            logging.StreamHandler()
        ]
    )

    # Separate files by platform
    twitter_files = [
        'cleaned_mental_health_twitter.csv',
        'cleaned_twitter_data_2.csv',
        'cleaned_sentiment_140.csv'
    ]
    
    reddit_files = [
        'cleaned_reddit_title.csv',
        'cleaned_reddit_combi.csv'
    ]

    results = {}
    
    try:
        # Analyze Twitter data
        logging.info("=== Processing Twitter Data ===")
        twitter_accuracy = analyze_platform_data(twitter_files, "Twitter")
        if twitter_accuracy is not None:
            results['Twitter'] = twitter_accuracy
        
        # Analyze Reddit data
        logging.info("=== Processing Reddit Data ===")
        reddit_accuracy = analyze_platform_data(reddit_files, "Reddit")
        if reddit_accuracy is not None:
            results['Reddit'] = reddit_accuracy
        
        # Print final results
        logging.info("=== Final Results ===")
        for platform, accuracy in results.items():
            logging.info(f"{platform} Model Accuracy: {accuracy:.4f}")

        # Combine all class predictions from both platforms
        for sentiment_type in ['negative']:
            twitter_file = f"{sentiment_type}_sentiment_predictions_lstm_twitter.csv"
            reddit_file = f"{sentiment_type}_sentiment_predictions_lstm_reddit.csv"
            
            try:
                twitter_df = pd.read_csv(twitter_file)
                reddit_df = pd.read_csv(reddit_file)
                
                twitter_df['Platform'] = 'Twitter'
                reddit_df['Platform'] = 'Reddit'
                
                combined_df = pd.concat([twitter_df, reddit_df], ignore_index=True)
                combined_output = f"combined_{sentiment_type}_sentiment_predictions_lstm.csv"
                combined_df.to_csv(combined_output, index=False)
                logging.info(f"Combined {sentiment_type} predictions saved to {combined_output}")
            except Exception as e:
                logging.error(f"Error combining {sentiment_type} predictions: {str(e)}")
        
        if not results:
            logging.error("No successful analyses completed.")
            sys.exit(1)

    except Exception as e:
        logging.error("Error occurred during execution:")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error message: {str(e)}")
        logging.error("Traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    print("Script started")
    main()
    print("Script completed")