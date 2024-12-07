import pandas as pd
import traceback
import numpy as np
import re
import os
from typing import Optional, Dict, List

class AdolescentStressDataCleaner:
    def __init__(self):
        # Define term dictionaries
        self.stress_terms = [
            'stress', 'anxiety', 'depression', 'worried', 'nervous',
            'panic', 'tension', 'pressure', 'overwhelm', 'burnout',
            'mental health', 'emotional', 'exhausted', 'tired'
        ]
        
        self.educational_terms = [
            'school', 'college', 'university', 'exam', 'test',
            'homework', 'study', 'class', 'grade', 'assignment',
            'teacher', 'professor', 'student', 'education'
        ]
        
        self.age_terms = [
            'teen', 'teenage', 'adolescent'
        ]
        
        self.slang_terms = [
           'lit', 'fam', 'salty', 'slay', 'tea', 
           'sus', 'ghosting', 'flex', 'shade', 'vibe',
           'bet', 'lowkey', 'highkey', 'on fleek', 'woke',
           'savage', 'bae', 'dope', 'snatched', 'no cap'
       ]


    def process_mental_health_twitter(self, file_path: str) -> Optional[pd.DataFrame]:
        """Process Mental Health Twitter dataset with enhanced error handling and logging"""
        print(f"\nProcessing Mental Health Twitter dataset: {file_path}")
        
        try:
            # First try to peek at the CSV structure
            print(f"Attempting to read first few lines of {file_path}")
            df_peek = pd.read_csv(file_path, nrows=5)
            print(f"Available columns: {df_peek.columns.tolist()}")
            
            # Read CSV with more flexible column handling
            df = pd.read_csv(file_path, 
                           encoding='utf-8',
                           na_values=[''],
                           keep_default_na=True)
            
            # Check if 'text' column exists, if not look for similar columns
            if 'text' not in df.columns:
                potential_text_columns = [col for col in df.columns if 'text' in col.lower()]
                if potential_text_columns:
                    text_col = potential_text_columns[0]
                    print(f"'text' column not found. Using '{text_col}' instead.")
                    df = df.rename(columns={text_col: 'text'})
                else:
                    print("Available columns:", df.columns.tolist())
                    raise ValueError("Could not find 'text' column or similar in the dataset")
            
            print(f"Successfully loaded dataset with {len(df)} rows")
            return self.process_dataset(df, 'text', 'cleaned_mental_health_twitter.csv')
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Please check the file path.")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: File '{file_path}' is empty.")
            return None
        except Exception as e:
            print(f"Error loading Mental Health Twitter dataset: {str(e)}")
            print("Full error details:", traceback.format_exc())
            return None

    def process_twitter_data_2(self, file_path: str) -> Optional[pd.DataFrame]:
        """Process twitter_data_2 dataset with enhanced error handling and logging"""
        print(f"\nProcessing twitter_data_2 dataset: {file_path}")
        
        try:
            # Read CSV with specific column names and handle empty fields
            df = pd.read_csv(file_path, 
                           sep=';',
                           encoding='utf-8',
                           na_values=[''],
                           keep_default_na=True)
            
            print(f"Successfully loaded dataset with {len(df)} rows")
            return self.process_dataset(df, 'text', 'cleaned_twitter_data_2.csv')
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Please check the file path.")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: File '{file_path}' is empty.")
            return None
        except Exception as e:
            print(f"Error loading twitter_data_2 dataset: {str(e)}")
            print("Full error details:", traceback.format_exc())
            return None

    def process_sentiment_140(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Process Sentiment140 dataset with enhanced error handling and logging.
        The dataset contains columns:
        - sentiment (0 = negative, 4 = positive)
        - id 
        - timestamp
        - query
        - username
        - text
        
        Args:
            file_path (str): Path to the Sentiment140 CSV file
            
        Returns:
            Optional[pd.DataFrame]: Processed DataFrame or None if processing fails
        """
        print(f"\nProcessing Sentiment140 dataset: {file_path}")
        
        # List of encodings to try
        encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8']
        
        for encoding in encodings:
            try:
                print(f"Attempting to read file with {encoding} encoding...")
                
                # First try to peek at the CSV structure
                df_peek = pd.read_csv(file_path, nrows=5, encoding=encoding,
                                    names=['sentiment', 'id', 'timestamp', 'query', 'username', 'text'])
                print(f"Successfully read preview with {encoding} encoding")
                print(f"Available columns: {df_peek.columns.tolist()}")
                
                # Read the full CSV with proper column names
                df = pd.read_csv(file_path,
                               encoding=encoding,
                               names=['sentiment', 'id', 'timestamp', 'query', 'username', 'text'],
                               na_values=[''],
                               keep_default_na=True,
                               on_bad_lines='warn')  # More permissive handling of problematic lines
                
                print(f"Successfully loaded full dataset with {encoding} encoding")
                
                # Basic data validation
                required_columns = ['sentiment', 'text']
                if not all(col in df.columns for col in required_columns):
                    raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
                
                # Convert sentiment labels (0 = negative, 4 = positive) to binary format (0 = negative, 1 = positive)
                df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
                print("Converted sentiment labels to binary format (0 = negative, 1 = positive)")
                
                # Handle timestamp conversion with error catching
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    print("Converted timestamp column to datetime format")
                except Exception as e:
                    print(f"Warning: Could not convert timestamps: {str(e)}")
                    print("Keeping original timestamp format")
                
                # Remove any rows with missing text
                initial_rows = len(df)
                df = df.dropna(subset=['text'])
                rows_dropped = initial_rows - len(df)
                if rows_dropped > 0:
                    print(f"Dropped {rows_dropped} rows with missing text")
                
                # Remove 'NO_QUERY' values from query column
                df['query'] = df['query'].replace('NO_QUERY', '')
                print("Cleaned 'NO_QUERY' values from query column")
                
                # Basic statistics
                print(f"\nDataset Statistics:")
                print(f"Total rows: {len(df)}")
                print(f"Positive sentiment tweets: {len(df[df['sentiment'] == 1])}")
                print(f"Negative sentiment tweets: {len(df[df['sentiment'] == 0])}")
                
                return self.process_dataset(df, 'text', 'cleaned_sentiment_140.csv')
                
            except UnicodeDecodeError:
                print(f"Failed to read with {encoding} encoding, trying next encoding...")
                continue
            except FileNotFoundError:
                print(f"Error: File '{file_path}' not found. Please check the file path.")
                return None
            except pd.errors.EmptyDataError:
                print(f"Error: File '{file_path}' is empty.")
                return None
            except Exception as e:
                print(f"Error processing file with {encoding} encoding: {str(e)}")
                print("Full error details:", traceback.format_exc())
                continue
    
        print("Error: Failed to read file with any of the attempted encodings")
        return None
        
    def process_reddit_title(self, file_path: str) -> Optional[pd.DataFrame]:
        """Process Reddit Title dataset with enhanced error handling and logging"""
        print(f"\nProcessing Reddit Title dataset: {file_path}")
        
        try:
            # Read CSV with specific handling for semicolon delimiter
            df = pd.read_csv(file_path, 
                           sep=';',
                           encoding='utf-8',
                           na_values=[''],
                           keep_default_na=True)
            
            # Verify required columns exist
            if 'title' not in df.columns:
                potential_title_cols = [col for col in df.columns if 'title' in col.lower()]
                if potential_title_cols:
                    title_col = potential_title_cols[0]
                    print(f"'title' column not found. Using '{title_col}' instead.")
                    df = df.rename(columns={title_col: 'title'})
                else:
                    raise ValueError("Could not find 'title' column in the dataset")
                    
            if 'label' not in df.columns:
                raise ValueError("Could not find 'label' column in the dataset")
            
            # Convert label to numeric if not already
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            
            print(f"Successfully loaded dataset with {len(df)} rows")
            print(f"Distribution of labels:\n{df['label'].value_counts()}")
            
            return self.process_dataset(df, 'title', 'cleaned_reddit_title.csv')
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Please check the file path.")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: File '{file_path}' is empty.")
            return None
        except Exception as e:
            print(f"Error loading Reddit Title dataset: {str(e)}")
            print("Full error details:", traceback.format_exc())
            return None

    def process_reddit_combi(self, file_path: str) -> Optional[pd.DataFrame]:
        """Process Reddit Combi dataset with enhanced error handling and logging"""
        print(f"\nProcessing Reddit Combi dataset: {file_path}")
        
        try:
            # Read CSV with specific handling for semicolon delimiter
            df = pd.read_csv(file_path, 
                           sep=';',
                           encoding='utf-8',
                           na_values=[''],
                           keep_default_na=True)
            
            # Verify required columns exist
            if 'Body_Title' not in df.columns:
                potential_title_cols = [col for col in df.columns 
                                      if any(term in col.lower() 
                                            for term in ['body_title', 'bodytitle', 'body'])]
                if potential_title_cols:
                    title_col = potential_title_cols[0]
                    print(f"'Body_Title' column not found. Using '{title_col}' instead.")
                    df = df.rename(columns={title_col: 'Body_Title'})
                else:
                    print("Available columns:", df.columns.tolist())
                    raise ValueError("Could not find 'Body_Title' or similar column in the dataset")
                    
            if 'label' not in df.columns:
                potential_label_cols = [col for col in df.columns 
                                      if any(term in col.lower() 
                                            for term in ['label', 'class', 'category'])]
                if potential_label_cols:
                    label_col = potential_label_cols[0]
                    print(f"'label' column not found. Using '{label_col}' instead.")
                    df = df.rename(columns={label_col: 'label'})
                else:
                    raise ValueError("Could not find 'label' column in the dataset")
            
            # Convert label to numeric if not already
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            
            # Drop rows with NaN labels
            initial_rows = len(df)
            df = df.dropna(subset=['label'])
            dropped_rows = initial_rows - len(df)
            
            if dropped_rows > 0:
                print(f"Dropped {dropped_rows} rows with invalid labels")
            
            print(f"Successfully loaded dataset with {len(df)} rows")
            print(f"Distribution of labels:\n{df['label'].value_counts().sort_index()}")
            
            # Process the dataset using the common processing method
            return self.process_dataset(df, 'Body_Title', 'cleaned_reddit_combi.csv')
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Please check the file path.")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: File '{file_path}' is empty.")
            return None
        except Exception as e:
            print(f"Error loading Reddit Combi dataset: {str(e)}")
            print("Full error details:", traceback.format_exc())
            return None

    def process_dataset(self, df: pd.DataFrame, text_column: str, output_file: str) -> Optional[pd.DataFrame]:
        """Process any dataset with the specified text column"""
        try:
            print(f"Initial post count: {len(df)}")
            
            # Remove duplicates
            df = df.drop_duplicates(subset=[text_column])
            print(f"Posts after deduplication: {len(df)}")
            
            # Clean text
            print("Cleaning text...")
            df['cleaned_text'] = df[text_column].apply(self.clean_text)
            
            # Calculate relevance scores and get matching terms
            print("Calculating relevance scores and identifying keywords...")
            df['relevance_score'] = df['cleaned_text'].apply(self.calculate_relevance_score)
            
            # Add columns for matching terms from each category
            df['stress_terms_found'] = df['cleaned_text'].apply(
                lambda x: ', '.join(self.find_matching_terms(x, self.stress_terms)) or 'none')
            df['educational_terms_found'] = df['cleaned_text'].apply(
                lambda x: ', '.join(self.find_matching_terms(x, self.educational_terms)) or 'none')
            df['age_terms_found'] = df['cleaned_text'].apply(
                lambda x: ', '.join(self.find_matching_terms(x, self.age_terms)) or 'none')
            df['slang_terms_found'] = df['cleaned_text'].apply(
                lambda x: ', '.join(self.find_matching_terms(x, self.slang_terms)) or 'none')
            
            # Filter based on relevance score
            threshold = 0.5
            
            # Create a boolean mask for filtering
            mask = df['relevance_score'] >= threshold
            
            # Create a new DataFrame instead of a view
            relevant_df = df[mask].copy()
            
            # Add term count columns to the new DataFrame
            relevant_df['stress_term_count'] = relevant_df['stress_terms_found'].apply(
                lambda x: 0 if x == 'none' else len(x.split(', ')))
            relevant_df['educational_term_count'] = relevant_df['educational_terms_found'].apply(
                lambda x: 0 if x == 'none' else len(x.split(', ')))
            relevant_df['age_term_count'] = relevant_df['age_terms_found'].apply(
                lambda x: 0 if x == 'none' else len(x.split(', ')))
            relevant_df['slang_term_count'] = relevant_df['slang_terms_found'].apply(
                lambda x: 0 if x == 'none' else len(x.split(', ')))
                
            print(f"Posts with relevance score >= {threshold}: {len(relevant_df)}")
            
            # Rest of the method remains the same...
            
            # Show score distribution before filtering
            print("Score distribution before filtering:")
            print(df['relevance_score'].describe())
            
            # Sample and display some retained posts
            print("\nSample of retained posts:")
            sample_posts = relevant_df.head(3)
            for _, row in sample_posts.iterrows():
                print(f"\nOriginal text: {row[text_column][:100]}...")
                print(f"Cleaned text: {row['cleaned_text'][:100]}...")
                print(f"Relevance score: {row['relevance_score']:.2f}")
                print(f"Stress terms: {row['stress_terms_found']}")
                print(f"Educational terms: {row['educational_terms_found']}")
                print(f"Age-related terms: {row['age_terms_found']}")
                print(f"Slang-related terms: {row['slang_terms_found']}")
            
            # Save to CSV
            relevant_df.to_csv(output_file, index=False)
            print(f"\nCleaned dataset saved to {output_file}")
            
            # Print summary statistics for matching terms
            print("\nKeyword matching summary:")
            print(f"Average stress terms per post: {relevant_df['stress_term_count'].mean():.2f}")
            print(f"Average educational terms per post: {relevant_df['educational_term_count'].mean():.2f}")
            print(f"Average age-related terms per post: {relevant_df['age_term_count'].mean():.2f}")
            print(f"Average slang-related terms per post: {relevant_df['slang_term_count'].mean():.2f}")
            
            return relevant_df
            
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            print("Full error details:", traceback.format_exc())
            return None

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and numbers but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def find_matching_terms(self, text: str, term_list: List[str]) -> List[str]:
        """Find all matching terms from the term list in the text"""
        return [term for term in term_list if term in text]

    def calculate_relevance_score(self, text: str) -> float:
        """Calculate relevance score based on presence of key terms"""
        if not text:
            return 0.0
            
        stress_matches = len(self.find_matching_terms(text, self.stress_terms))
        edu_matches = len(self.find_matching_terms(text, self.educational_terms))
        age_matches = len(self.find_matching_terms(text, self.age_terms))
        slang_matches = len(self.find_matching_terms(text, self.slang_terms))
        
        # Calculate score components
        stress_score = min(stress_matches / 2, 1.0)  # Cap at 1.0
        edu_score = min(edu_matches / 2, 1.0)
        age_score = min(age_matches / 2, 1.0)
        slang_score = min(age_matches / 2, 1.0)
        
        # Weighted average (stress terms weighted more heavily)
        total_score = (stress_score * 0.30 + edu_score * 0.20 + age_score * 0.30 + slang_score*0.20)
        return round(total_score, 3)

def main():
    cleaner = AdolescentStressDataCleaner()
    
    try:
        # Process Mental Health Twitter dataset
        print("\nProcessing Mental Health Twitter dataset...")
        result_df1 = cleaner.process_mental_health_twitter('Mental-Health-Twitter.csv')
        if result_df1 is not None and len(result_df1) > 0:
            print("\nMental Health Twitter processing completed successfully!")
            print(f"Total posts retained: {len(result_df1)}")
            print("\nScore distribution for retained posts:")
            print(result_df1['relevance_score'].describe())
        else:
            print("\nWarning: Processing failed for Mental Health Twitter dataset.")
            
        # Process twitter_data_2 dataset
        print("\nProcessing twitter_data_2 dataset...")
        result_df2 = cleaner.process_twitter_data_2('twitter_data_2.csv')
        if result_df2 is not None and len(result_df2) > 0:
            print("\nTwitter data 2 processing completed successfully!")
            print(f"Total posts retained: {len(result_df2)}")
            print("\nScore distribution for retained posts:")
            print(result_df2['relevance_score'].describe())
        else:
            print("\nWarning: Processing failed for twitter_data_2 dataset.")
            
        # Process Reddit_Title dataset
        result_df3 = cleaner.process_reddit_title('Reddit_Title.csv')
        if result_df3 is not None and len(result_df3) > 0:
            print("\nReddit Title processing completed successfully!")
            print(f"Total posts retained: {len(result_df3)}")
            print("\nScore distribution for retained posts:")
            print(result_df3['relevance_score'].describe())
        else:
            print("\nWarning: Processing failed for Reddit Title dataset.")
            

        # Process Reddit_Combi dataset
        result_df4 = cleaner.process_reddit_combi('Reddit_Combi.csv')
        if result_df4 is not None and len(result_df4) > 0:
            print("\nReddit Combi processing completed successfully!")
            print(f"Total posts retained: {len(result_df4)}")
            print("\nScore distribution for retained posts:")
            print(result_df4['relevance_score'].describe())
        else:
            print("\nWarning: Processing failed for Reddit Combi dataset.")
            
        # Process Sentiment_140 dataset
        result_df5 = cleaner.process_sentiment_140('sentiment_140.csv')
        if result_df5 is not None and len(result_df5) > 0:
            print("\Sentiment 140 processing completed successfully!")
            print(f"Total posts retained: {len(result_df5)}")
            print("\nScore distribution for retained posts:")
            print(result_df5['relevance_score'].describe())
        else:
            print("\nWarning: Processing failed for Sentiment 140 dataset.")
            

            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()