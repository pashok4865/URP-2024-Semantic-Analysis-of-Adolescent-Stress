import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import os
import re

# Download stopwords from nltk
nltk.download('stopwords')

class StressTopicAnalyzer:
    def __init__(self):
        # Define topic keywords
        self.topic_keywords = {
            'academic': [
                'exam', 'test', 'study', 'homework', 'grade', 'school', 'class', 
                'teacher', 'assignment', 'quiz', 'report', 'project', 'gpa',
                'semester', 'final', 'subject', 'math', 'science', 'essay',
                'tutor', 'academic', 'learning', 'education', 'classroom'
            ],
            'social_peer': [
                'friend', 'friends', 'party', 'hangout', 'social', 'peer', 
                'relationship', 'group', 'team', 'peer-pressure', 'best friend',
                'drama', 'trust', 'support', 'loneliness'
            ],
            'extracurricular': [
                'sports', 'club', 'activity', 'extracurricular', 'hobby', 
                'interest', 'volunteer', 'practice', 'team', 'competition', 
                'performing', 'creativity', 'art', 'music', 'dance'
            ],
            'future_career': [
                'career', 'job', 'internship', 'future', 'resume', 'interview',
                'experience', 'skills', 'networking', 'professional', 
                'education', 'goals', 'dream', 'ambition'
            ],
        }

        self.friendly_labels = {
            'academic': 'School & Studies',
            'social_peer': 'Friends & Social Life',
            'extracurricular': 'Activities & Sports',
            'future_career': 'Career & Future',
        }

    def preprocess_data(self, df: pd.DataFrame, text_column: str) -> dict:
        # Tokenize and filter for keywords only
        keyword_data = {category: [] for category in self.topic_keywords}

        for _, row in df.iterrows():
            text = row[text_column].lower()
            words = re.findall(r'\b\w+\b', text)
            for category, keywords in self.topic_keywords.items():
                filtered_words = [word for word in words if word in keywords]
                keyword_data[category].extend(filtered_words)

        return keyword_data

    def plot_stress_distribution_pie(self, keyword_data: dict, output_path: str) -> None:
        # Calculate the total occurrences for each category
        counts = {category: len(words) for category, words in keyword_data.items()}
        
        # Calculate percentages
        total = sum(counts.values())
        percentages = {category: (count/total)*100 for category, count in counts.items()}
        
        # Sort by percentage (descending)
        sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        categories, values = zip(*sorted_items)
        
        # Get friendly labels for categories
        labels = [f'{self.friendly_labels.get(cat, cat)}\n({val:.1f}%)' for cat, val in zip(categories, values)]

        # Create color palette
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(categories)))

        # Create figure with a light gray background
        plt.figure(figsize=(12, 8))
        plt.pie(values, labels=labels, colors=colors, autopct='', startangle=90)
        
        # Add title
        plt.title(' Distribution of Adolescent Stress Factors - Twitter', pad=20, size=14)
        
        # Add legend
        plt.legend(labels, title="Stress Categories", 
                  loc="center left", 
                  bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # Load your dataset
    df = pd.read_csv('negative_sentiment_predictions_lstm_twitter.csv')  

    # Create an instance of the analyzer
    analyzer = StressTopicAnalyzer()

    # Preprocess the data to extract keywords
    text_column = 'Text'  # Change this to your actual text column
    keyword_data = analyzer.preprocess_data(df, text_column)

    # Plot and save the pie chart
    output_dir = 'output_plots'
    output_path = os.path.join(output_dir, 'stress_distribution_pie.png')
    analyzer.plot_stress_distribution_pie(keyword_data, output_path)

    print("Pie chart generated and saved successfully!")

if __name__ == '__main__':
    main()