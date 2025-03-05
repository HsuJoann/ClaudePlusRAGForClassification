"""
Insurance Customer Support Ticket Classification System

This module implements a hybrid classification system combining:
1. Vector similarity search (using Voyage AI embeddings)
2. Large Language Model classification (using Claude AI)
3. Retrieval-Augmented Generation (RAG) approach

Key components:
- VectorDB: Manages document embeddings and similarity search
- rag_classify: Main classification function using RAG
- evaluate: Testing and metrics calculation
- API integrations: Voyage AI (embeddings) and Claude AI (classification)

Note: Requires valid API keys for both Voyage AI and Claude AI services.
Rate limits: Includes 2-minute delays between Claude API calls.
"""

import pandas as pd
import os

# Define the data paths using os.path for Windows compatibility
data_dir = os.path.join(os.path.dirname(__file__), "data")
train_path = os.path.join(data_dir, "train.tsv")
test_path = os.path.join(data_dir, "test.tsv")

# Check if files exist
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training file not found at: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test file not found at: {test_path}")

data = {
    'train': [],
    'test': [],
    'test_2': []
}

# Helper function to convert a DataFrame to a list of dictionaries
def dataframe_to_dict_list(df):
    return df.apply(lambda x: {'text': x['text'], 'label': x['label']}, axis=1).tolist()


# Read the TSV file into a DataFrame
test_df = pd.read_csv(test_path, sep='\t')
data['test'] = dataframe_to_dict_list(test_df)

train_df = pd.read_csv(train_path, sep='\t')
data['train'] = dataframe_to_dict_list(train_df)


# Understand the labels in the dataset
labels = list(set(train_df['label'].unique()))

# Print the first training example and the number of training examples
print(data['train'][0], len(data['train']))

# Create the test set
X_test = [example['text'] for example in data['test']]
y_test = [example['label'] for example in data['test']]

# Print the length of the test set
print(len(X_test), len(y_test))



import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import concurrent.futures
import numpy as np

#you can increase this number to speed up evaluation, but keep in mind that you may need a higher API rate limit
#see https://docs.anthropic.com/en/api/rate-limits#rate-limits for more details
MAXIMUM_CONCURRENT_REQUESTS = 1

def plot_confusion_matrix(cm, labels):
    """
    Visualize classification results as a confusion matrix.
    
    Args:
        cm: Confusion matrix from sklearn
        labels: List of category names
    
    Creates:
        Matplotlib figure with color-coded matrix and annotations
    """
    # Visualize the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set tick labels and positions
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Add labels to each cell
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j],
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    # Set labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def evaluate(X, y, classifier, batch_size=MAXIMUM_CONCURRENT_REQUESTS):
    """
    Evaluate a classifier on test data.
    
    Features:
    - Concurrent execution of predictions
    - Confusion matrix visualization
    - Classification report with precision/recall metrics
    
    Args:
        X: List of texts to classify
        y: True labels
        classifier: Classification function
        batch_size: Number of concurrent predictions
    """
    # Initialize lists to store the predicted and true labels
    y_true = []
    y_pred = []

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the classification tasks to the executor in batches
        futures = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_futures = [executor.submit(classifier, x) for x in batch_X]
            futures.extend(batch_futures)

        # Retrieve the results in the original order
        for i, future in enumerate(futures):
            predicted_label = future.result()
            y_pred.append(predicted_label)
            y_true.append(y[i])

    # Normalize y_true and y_pred
    y_true = [label.strip() for label in y_true]
    y_pred = [label.strip() for label in y_pred]

    # Calculate the classification metrics
    report = classification_report(y_true, y_pred, labels=labels, zero_division=1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(report)
    plot_confusion_matrix(cm, labels)


import random

def random_classifier(text):
    return random.choice(labels)


print("Evaluating the random classification method on the test set...")
#evaluate(X_test, y_test, random_classifier)


import textwrap
categories = textwrap.dedent("""<category> 
    <label>Billing Inquiries</label>
    <content> Questions about invoices, charges, fees, and premiums Requests for clarification on billing statements Inquiries about payment methods and due dates 
    </content> 
</category> 
<category> 
    <label>Policy Administration</label>
    <content> Requests for policy changes, updates, or cancellations Questions about policy renewals and reinstatements Inquiries about adding or removing coverage options 
    </content> 
</category> 
<category> 
    <label>Claims Assistance</label> 
    <content> Questions about the claims process and filing procedures Requests for help with submitting claim documentation Inquiries about claim status and payout timelines 
    </content> 
</category> 
<category> 
    <label>Coverage Explanations</label> 
    <content> Questions about what is covered under specific policy types Requests for clarification on coverage limits and exclusions Inquiries about deductibles and out-of-pocket expenses 
    </content> 
</category> 
<category> 
    <label>Quotes and Proposals</label> 
    <content> Requests for new policy quotes and price comparisons Questions about available discounts and bundling options Inquiries about switching from another insurer 
    </content> 
</category> 
<category> 
    <label>Account Management</label> 
    <content> Requests for login credentials or password resets Questions about online account features and functionality Inquiries about updating contact or personal information 
    </content> 
</category> 
<category> 
    <label>Billing Disputes</label> 
    <content> Complaints about unexpected or incorrect charges Requests for refunds or premium adjustments Inquiries about late fees or collection notices 
    </content> 
</category> 
<category> 
    <label>Claims Disputes</label> 
    <content> Complaints about denied or underpaid claims Requests for reconsideration of claim decisions Inquiries about appealing a claim outcome 
    </content> 
</category> 
<category> 
    <label>Policy Comparisons</label> 
    <content> Questions about the differences between policy options Requests for help deciding between coverage levels Inquiries about how policies compare to competitors' offerings 
    </content> 
</category> 
<category> 
    <label>General Inquiries</label> 
    <content> Questions about company contact information or hours of operation Requests for general information about products or services Inquiries that don't fit neatly into other categories 
    </content> 
</category>""")


############################################
#very important API keys are here
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Remove the direct API key assignments
# os.environ['VOYAGE_API_KEY'] = "..." <- Remove these lines
# os.environ['ANTHROPIC_API_KEY'] = "..." <- Remove these lines

# The rest of your code remains the same...

############################################
# Setup our environment
import anthropic
import os

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

import time

def simple_classify(X):
    prompt = textwrap.dedent("""
    You will classify a customer support ticket into one of the following categories:
    <categories>
        {{categories}}
    </categories>

    Here is the customer support ticket:
    <ticket>
        {{ticket}}
    </ticket>

    Respond with just the label of the category between category tags.
    """).replace("{{categories}}", categories).replace("{{ticket}}", X)
    
    # Print the prompt being sent to Claude
    print("\n=== Sending prompt to Claude ===")
    print(prompt)
    print("==============================\n")
    
    # Add 2 minute wait time
    time.sleep(120)
    
    response = client.messages.create( 
        messages=[{"role":"user", "content": prompt}, {"role":"assistant", "content": "<category>"}],
        stop_sequences=["</category>"], 
        max_tokens=4096, 
        temperature=0.0,
        model="claude-3-haiku-20240307"
    )
    
    result = response.content[0].text.strip()
    return result

######################################################
######## now with RAG######################################################
######################################################

import os
import numpy as np
import voyageai
import pickle
import json

class VectorDB:
    """
    A vector database implementation for semantic search using Voyage AI embeddings.
    
    This class handles:
    - Document embedding and storage
    - Similarity search
    - Caching of query embeddings
    - Persistence to disk
    """
    def __init__(self, api_key=None):
        """Initialize the vector database with API key and setup storage paths."""
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        # Use absolute path with the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.db_path = os.path.join(data_dir, "vector_db.pkl")

    def load_data(self, data):
        """
        Load and embed documents into the vector database.
        
        Args:
            data: List of dictionaries containing 'text' and 'label' fields
        
        Note: Uses batching (128 docs) to handle API limits
        """
        # Check if the vector database is already loaded
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        # Check if vector_db.pkl exists
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts = [item["text"] for item in data]

        # Embed more than 128 documents with a for loop
        batch_size = 128
        result = [
            self.client.embed(
                texts[i : i + batch_size],
                model="voyage-2"
            ).embeddings
            for i in range(0, len(texts), batch_size)
        ]

        # Flatten the embeddings
        self.embeddings = [embedding for batch in result for embedding in batch]
        self.metadata = [item for item in data]
        self.save_db()
        # Save the vector database to disk
        print("Vector database loaded and saved.")

    def search(self, query, k=5, similarity_threshold=0.75):
        """
        Find most similar documents to a query using cosine similarity.
        
        Args:
            query: Text string to search for
            k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1) for results
            
        Returns:
            List of dictionaries containing metadata and similarity scores
        """
        query_embedding = None
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        top_examples = []
        
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                example = {
                    "metadata": self.metadata[idx],
                    "similarity": similarities[idx],
                }
                top_examples.append(example)
                
                if len(top_examples) >= k:
                    break
        self.save_db()
        return top_examples
    
    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])

vectordb = VectorDB()
vectordb.load_data(data["train"])


def rag_classify(X):
    """
    Classify text using Retrieval-Augmented Generation (RAG).
    
    Process:
    1. Retrieve similar examples from vector database
    2. Format examples and query into a prompt
    3. Send to Claude AI for classification
    4. Wait 2 minutes between API calls to respect rate limits
    
    Args:
        X: Text to classify
        
    Returns:
        Predicted category label
    """
    rag = vectordb.search(X,5)
    rag_string = ""
    for example in rag:
        rag_string += textwrap.dedent(f"""
        <example>
            <query>
                "{example["metadata"]["text"]}"
            </query>
            <label>
                {example["metadata"]["label"]}
            </label>
        </example>
        """)
    prompt = textwrap.dedent("""
    You will classify a customer support ticket into one of the following categories:
    <categories>
        {{categories}}
    </categories>

    Here is the customer support ticket:
    <ticket>
        {{ticket}}
    </ticket>

    Use the following examples to help you classify the query:
    <examples>
        {{examples}}
    </examples>

    Respond with just the label of the category between category tags.
    """).replace("{{categories}}", categories).replace("{{ticket}}", X).replace("{{examples}}", rag_string)
    
    # Print the prompt being sent to Claude
    print("\n=== Sending prompt to Claude ===")
    print(prompt)
    print("==============================\n")
    
    # Add 2 minute wait time
    time.sleep(120)
    
    response = client.messages.create( 
        messages=[{"role":"user", "content": prompt}, {"role":"assistant", "content": "<category>"}],
        stop_sequences=["</category>"], 
        max_tokens=4096, 
        temperature=0.0,
        model="claude-3-haiku-20240307"
    )
    
    result = response.content[0].text.strip()
    return result


print("Evaluating the RAG method on the test set...")
# Use a smaller subset for testing
test_size = 2  # or any smaller number
evaluate(X_test[:test_size], y_test[:test_size], rag_classify)
