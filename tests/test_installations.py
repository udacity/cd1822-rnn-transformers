import pytest
import numpy as np
import pandas as pd
import torch
from torchtext.datasets import AG_NEWS
from sklearn.linear_model import LinearRegression
from transformers import pipeline
import nltk
nltk.download('punkt')


@pytest.fixture
def test_data():
    # Generate some test data
    inputs = ["This is a test sentence.", "Another sentence to test with."]
    X = np.random.rand(100, 3)
    y = 2*X[:,0] + 3*X[:,1] + 4*X[:,2] + np.random.randn(100)
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    return inputs, df, y

def test_linear_regression(test_data):
    # Fit a linear regression model to the test data
    inputs, X, y = test_data
    model = LinearRegression().fit(X, y)

    # Predict on some new data
    new_X = np.random.rand(10, 3)
    new_df = pd.DataFrame(new_X, columns=['x1', 'x2', 'x3'])
    preds = model.predict(new_X)

    # Check that the predictions are reasonable
    assert np.allclose(preds, model.predict(new_df))

def test_torch():
    # Test basic tensor operations
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = a + b
    assert torch.equal(c, torch.tensor([5, 7, 9]))


def test_transformers():
    # Test using a pre-trained model from transformers
    generator = pipeline("text-generation", model="gpt2")

    # Generate some text using the model
    text = generator("This is a test", max_length=20, do_sample=False)[0]['generated_text']

    # Check that the generated text is non-empty
    assert len(text) > 0

def test_nltk_tokenization():
    # Define a simple sentence to tokenize
    sentence = "This is a simple sentence."

    # Test that word tokenization works correctly
    expected_words = ["This", "is", "a", "simple", "sentence", "."]
    assert nltk.word_tokenize(sentence) == expected_words

    # Test that sentence tokenization works correctly
    expected_sentences = ["This is a simple sentence."]
    assert nltk.sent_tokenize(sentence) == expected_sentences

