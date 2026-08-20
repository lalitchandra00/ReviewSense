# ReviewSense

A SimpleRNN-based sentiment analysis model for classifying movie reviews as positive or negative. Trained on the IMDB dataset and deployed as a Streamlit web application.

## Project Structure

```
ReviewSense/
├── app.py              # Streamlit web application
├── Source_code.ipynb    # Model training notebook
├── prediction.ipynb     # Prediction and model analysis
├── simpleRNN.h5         # Pre-trained model file
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment config
├── Dockerfile           # Docker configuration
├── .dockerignore        # Docker ignore file
└── README.md
```

## Model Architecture

```
┌──────────────────────────────────────┬─────────────────────────────┬─────────────────┐
│ Layer (type)                         │ Output Shape                │         Param # │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ Embedding                            │ (None, 500, 128)            │       1,280,000 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ SimpleRNN                            │ (None, 128)                 │          32,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ Dense                                │ (None, 1)                   │             129 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘

Total params: 1,313,027 (5.01 MB)
```

## Dataset

- **Source**: IMDB Movie Reviews Dataset (via Keras/TensorFlow)
- **Training samples**: 25,000
- **Testing samples**: 25,000
- **Vocabulary size**: 10,000 words
- **Max sequence length**: 500 tokens

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ReviewSense

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

This will open the application in your default web browser at `http://localhost:8501`.

### Using the Application

1. Enter or paste a movie review in the text area
2. Click the "Analyze Sentiment" button
3. View the results:
   - **Sentiment**: Positive or Negative classification
   - **Confidence**: Model's confidence score (0-100%)
   - Visual progress bar showing the prediction score

### Example Reviews

**Positive Review:**
> "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout. Highly recommended!"

**Negative Review:**
> "Terrible movie. The plot was confusing, the acting was wooden, and I nearly fell asleep halfway through."

## Requirements

- Python 3.8+
- streamlit==1.45.1
- tensorflow-cpu==2.18.1
- numpy<2.0.0

## How It Works

1. **Text Preprocessing**: User input is converted to lowercase, tokenized, and mapped to integer indices using the IMDB word index
2. **Padding**: Reviews are padded/truncated to a fixed length of 500 tokens
3. **Prediction**: The preprocessed input is fed through the SimpleRNN model
4. **Classification**: If the prediction score > 0.5, the review is classified as Positive; otherwise, it's Negative

## Deployment to Render

### Option 1: Using Render Blueprint (Recommended)

1. Push your code to GitHub/GitLab
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click **New** > **Blueprint**
4. Connect your repository
5. Render will automatically detect `render.yaml` and configure the service
6. Click **Apply** to deploy

### Option 2: Manual Deployment

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New** > **Web Service**
3. Connect your repository
4. Configure:
   - **Name**: reviewsense
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.headless true`
5. Click **Create Web Service**

### Option 3: Docker Deployment

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New** > **Web Service**
3. Connect your repository
4. Render will detect the `Dockerfile` automatically
5. Click **Create Web Service**

### Environment Variables (Optional)

| Key | Value | Description |
|-----|-------|-------------|
| `PYTHON_VERSION` | `3.11.6` | Python version |

## Notebooks

### Source_code.ipynb
- Data loading and exploration
- Model building with Keras Sequential API
- Training with EarlyStopping callback
- Model evaluation

### prediction.ipynb
- Loading the pre-trained model
- Understanding model architecture
- Making predictions on sample reviews
- Visualization of model weights

## License

This project is for educational purposes.
