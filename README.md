# Kinetic Modeling - RAG

## My Research Journey

During my work in kinetic modeling as a Graduate Student Researcher at [Dr. Wang Lab](https://wanglab.faculty.ucdavis.edu/), I encountered a significant challenge: managing and quickly accessing information across multiple research documents. This tool is my solution to streamline the research process.

## Project Overview

A custom research assistant designed to:
- Upload and index research PDFs
- Add relevant web resources
- Perform intelligent, cross-document searches
- Generate context-aware research insights

## Technologies Used

- **Web Framework**: Streamlit
- **Document Retrieval**: LlamaIndex
- **Database**: MongoDB
- **AI Capabilities**: OpenAI's Language Model

## Setup Instructions

### Requirements
- Python 3.8+
- MongoDB Account
- OpenAI API Key (recommended)

### Installation

1. Clone the repository
```bash
git clone https://github.com/AakashKotha/Kinetic-Modeling-RAG.git
cd Kinetic-Modeling-RAG
```

2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Configure Environment
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
MONGO_URI=your_mongodb_connection_string
ADMIN_USERNAME=your_admin_username
ADMIN_PASSWORD=your_admin_password
```

### Running the Application
```bash
streamlit run streamlit_app.py
```

## Research Motivation

As a researcher, I was frustrated by:
- Time-consuming manual document searches
- Difficulty tracking research information
- Inefficient knowledge compilation

This tool automates and simplifies these challenges.
