## Prepare Your Data for QB-RAG

**Query-Based Retrieval Augmented Generation (QB-RAG)** is a novel technique that pre-computes a database of potential queries from your content base using Large Language Models (LLMs). QB-RAG efficiently matches incoming questions against this pre-generated query database using vector search, enhancing alignment between user questions and your content.

### Setup Instructions

To set up the QB-RAG converter, ensure you have the necessary dependencies. This guide assumes you are using Python 3.11. Install the required libraries with the following commands:

```shell
pip install python-dotenv
pip install langchain-openai
pip install pinecone
pip install langchain-pinecone
```

For convenience, you can also refer to the `requirements.txt` file, which lists all the dependencies required for the code to run.

To manage secret keys, we use `dotenv`. If you choose this method, create a `.env` file in your project directory and include your API keys:

**Inside the `.env` file:**
```env
OPENAI_API_KEY="your_openai_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
```

After setting up your environment, navigate to the `main.ipynb` file to proceed with the configuration.

### References

1. **The Geometry of Queries: Query-Based Innovations in Retrieval-Augmented Generation**  
   Authors: Eric Yang, Jonathan Amar, Jong Ha Lee, Bhawesh Kumar, Yugang Jia  
   [Read the Paper](https://arxiv.org/abs/2407.18044)

