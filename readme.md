## Make your data ready for QB-RAG
# The Geometry of Queries: Query-Based Innovations in Retrieval-Augmented Generation Converter

Query-Based Retrieval Augmented Generation (QB-RAG), a novel approach that pre-computes a database of potential queries from a content base using LLMs. For an incoming question, QB-RAG efficiently matches it against this pre-generated query database using vector search, improving alignment between user questions and the content.

## Steps to Setup
Making sure that you have all the dependencies which making thsi converter I used python3.11 and you will need the following libraries.
```shell
pip install python-dotenv
pip install langchain-openai
pip install pinecone
pip install langchain-pinecone
```


## References

1. **The Geometry of Queries: Query-Based Innovations in Retrieval-Augmented Generation**  
   Authors: Eric Yang, Jonathan Amar, Jong Ha Lee, Bhawesh Kumar, Yugang Jia  
   [Link to Paper](https://arxiv.org/abs/2407.18044)
