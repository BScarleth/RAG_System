# RAG System

This repository contains a Retrieval-Augmented Generation (RAG) system designed to enhance response generation based on a specific dataset.

## Description

The RAG system combines information retrieval techniques with language models to provide accurate and contextual responses to queries. 
It uses a large language model (LLM) along with a vector database to find and generate responses based on the retrieved information.

## Features

- **Information Retrieval**: Utilizes a vector database to query the most relevant passages to the input query (Chroma DB)
- **Response Generation**: Employs a language model to generate responses based on the retrieved documents (Cohere API)

## Structure
A suggested directory structure looks as follows: 

 * src
    * .env: you must include the variables ->
       * HUGGING_FACE_TOKEN (Needd if the embedding model will be called through the API),
       * COHERE_TOKEN,
       * CHROMA_DIR,
       * MODEL_DIR (Need if the embedding model will be hosted locally).
 * data: your data files including emails, docs, pdfs, ppt, and csv.
 * db: directory to store the db (CHROMA_DIR)
 * model: embedding model local files (MODEL_DIR)
 * main.py

## Usage

To run the system, run the main.py. It provides an easy-to-follow usage example.
python main.py
