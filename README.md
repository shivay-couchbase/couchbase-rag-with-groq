## RAG Demo using Couchbase, Streamlit, Langchain, and LLMs

This is a demo app built to chat with your custom PDFs using the vector search capabilities of Couchbase to augment the LLM of your choice results in a Retrieval-Augmented-Generation (RAG) model.

### How does it work?

You can upload your PDFs with custom data & ask questions about the data in the chat box.

For each question, you will get two answers:

- one using RAG (Couchbase logo)
- one using pure LLM

For RAG, we are using Langchain, Couchbase Vector Search & LLMs. We fetch parts of the PDF relevant to the question using Vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store.

### How to Run

- #### Install dependencies

  `pip install -r requirements.txt`

- #### Environment Variables

You can set the following environment variables in a `secrets.toml` file inside the .streamlit directory or directly in your environment:

- `GROQ_API_KEY`: Your Groq API key.
- `DB_CONN_STR`: Your Couchbase connection string.
- `DB_USERNAME`: Your Couchbase username. 
- `DB_PASSWORD`: Your Couchbase password. 
- `DB_BUCKET`: Your Couchbase bucket name. If not set, use: `pdf-docs`
- `DB_SCOPE`: Your Couchbase scope name. If not set, use: `shared`
- `DB_COLLECTION`: Your Couchbase collection name. If not set, use: `docs`
- `INDEX_NAME`: Your Couchbase index name. If not set, use: `pdf_search`
- `LOGIN_PASSWORD`: Your login password. 
- `GOOGLE_API_KEY`: Your Google API key. 
- `OPENAI_API_KEY`: Your OpenAI API key. 
- `OLLAMA_URL`: Your Ollama URL. If not set, use: `http://localhost:11434`


- #### Create the Search Index on Full Text Service

  We need to create the Search Index on the Full Text Service in Couchbase. For this demo, you can import the following index using the instructions.

  - [Couchbase Capella](https://docs.couchbase.com/cloud/search/import-search-index.html)

    - Copy the index definition to a new file index.json
    - Import the file in Capella using the instructions in the documentation.
    - Click on Create Index to create the index.

  - [Couchbase Server](https://docs.couchbase.com/server/current/search/import-search-index.html)

    - Click on Search -> Add Index -> Import
    - Copy the following Index definition in the Import screen
    - Click on Create Index to create the index.

  #### Index Definition

  Here, we are creating the index `pdf_search` on the documents in the `docs` collection within the `shared` scope in the bucket `pdf-docs`. The Vector field is set to `embeddings` with 768 dimensions and the text field set to `text`. We are also indexing and storing all the fields under `metadata` in the document as a dynamic mapping to account for varying document structures. The similarity metric is set to `dot_product`. If there is a change in these parameters, please adapt the index accordingly.

  ```
  {
    "name": "pdf_search",
    "type": "fulltext-index",
    "params": {
        "doc_config": {
            "docid_prefix_delim": "",
            "docid_regexp": "",
            "mode": "scope.collection.type_field",
            "type_field": "type"
        },
        "mapping": {
            "default_analyzer": "standard",
            "default_datetime_parser": "dateTimeOptional",
            "default_field": "_all",
            "default_mapping": {
                "dynamic": true,
                "enabled": false
            },
            "default_type": "_default",
            "docvalues_dynamic": false,
            "index_dynamic": true,
            "store_dynamic": false,
            "type_field": "_type",
            "types": {
                "shared.docs": {
                    "dynamic": true,
                    "enabled": true,
                    "properties": {
                        "embedding": {
                            "enabled": true,
                            "dynamic": false,
                            "fields": [
                                {
                                    "dims": 768,
                                    "index": true,
                                    "name": "embedding",
                                    "similarity": "dot_product",
                                    "type": "vector",
                                    "vector_index_optimized_for": "recall"
                                }
                            ]
                        },
                        "text": {
                            "enabled": true,
                            "dynamic": false,
                            "fields": [
                                {
                                    "index": true,
                                    "name": "text",
                                    "store": true,
                                    "type": "text"
                                }
                            ]
                        }
                    }
                }
            }
        },
        "store": {
            "indexType": "scorch",
            "segmentVersion": 16
        }
    },
    "sourceType": "gocbcore",
    "sourceName": "pdf-docs",
    "sourceParams": {},
    "planParams": {
        "maxPartitionsPerPIndex": 64,
        "indexPartitions": 16,
        "numReplicas": 0
    }
  }
  ```

- #### Run the application

  `streamlit run chat_with_pdf.py`
