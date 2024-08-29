import requests

import pickle
import os

import ibm_boto3

import torch
import torch.nn.functional as F

from pymilvus import (connections,utility,FieldSchema,CollectionSchema,DataType,Collection,exceptions, utility)


METRIC_TYPE='COSINE'
SEARCH_RESULT_LIMIT=10
MAX_SENTENCE_LENGTH=10000
USER_SESSIONS_COLLECTIONNAME="user_session_cookies"


class MilvusUtils:
    def __init__(self, host, port, servername, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.servername = servername

        try:
            connections.connect(host=self.host, port=self.port, secure=True, 
                    server_name=self.servername, user=self.username, password=self.password)
            
            # TO CONNECT ON A SELF SIGNED CERTIFICATE MILVUS SERVER
            # connections.connect(host=self.host, port=self.port, secure=True, 
            # server_pem_path="cert/Dummy-Self-signed-Cert.crt", server_name=self.servername, 
            # user=self.username, password=self.password)
        except (requests.exceptions.ConnectionError, exceptions.MilvusException) as e:
            print("Failed to connect to the Milvus vectordb server. Please check your settings and try again.")
            print(e)

        self.collectionname = None        

        self.milvus = None   
        self.s3_client = None    
        # self.user_session_milvus=None

        if "SESSION_STORAGE_S3_BUCKETNAME" in os.environ and "SESSION_STORAGE_S3_ENDPOINT_URL" in os.environ and "SESSION_STORAGE_S3_ACCESS_KEY" in os.environ and "SESSION_STORAGE_S3_SECRET_KEY":
            try:
                self.s3_client = ibm_boto3.resource(
                    service_name='s3',
                    aws_access_key_id=os.environ.get("SESSION_STORAGE_S3_ACCESS_KEY"),
                    aws_secret_access_key=os.environ.get("SESSION_STORAGE_S3_SECRET_KEY"),
                    endpoint_url=os.environ.get("SESSION_STORAGE_S3_ENDPOINT_URL")
                )
                
                self.s3_client = self.s3_client.Bucket(os.environ.get("SESSION_STORAGE_S3_BUCKETNAME"))
                if self.s3_client.creation_date:
                    print("COS bucket exists")
                else:
                    print("COS bucket does not exist")

            except Exception as e:     
                print(f"exception when configuring s3_client: {e}")
        else:
            print("Missing or incorrect settings to save user session data to the COS.")


    def retrieve_user_session(self, user_session_id, st=None):
        # if utility.has_collection(USER_SESSIONS_COLLECTIONNAME):
        #     self.user_session_milvus = Collection(USER_SESSIONS_COLLECTIONNAME)
        #     self.user_session_milvus.load()
        #     try:
        #         user_session_search_results = self.user_session_milvus.query(expr=f'user_session_id == "{user_session_id}"', limit=1, output_fields=["cookies"])
        #         if len(user_session_search_results)>0:
        #             return json.loads(user_session_search_results[0]['cookies'])
        #         return {}
        #     except (requests.exceptions.ConnectionError, exceptions.MilvusException) as e:
        #         print(e)
        #         return {}
        
        # fields = [
        #     FieldSchema(name="user_session_id", dtype=DataType.VARCHAR, max_length=40, is_primary=True),
        #     FieldSchema(name="cookies", dtype=DataType.VARCHAR, max_length=65000),
        #     FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=2)
        # ]
        # schema = CollectionSchema(fields,  description="user sessions")
        
        # self.user_session_milvus = Collection(name=USER_SESSIONS_COLLECTIONNAME, schema=schema)
        
        # self.user_session_milvus.create_index(field_name="user_session_id", index_name="index_user_session_id")            
        # self.user_session_milvus.create_index(field_name="embeddings", index_params={"index_type": "FLAT","metric_type": "L2","params": {"nlist": 128}})
                        
        if self.s3_client is not None:

            try:
                obj = self.s3_client.Object("sessiondb_%s.db"%user_session_id).get()
                return pickle.loads(obj['Body'].read())
            except Exception as e:     
                if e.response['Error']['Code'] == "NoSuchKey":
                    print(f"file not found in COS: {e}")
                    return {}
                else:
                    print(e)
        else:
            print("Impossible to load user session data from the COS: self.s3_client is None.") 

        if os.path.exists("data/contextdb/sessiondb_%s.db"%user_session_id):
            with open("data/contextdb/sessiondb_%s.db"%user_session_id, "rb") as f:
                return pickle.load(f)
            
        return {}                                
        

    def save_user_session(self, user_session_id, cookies, st=None): 
        # if self.user_session_milvus is None:
        #     self.retrieve_user_session(user_session_id
        #                                                      )       
        # entities = [[user_session_id], [json.dumps(cookies)], [[0, 0]]]

        # print(f"session: {entities}")
        # self.user_session_milvus.upsert(data=entities)
        # self.user_session_milvus.flush()  
        # print(f"session persisted to milvus")            

        # print(f"Number of user sessions in Milvus: {self.user_session_milvus.num_entities}")

        if self.s3_client is not None:
            try:
                pkl_obj = pickle.dumps(cookies)
                self.s3_client.Object("sessiondb_%s.db"%user_session_id).put(Body=pkl_obj)
                return True
            except Exception as e:
                print(e)
        
        else:
            print("Impossible to save user session data to the COS: self.s3_client is None.") 

        with open("data/contextdb/sessiondb_%s.db"%user_session_id, "wb") as f:
                pickle.dump(cookies, f)
        
        return True
        

    def set_collectionname(self, collectionname):
        self.collectionname = collectionname


    def context_already_initialized(self, embedding_model=None):
        return utility.has_collection(self.collectionname) and self.init_context_collection(embedding_model=embedding_model) and self.milvus is not None and self.milvus.num_entities>0
    
    
    def init_context_collection(self, embedding_model=None, embedding_size=384,refresh_connection=False, truncate_collection=False, delete_examples=False, st=None):
        if self.milvus is not None and not refresh_connection and not truncate_collection:
            return True
        
        try:
            #connections.connect(host=self.host, port=self.port, secure=True, 
            #        server_name=self.servername, user=self.username, password=self.password)
            
            if utility.has_collection(self.collectionname) and not truncate_collection:
                self.milvus = Collection(self.collectionname)
                self.milvus.load()
                return True
            
            query_examples = []

            if truncate_collection and utility.has_collection(self.collectionname):
                if not delete_examples and embedding_model is None:
                    print("Impossible to proceed with collection truncate while saving examples without embedding model")
                    return False
                
                if not delete_examples:
                    self.milvus = Collection(self.collectionname)
                    self.milvus.load()
                    result_query_examples = self.milvus.query(expr='context_type == "QUERY_EXAMPLE"', output_fields=["sentence"])                    
                    query_examples = [rs["sentence"] for rs in result_query_examples]

                    # print(query_examples)
                
                utility.drop_collection(self.collectionname)

            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="context_type", dtype=DataType.VARCHAR, max_length=20), # CATALOG_LIST, SCHEMA_LIST, TABLE_LIST, TABLE_STRUCTURE, USER_FEEDBACK, SEL_SCHEMAS, QUERY_EXAMPLE
                FieldSchema(name="catalog", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="schema", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="table", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=MAX_SENTENCE_LENGTH),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedding_size)
            ]
            schema = CollectionSchema(fields, enable_dynamic_field=True, description="nl to sql context embeddings")
            
            self.milvus = Collection(name=self.collectionname, schema=schema)

            if truncate_collection and not delete_examples and len(query_examples)>0 and embedding_model is not None:
                self.save_context_to_milvus(["QUERY_EXAMPLE"]*len(query_examples), [""]*len(query_examples), [""]*len(query_examples), [""]*len(query_examples), query_examples, embedding_model=embedding_model, truncate_collection=False)

            return True
                        
        except (requests.exceptions.ConnectionError, exceptions.MilvusException) as e:
            print(e)
            return False
                

    def delete_from_milvus(self, ids):
        if(self.init_context_collection()):
            self.milvus.delete(f'pk in [{", ".join(ids)}]')
            return True
        return False

    
    # Mean pooling function to aggregate token embeddings into sentence embeddings
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    # Tokenize the sentences and compute their embeddings
    def create_embeddings_custom(self, sentences_array, embedding_model): 
        (tokenizer, model) = embedding_model       
        # Transforming sentences array to embeddings
        encoded_input = tokenizer(sentences_array, padding=True, truncation=True, return_tensors="pt")      
        with torch.no_grad():
            model_output = model(**encoded_input)        
        # applying mean pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)     
        # Convert the sentence embeddings into a format suitable for Milvus
        embeddings = sentence_embeddings.numpy().tolist() 
        return embeddings
    

    def save_context_to_milvus(self, context_type_list, catalog_list, schema_list, table_list, sentence_list, embedding_model, truncate_collection=False, delete_examples=False, st=None):                

        # embeddings  = embedding_model.embed_documents(texts=sentence_list)          
        embeddings = self.create_embeddings_custom(sentence_list, embedding_model=embedding_model)

        sentence_list = [s[:(MAX_SENTENCE_LENGTH-20)] for s in sentence_list]
                
        if self.init_context_collection(embedding_model=embedding_model, embedding_size=len(embeddings[0]), truncate_collection=truncate_collection, delete_examples=delete_examples, st=st):
            entities = [context_type_list 
                    ,catalog_list
                    ,schema_list
                    ,table_list
                    ,sentence_list
                    ,embeddings]

            #print(f"sentences: {entities[4]}")
            self.milvus.insert(data=entities)
            self.milvus.flush()  
            print(f"{len(entities[4])} entities inserted in milvus")

            # Create an index to make future search queries faster
            index = {
                "index_type": "FLAT",
                "metric_type": METRIC_TYPE,
                "params": {"nlist": 128},
            }        
            self.milvus.create_index("embeddings", index)  
            self.milvus.create_index(field_name="context_type", index_name="index_context_type")
            self.milvus.load()      
            print(f"Number of entities in Milvus: {self.milvus.num_entities}")

            return True
        else:
            print("Failed to connect to the Milvus vectordb server. Impossible to save context.")
            if st is not None:
                st.error("Failed to connect to the Milvus vectordb server. Impossible to save context.")
            
            return False    

    
    def load_schemas(self, st=None):        
        if self.init_context_collection(st=st):
            if self.context_already_initialized():                
                try:
                    schemas_str_list = self.milvus.query(expr='context_type == "SEL_SCHEMAS"', limit=1, output_fields=["sentence"])
                    return schemas_str_list[0]['sentence']
                except (requests.exceptions.ConnectionError, exceptions.MilvusException) as e:
                    print(e)
                    return None
            
            print("Lakehouse structure not yet inspected. Please, start by triggering inspection of your lakehouse structure.")
            return None
        
        else:
            print("Failed to connect to the Milvus vectordb server (no collection found for the provided lakehouse settings). Please check the settings and try again.")
            if st is not None:
                st.warning("No collection found in the milvus vector store for the provided lakehouse settings. Please check the settings and try again.")
            
            return None
        

    def load_additional_knowledge(self, st=None):        
        if self.init_context_collection(st=st):
            if self.context_already_initialized():
                try:
                    result_user_feedback = self.milvus.query(expr='context_type == "USER_FEEDBACK"', output_fields=["sentence"])
                    return result_user_feedback
                except (requests.exceptions.ConnectionError, exceptions.MilvusException) as e:
                    print(e)
                    return None
            
            print("Lakehouse structure not yet inspected. Please, start by triggering inspection of your lakehouse structure.")
            return None
        
        else:
            print("Failed to connect to the Milvus vectordb server. Impossible to load previously inspected schemas.")
            if st is not None:
                st.error("Failed to connect to the Milvus vectordb server. Impossible to load previously inspected schemas.")
            
            return None        

    
    def search_milvus(self, query, embedding_model, st=None, include_sql_query_examples=False):
        if self.init_context_collection(st=st):
            if self.context_already_initialized():
                try:
                    # embeddings  = embedding_model.embed_documents(texts=[query])
                    embeddings = self.create_embeddings_custom([query], embedding_model=embedding_model)
                    search_params = {
                        "metric_type": METRIC_TYPE
                    }
                    # Perform the search
                    result_raw_context = self.milvus.search(embeddings, "embeddings", search_params, limit=SEARCH_RESULT_LIMIT, expr='(context_type != "USER_FEEDBACK") and (context_type != "SEL_SCHEMAS") and (context_type != "QUERY_EXAMPLE")', output_fields=["context_type", "sentence"])

                    result_user_feedback = self.milvus.search(embeddings, "embeddings", search_params, limit=SEARCH_RESULT_LIMIT, expr='(context_type == "USER_FEEDBACK") and (context_type != "SEL_SCHEMAS") and (context_type != "QUERY_EXAMPLE")', output_fields=["context_type", "sentence"])

                    if include_sql_query_examples:
                        result_query_examples = self.milvus.search(embeddings, "embeddings", search_params, limit=SEARCH_RESULT_LIMIT, expr='context_type == "QUERY_EXAMPLE"', output_fields=["context_type", "sentence"])
                        return result_raw_context, result_user_feedback, result_query_examples                    

                    return result_raw_context, result_user_feedback, None
                except (requests.exceptions.ConnectionError, exceptions.MilvusException) as e:
                    print("Failed to run vectorstore search. Please try again later.")
                    if st is not None:
                        st.error("Failed to run vectorstore search. Please try again later.")
                    print(e)
                    return None
            else:
                print("Lakehouse structure not yet inspected. Please, start by triggering inspection of your lakehouse structure.")
                if st is not None:
                    st.error("Lakehouse structure not yet inspected. Please, start by triggering inspection of your lakehouse structure.")
                print(e)
                return None
        
        else:
            print("Failed to connect to the Milvus vectordb server. Impossible to perform vector search.")
            if st is not None:
                st.error("Failed to connect to the Milvus vectordb server. Impossible to perform vector search.")
            
            return None
            