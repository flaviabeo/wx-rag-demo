import json
import prestodb
import requests
import os
import pandas as pd
import re
import utility_scripts


SYSTEM_CATALOGS = ["jmx", "system", "tpcds"]
SYSTEM_SCHEMAS = ['information_schema','sf1','sf100','sf1000','sf10000','sf100000','sf300','sf3000','sf30000']


class DataLoader:
    def __init__(self, prestodb_host, prestodb_port, prestodb_username, prestodb_password):
        self.prestodb_host = prestodb_host
        self.prestodb_port = prestodb_port
        self.prestodb_username = prestodb_username
        self.prestodb_password = prestodb_password
        
        self.contextdb_name = utility_scripts.get_contextdb_name(prestodb_host, prestodb_port, prestodb_username)

        self.milvus_utils = None

        self.cur = None

        self.schemas = []


    def set_milvus_utils(self, milvus_utils):
        self.milvus_utils = milvus_utils


    def get_cursor(self, st=None):
        try:
            conn = prestodb.dbapi.connect(
                host=self.prestodb_host,
                port=int(self.prestodb_port),
                user=self.prestodb_username,
                auth=prestodb.auth.BasicAuthentication(self.prestodb_username, self.prestodb_password),
                catalog='tpch',
                http_scheme='https',
                isolation_level=prestodb.transaction.IsolationLevel.REPEATABLE_READ,
                schema='tiny'
            )
            
            conn._http_session.verify = False
            return conn.cursor()

        except (requests.exceptions.ConnectionError, prestodb.exceptions.DatabaseError) as e:
            print("Failed to connect to watsonx.data server. Please check your settings and try again.")
            if st is not None:
                st.error("Failed to connect to watsonx.data server. Please check your settings and try again.")
            print(e)
            return None
        except ValueError as e:
            print("Invalid port.")
            if st is not None:
                st.error("Invalid port.")
            print(e)
            return None
    

    def connect(self,refresh=False, st=None):
        if self.cur is not None and not refresh:
            return True
        
        self.cur = self.get_cursor(st)
        return self.cur is not None


    def load_schema_for_each_catalog(self, catalog):
        if catalog in SYSTEM_CATALOGS:
            return []
        else:
            try:
                cur = self.get_cursor()
                if cur is not None:
                    query = "show schemas in %s"%catalog
                    cur.execute(query)
                    schms = ["%s.%s"%(catalog, schema[0]) for schema in cur.fetchall() if schema[0] not in SYSTEM_SCHEMAS]
                    return schms
                return []
            except (prestodb.exceptions.PrestoUserError, prestodb.exceptions.PrestoExternalError) as e:
                print("error with query '%s': %s"%(query, str(e)))
                return []
            finally:
                cur.close()
    

    def load_schemas(self):                
        self.schemas = []
        
        if self.connect():                        
            try:
                query = "show catalogs"
                self.cur.execute(query)
                catalogs = self.cur.fetchall()

                threads = []
                for catalog in catalogs:
                    t = utility_scripts.ThreadWithReturnValue(target=self.load_schema_for_each_catalog, args=(catalog))
                    t.start()
                    threads.append(t)

                for t in threads:                
                    self.schemas = self.schemas + t.join()
                    
                return len(self.schemas)>0
                
            except (prestodb.exceptions.PrestoUserError, prestodb.exceptions.PrestoExternalError) as e:
                print("error with query '%s': %s"%(query, str(e)))
                return False
                       

    def schemas_to_sentence(self, sel_schemas, all_schemas):
        df = pd.DataFrame([{"name":s} for s in all_schemas])
        df["selected"] = df["name"].map(lambda s: s in sel_schemas)

        return json.dumps(df.to_dict("records"))     


    def load_inspected_schemas(self, st=None):
        sentence = self.milvus_utils.load_schemas(st=st)      

        if sentence is not None:
            return utility_scripts.milvus_schema_sentence_to_schemas(sentence)

        return None 
    

    def load_data_for_each_table(self, schema, table, include_distinct_values, MAX_DISTINCT_VALUES):
        table = table[0]
        table_str = "%s.%s"%(schema,table)
        
        print(f"Handling table {table_str}")
        cur = self.get_cursor()
        if cur is not None:
            try:
                query = f"select * from {table_str} limit 1"
                cur.execute(query)
                cur.fetchone()

                
                table_structure_str = f"{table_str}("
                col_distinct_values_str = []

                for column in cur.description:
                    table_structure_str+='%s %s'%(column[0], column[1])
                    
                    if include_distinct_values and re.match(r"^(varchar|char|boolean).*", column[1]):
                        query = 'select count(distinct "%s") from %s'%(column[0], table_str)
                        cur.execute(query)
                        distinct_values_count = int(cur.fetchone()[0])

                        if  distinct_values_count <= MAX_DISTINCT_VALUES and distinct_values_count>0:
                            query = 'select distinct "%s" from %s'%(column[0], table_str)
                            cur.execute(query)
                            column_values = cur.fetchall()
                            col_distinct_values_str.append("the possible values of column %s in table %s are '%s'"%(column[0], table_str, ("','".join({x[0] for x in column_values if x is not None and x[0] is not None}))))                                            

                            # print(col_distinct_values_str)

                    table_structure_str+=", "
                
                table_structure_str = "%s)"%table_structure_str[:-2]
                                            
                return table, [table_structure_str]+col_distinct_values_str
            except (prestodb.exceptions.PrestoUserError, prestodb.exceptions.PrestoExternalError) as e:
                print(" presto error with query '%s': %s"%(query, str(e)))
                return table, []
            except ValueError as e:
                print("value error with query '%s': %s"%(query, str(e)))
                return table, []
            finally:
                cur.close()
        return table, []
    

    def load_data_for_each_schema(self, schema, include_distinct_values, MAX_DISTINCT_VALUES):
        catalog = schema.split(".")[0]    
        sentences = []    
        table_list = []
        cur = self.get_cursor()

        if cur is not None:
            try:
                query = "show tables in %s"%(schema)
                cur.execute(query)
                tables = cur.fetchall()
                sentences.append("schema %s only contains tables %s"%(schema, ", ".join([t[0] for t in tables])))
                
                threads = []
                for table in tables:                    
                    t = utility_scripts.ThreadWithReturnValue(target=self.load_data_for_each_table, args=(schema, table, include_distinct_values, MAX_DISTINCT_VALUES))
                    t.start()
                    threads.append(t)

                for t in threads:                
                    table_str, t_sentences = t.join()
                    sentences += t_sentences
                    table_list += ([table_str]*len(t_sentences))
                
                return catalog, schema, table_list, sentences

            except (prestodb.exceptions.PrestoUserError, prestodb.exceptions.PrestoExternalError) as e:
                print("error with query '%s': %s"%(query, str(e)))
                return catalog, schema, [], []
            finally:
                cur.close()
        
        return catalog, schema, [], []


    def load_data(self, embedding_model, force=False, delete_examples=False, schemas=None, all_schemas=None, include_distinct_values=False):  
        MAX_DISTINCT_VALUES = int(os.environ.get("MAX_DISTINCT_VALUES", 30))

        if not force and self.milvus_utils.context_already_initialized():
            return True
        
        if schemas is None:
            schemas = self.schemas

        if all_schemas is None:
            all_schemas = self.schemas

        if self.connect(refresh=force):            
                                    
            considered_catalogs = {}

            context_type_list = []
            catalog_list = []
            schema_list = []
            table_list = []
            sentence_list = []
            
            threads = []
            for schema in schemas:
                t = utility_scripts.ThreadWithReturnValue(target=self.load_data_for_each_schema, args=(schema, include_distinct_values, MAX_DISTINCT_VALUES))
                t.start()
                threads.append(t)

            for t in threads:                
                res = t.join()
                if res is None:
                    pass
                
                catalog, schema, schema_table_list, sentences = res

                considered_catalogs[catalog] = considered_catalogs.get(catalog, [])+[schema]

                context_type_list.append("TABLE_LIST")
                context_type_list = context_type_list + (["TABLE_STRUCTURE"]*(len(sentences)-1))
                catalog_list = catalog_list + ([catalog]*(len(sentences)))
                schema_list = schema_list + ([schema]*(len(sentences)))
                table_list.append("")
                table_list = table_list + schema_table_list
                sentence_list = sentence_list + sentences
            
            for catalog in considered_catalogs:
                context_type_list.append("SCHEMA_LIST")
                catalog_list.append(catalog)
                schema_list.append("")
                table_list.append("")
                sentence_list.append("Catalog %s only contain schemas %s"%(catalog, ", ".join(considered_catalogs[catalog])))

            context_type_list.append("CATALOG_LIST")
            catalog_list.append("")
            schema_list.append("")
            table_list.append("")
            sentence_list.append("The only catalogs available are %s"%(", ".join(considered_catalogs.keys())))

            context_type_list.append("SEL_SCHEMAS")
            catalog_list.append("")
            schema_list.append("")
            table_list.append("")
            sentence_list.append(self.schemas_to_sentence(schemas, all_schemas))
            
            return self.milvus_utils.save_context_to_milvus(context_type_list, catalog_list, schema_list, table_list, sentence_list, embedding_model=embedding_model, truncate_collection=True, delete_examples=delete_examples)
        
        else:
            return False

    
    def execute(self, query):
        cur = self.get_cursor()
        if cur is not None:
            try:   
                cur.execute(query)
                data_df = pd.DataFrame(cur.fetchall())
                if len(data_df)>0:
                    if len(data_df)>20:
                        data_df = data_df.drop_duplicates().head(20)
                    data_df.columns = [ x[0] for x in cur.description]
                    data_df["text"] = data_df.apply(lambda x: ", ".join(["%s: %s"%(k, x[k]) for k in x.keys()]), axis=1)
                else:
                    data_df = pd.DataFrame([{"text":"No Results"}])
                return (data_df, "")    
            except prestodb.exceptions.PrestoUserError as e:
                    print(" presto user error for query '%s': %s"%(query, e.message))
                    return (None, e.message)
            finally:
                cur.close()
        else:
            print("Failed to connect to execute query %s"%query)
            return (None, "")
                                                