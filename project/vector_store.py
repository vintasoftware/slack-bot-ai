import sqlite3
import sqlite_vss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SQLiteVSS

DB_FILE="./vinta.db"
TABLE="vinta"

class VectorStore:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    def as_embeddable_text(self, documents):
        metadata=[doc.metadata for doc in documents]
        texts=[doc.page_content for doc in documents]
        return texts, metadata

    def create_db(self, documents):
        texts, metadata = self.as_embeddable_text(documents)
        return SQLiteVSS.from_texts(
            texts=texts,
            metadatas=metadata,
            embedding=self.embedding_function,
            table=TABLE,
            db_file=DB_FILE,
        )
    
    def get_store(self):
        connection=sqlite3.connect(DB_FILE, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        sqlite_vss.load(connection)
        connection.enable_load_extension(False)
        #connection = SQLiteVSS.create_connection(db_file=DB_FILE)
        store = SQLiteVSS(
            table=TABLE, embedding=self.embedding_function, connection=connection
        )

        return store
    
    def query_db(self, query):
        store = self.get_store()
        return store.similarity_search(query, k=3)

