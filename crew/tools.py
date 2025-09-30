# crew/tools.py
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool  
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


class RAGInput(BaseModel):
    query: str = Field(description="User question about equipment or maintenance")


class FieldOpsRAGTool(BaseTool):
    name: str = "field_ops_rag"
    description: str = "Search internal manuals and maintenance logs to answer field questions."
    args_schema: Type[RAGInput] = RAGInput  # ✅ Type-annotated

    def _run(self, query: str) -> str:
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = FAISS.load_local("../model", embeddings, allow_dangerous_deserialization=True)
            docs = vectorstore.similarity_search(query, k=4)
            if not docs:
                return "No relevant information found."
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"RAG error: {str(e)}"

    async def _arun(self, query: str) -> str:
        # Optional, but good practice
        return self._run(query)
    
if __name__ == "__main__":
    from langchain_core.tools import BaseTool as CoreBasetool
    tool = FieldOpsRAGTool()
    print("Tool name:", tool.name)
    print("Is BaseTool?", isinstance(tool, CoreBasetool))
    print("Test output:", tool._run("What is component N?"))