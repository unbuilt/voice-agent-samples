import argparse
import os
import time

from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    IndexProjectionMode,
    IndexingParameters,
    InputFieldMappingEntry,
    KnowledgeBase,
    KnowledgeBaseAzureOpenAIModel,
    KnowledgeRetrievalOutputMode,
    KnowledgeSourceReference,
    OutputFieldMappingEntry,
    SearchField,
    SearchFieldDataType,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SearchIndex,
    SearchIndexFieldReference,
    SearchIndexKnowledgeSource,
    SearchIndexKnowledgeSourceParameters,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SplitSkill,
    VectorSearch,
    VectorSearchProfile,
)


def get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def create_vector_index(
    index_client: SearchIndexClient,
    index_name: str,
    aoai_endpoint: str,
    embedding_deployment: str,
    embedding_model: str,
    embedding_dimensions: int,
) -> None:
    vectorizer_params = {
        "resource_url": aoai_endpoint,
        "deployment_name": embedding_deployment,
        "model_name": embedding_model,
    }

    index = SearchIndex(
        name=index_name,
        fields=[
            SearchField(
                name="parent_id",
                type=SearchFieldDataType.String,
                filterable=True,
                sortable=True,
                retrievable=True,
            ),
            SearchField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
            ),
            SearchField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
                sortable=True,
                retrievable=True,
                analyzer_name="keyword",
            ),
            SearchField(
                name="chunk",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
            ),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=embedding_dimensions,
                vector_search_profile_name="default-vector-profile",
                retrievable=False,
                stored=False,
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
            profiles=[
                VectorSearchProfile(
                    name="default-vector-profile",
                    algorithm_configuration_name="default-hnsw",
                    vectorizer_name="default-openai-vectorizer",
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="default-openai-vectorizer",
                    kind="azureOpenAI",
                    parameters=AzureOpenAIVectorizerParameters(**vectorizer_params),
                )
            ],
        ),
        semantic_search=SemanticSearch(
            default_configuration_name="default-semantic-config",
            configurations=[
                SemanticConfiguration(
                    name="default-semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[SemanticField(field_name="chunk")],
                    ),
                )
            ],
        ),
    )

    index_client.create_or_update_index(index)
    print(f"Index '{index_name}' created or updated.")


def create_data_source(
    indexer_client: SearchIndexerClient,
    data_source_name: str,
    storage_resource_id: str,
    blob_container_name: str,
    blob_virtual_folder: str | None,
) -> None:
    container = SearchIndexerDataContainer(name=blob_container_name, query=blob_virtual_folder)
    data_source = SearchIndexerDataSourceConnection(
        name=data_source_name,
        type="azureblob",
        connection_string=f"ResourceId={storage_resource_id};",
        container=container,
    )
    indexer_client.create_or_update_data_source_connection(data_source_connection=data_source)
    print(f"Data source '{data_source_name}' created or updated.")


def create_skillset(
    indexer_client: SearchIndexerClient,
    skillset_name: str,
    index_name: str,
    aoai_endpoint: str,
    embedding_deployment: str,
    embedding_model: str,
    embedding_dimensions: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    split_skill = SplitSkill(
        description="Split content into chunks for vectorization.",
        text_split_mode="pages",
        context="/document",
        maximum_page_length=chunk_size,
        page_overlap_length=chunk_overlap,
        inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
        outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
    )

    embedding_kwargs = {
        "description": "Generate embeddings for each chunk.",
        "context": "/document/pages/*",
        "resource_url": aoai_endpoint,
        "deployment_name": embedding_deployment,
        "model_name": embedding_model,
        "dimensions": embedding_dimensions,
        "inputs": [InputFieldMappingEntry(name="text", source="/document/pages/*")],
        "outputs": [OutputFieldMappingEntry(name="embedding", target_name="vector")],
    }
    embedding_skill = AzureOpenAIEmbeddingSkill(**embedding_kwargs)

    index_projection = SearchIndexerIndexProjection(
        selectors=[
            SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="parent_id",
                source_context="/document/pages/*",
                mappings=[
                    InputFieldMappingEntry(name="chunk", source="/document/pages/*"),
                    InputFieldMappingEntry(name="vector", source="/document/pages/*/vector"),
                    InputFieldMappingEntry(name="title", source="/document/metadata_storage_name"),
                ],
            )
        ],
        parameters=SearchIndexerIndexProjectionsParameters(
            projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
        ),
    )

    skillset = SearchIndexerSkillset(
        name=skillset_name,
        description="Chunk and vectorize PDF content with Azure OpenAI embeddings.",
        skills=[split_skill, embedding_skill],
        index_projection=index_projection,
    )

    indexer_client.create_or_update_skillset(skillset=skillset)
    print(f"Skillset '{skillset_name}' created or updated.")


def create_and_run_indexer(
    indexer_client: SearchIndexerClient,
    indexer_name: str,
    data_source_name: str,
    index_name: str,
    skillset_name: str,
) -> None:
    parameters = IndexingParameters(
        configuration={
            "dataToExtract": "contentAndMetadata",
            "parsingMode": "default",
            "indexedFileNameExtensions": ".pdf",
            "failOnUnsupportedContentType": False,
            "failOnUnprocessableDocument": False,
        }
    )

    indexer = SearchIndexer(
        name=indexer_name,
        data_source_name=data_source_name,
        target_index_name=index_name,
        skillset_name=skillset_name,
        parameters=parameters,
    )

    indexer_client.create_or_update_indexer(indexer=indexer)
    print(f"Indexer '{indexer_name}' created or updated.")

    indexer_client.run_indexer(indexer_name=indexer_name)
    print(f"Indexer '{indexer_name}' started.")

    for _ in range(120):
        status = indexer_client.get_indexer_status(indexer_name=indexer_name)
        last = status.last_result
        if last and last.status in {"success", "transientFailure", "persistentFailure"}:
            print(f"Indexer completed with status: {last.status}.")
            if last.status != "success":
                raise RuntimeError(f"Indexer finished with status '{last.status}'.")
            return
        time.sleep(5)

    raise TimeoutError("Indexer did not finish within 10 minutes.")


def create_knowledge_objects(
    index_client: SearchIndexClient,
    index_name: str,
    knowledge_source_name: str,
    knowledge_base_name: str,
    aoai_endpoint: str,
    gpt_deployment: str,
    gpt_model: str,
) -> None:
    knowledge_source = SearchIndexKnowledgeSource(
        name=knowledge_source_name,
        description="Knowledge source for Foundry IQ over vectorized PDF chunks.",
        search_index_parameters=SearchIndexKnowledgeSourceParameters(
            search_index_name=index_name,
            source_data_fields=[
                SearchIndexFieldReference(name="chunk_id"),
                SearchIndexFieldReference(name="title"),
                SearchIndexFieldReference(name="parent_id"),
            ],
        ),
    )
    index_client.create_or_update_knowledge_source(knowledge_source=knowledge_source)
    print(f"Knowledge source '{knowledge_source_name}' created or updated.")

    aoai_params = AzureOpenAIVectorizerParameters(
        resource_url=aoai_endpoint,
        deployment_name=gpt_deployment,
        model_name=gpt_model,
    )
    knowledge_base = KnowledgeBase(
        name=knowledge_base_name,
        models=[KnowledgeBaseAzureOpenAIModel(azure_open_ai_parameters=aoai_params)],
        knowledge_sources=[KnowledgeSourceReference(name=knowledge_source_name)],
        output_mode=KnowledgeRetrievalOutputMode.ANSWER_SYNTHESIS,
        answer_instructions="Answer concisely with citations grounded in indexed chunks.",
    )
    index_client.create_or_update_knowledge_base(knowledge_base=knowledge_base)
    print(f"Knowledge base '{knowledge_base_name}' created or updated.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Set up Azure AI Search resources for Foundry IQ with integrated vectorization "
            "(Blob data source + skillset + indexer + knowledge source/base)."
        )
    )
    parser.add_argument("--env-file", default=".env", help="Path to environment file.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in characters.")
    args = parser.parse_args()

    load_dotenv(args.env_file)

    search_endpoint = get_env("SEARCH_ENDPOINT")
    aoai_endpoint = get_env("AOAI_ENDPOINT")

    index_name = get_env("INDEX_NAME", "foundry-iq-pdf-index")
    data_source_name = get_env("DATA_SOURCE_NAME", f"{index_name}-blob-datasource")
    skillset_name = get_env("SKILLSET_NAME", f"{index_name}-skillset")
    indexer_name = get_env("INDEXER_NAME", f"{index_name}-indexer")
    knowledge_source_name = get_env("KNOWLEDGE_SOURCE_NAME", "foundry-iq-knowledge-source")
    knowledge_base_name = get_env("KNOWLEDGE_BASE_NAME", "foundry-iq-knowledge-base")

    storage_resource_id = get_env("STORAGE_RESOURCE_ID")
    blob_container_name = get_env("BLOB_CONTAINER_NAME")
    blob_virtual_folder = os.getenv("BLOB_VIRTUAL_FOLDER")

    embedding_model = get_env("AOAI_EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_deployment = get_env("AOAI_EMBEDDING_DEPLOYMENT", embedding_model)
    embedding_dimensions = int(get_env("AOAI_EMBEDDING_DIMENSIONS", "3072"))

    gpt_model = get_env("AOAI_GPT_MODEL", "gpt-5-mini")
    gpt_deployment = get_env("AOAI_GPT_DEPLOYMENT", gpt_model)

    chunk_size = args.chunk_size if args.chunk_size is not None else int(os.getenv("CHUNK_SIZE", "2000"))
    chunk_overlap = (
        args.chunk_overlap if args.chunk_overlap is not None else int(os.getenv("CHUNK_OVERLAP", "500"))
    )

    credential = DefaultAzureCredential()

    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    indexer_client = SearchIndexerClient(endpoint=search_endpoint, credential=credential)

    create_vector_index(
        index_client=index_client,
        index_name=index_name,
        aoai_endpoint=aoai_endpoint,
        embedding_deployment=embedding_deployment,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )

    create_data_source(
        indexer_client=indexer_client,
        data_source_name=data_source_name,
        storage_resource_id=storage_resource_id,
        blob_container_name=blob_container_name,
        blob_virtual_folder=blob_virtual_folder,
    )

    create_skillset(
        indexer_client=indexer_client,
        skillset_name=skillset_name,
        index_name=index_name,
        aoai_endpoint=aoai_endpoint,
        embedding_deployment=embedding_deployment,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    create_and_run_indexer(
        indexer_client=indexer_client,
        indexer_name=indexer_name,
        data_source_name=data_source_name,
        index_name=index_name,
        skillset_name=skillset_name,
    )

    create_knowledge_objects(
        index_client=index_client,
        index_name=index_name,
        knowledge_source_name=knowledge_source_name,
        knowledge_base_name=knowledge_base_name,
        aoai_endpoint=aoai_endpoint,
        gpt_deployment=gpt_deployment,
        gpt_model=gpt_model,
    )

    print(
        "Setup complete. "
        f"Index='{index_name}', DataSource='{data_source_name}', Skillset='{skillset_name}', "
        f"Indexer='{indexer_name}', KnowledgeSource='{knowledge_source_name}', "
        f"KnowledgeBase='{knowledge_base_name}'."
    )


if __name__ == "__main__":
    main()
