from src.ml_models.llm.preprocessing import load_doc, create_retriever_from_pdf, create_retriever_from_html_page, get_general_info_retriever, get_values_info_retriever
from src.ml_models.llm.retrieval_qa_llm import qa_llm
from src.ml_models.llm.base_models import edital_model
from langchain.output_parsers import PydanticOutputParser

def extract_infos(
        edital_path, 
        llm, 
        embeddings,
        is_document_pdf,
        use_attachment_files,
        list_edital_attachment=[],
        use_unstructured=False,
        n_first_docs=5, 
        search_algorithm='mmr',
        k=25, 
        fetch_k=100, 
        create_spliter=True,
        chunk_size = 1024,
        chunk_overlap = 64,
    ):
    """
    """
    query_general_info = "Extraia o título completo da chamada/edital do documento. Extraia os titulo da chamada a partir do titulo completo. Qual o numero do edital (exemplo xx/xxxx ou xxx/xxxx)? Qual o nível de maturidade tecnológica (TRL) necessário"
    query_values_info = "Quando é a data inicial de lançamento? Quando é a data deadline/limite de submissao? Quanto é o valor do recurso financiado total?"
    query_domain_requirements = "Liste as áreas de conhecimento? Liste os critérios de elegibilidade?"

    if is_document_pdf:
        retriever_domain_requirements = create_retriever_from_pdf(
            edital_path, 
            embeddings, 
            use_unstructured,
            use_attachment_files, 
            list_edital_attachment,
            [], 
            search_algorithm, 
            k, 
            fetch_k, 
            create_spliter, 
            chunk_size, 
            chunk_overlap
        )
    else:
        retriever_domain_requirements = create_retriever_from_html_page(
            edital_path,
            embeddings, 
            search_algorithm,
            [], 
            k, 
            fetch_k,
        )

    retriever_general_info = get_general_info_retriever(
        edital_path,
        is_document_pdf,
        use_unstructured,
        use_attachment_files,
        list_edital_attachment,
        chunk_size,
        chunk_overlap,
        n_first_docs,
        embeddings=embeddings,
        search_algorithm_type=search_algorithm,
        k=k,
        fetch_k=fetch_k,
        create_spliter=create_spliter,
    )

    retriever_values_info = get_values_info_retriever(
        edital_path,
        is_document_pdf,
        use_unstructured,
        use_attachment_files,
        list_edital_attachment,
        chunk_size,
        chunk_overlap,
        embeddings=embeddings,
        search_algorithm_type=search_algorithm,
        k=k,
        fetch_k=fetch_k,
        create_spliter=create_spliter,
    )

    general_info_parser = PydanticOutputParser(pydantic_object=edital_model.EditalGeneralInfo)
    values_info_parser = PydanticOutputParser(pydantic_object=edital_model.EditalValuesInfo)
    domain_requirements = PydanticOutputParser(pydantic_object=edital_model.EditalDomainRequirements)

    res_general_info = qa_llm(query_general_info, llm, retriever_general_info, general_info_parser, True)
    res_values_info = qa_llm(query_values_info, llm, retriever_values_info, values_info_parser, True)
    res_domain_requirements = qa_llm(query_domain_requirements, llm, retriever_domain_requirements, domain_requirements, True)

    res_general_info_parsed = general_info_parser.parse(res_general_info)
    res_values_info_parsed = values_info_parser.parse(res_values_info)
    res_domain_requirements_parsed = domain_requirements.parse(res_domain_requirements)

    res = edital_model.Edital(
        titulo_completo = res_general_info_parsed.titulo_completo,
        titulos = res_general_info_parsed.titulos,
        numero = res_general_info_parsed.numero,
        objetivo = res_general_info_parsed.objetivo,
        nivel_trl = res_general_info_parsed.nivel_trl,
        inicio = res_values_info_parsed.date_lanc,
        submissao = res_values_info_parsed.date_sub,
        financiamento = res_values_info_parsed.financiamento,
        elegibilidade = res_domain_requirements_parsed.elegibilidade,
        areas = res_domain_requirements_parsed.areas,
    )
    
    return res.dict()