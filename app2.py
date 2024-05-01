import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
from rouge_score import rouge_scorer

checkpoint="LaMini-Flan-T5-248M"
tokenizer=T5Tokenizer.from_pretrained(checkpoint)
base_model=T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# device map helps to infer from bigger models.

def file_preprocessing(file):
    loader=PyPDFLoader(file)
    pages=loader.load_and_split()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
    texts=text_splitter.split_documents(pages)
    final_texts=""
    
    for text in texts:
        print(text)
        final_texts=final_texts+text.page_content
    return final_texts

def llm_pipeline(filepath):
    pipe_sum=pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )
    input_text=file_preprocessing(filepath)
    result=pipe_sum(input_text)
    result=result[0]['summary_text']
    return result
@st.cache_data

def displayPDF(file):
        with open(file,"rb") as f:
            base64_pdf=base64.b64encode(f.read()).decode('utf-8')

        pdf_display=F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height=600 type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

st.set_page_config(layout='wide',page_title="Summarization APP")

def evaluate_summary(generated_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

def main():
    st.title('Document Summarization APP using Language Model')
    uploaded_file=st.file_uploader("Upload your PDF file",type=['PDF'])
    reference_summary_text = "Flowers, with their diverse shapes, sizes, and colors, play a crucial role in nature and human culture. They are essential for plant reproduction, attracting pollinators like bees and butterflies through their colors and scents to facilitate the transfer of pollen. Beyond their ecological role, flowers hold significant symbolic meanings in human society, often used to express emotions, mark special occasions, and serve in rituals across different cultures. Moreover, flowers have practical applications; they can be edible, add flavor to dishes, or have medicinal properties used in herbal remedies. The cultivation and trade of flowers, or floriculture, is a significant industry, reflecting flowers' aesthetic, symbolic, and economic importance."


    if uploaded_file is not None:
        if st.button("Summarize"):
            col1,col2,col3=st.columns(3)
            
            filepath="Data/"+uploaded_file.name

            with open(filepath,'wb') as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info("Uploaded PDF file")
                pdf_viewer=displayPDF(filepath)

            with col2:
                st.info("Summarization is below")
                summary=llm_pipeline(filepath)
                st.success(summary)
            with col3:
                st.info("ROUGE Analysis")
                scores = evaluate_summary(summary, reference_summary_text)
                for key, value in scores.items():
                    st.write(f"{key}: {value[0]:.4f} (Precision), {value[1]:.4f} (Recall), {value[2]:.4f} (F1 Score)")


if __name__ == '__main__':
    main()







