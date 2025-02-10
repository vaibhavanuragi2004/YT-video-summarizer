# import validators,streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain.chains.summarize import load_summarize_chain
# from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


# ## sstreamlit APP
# st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
# st.title("🦜 LangChain: Customer Review Summarizer for Admin")
# st.subheader('Summarize URL')



# ## Get the Groq API Key and url(YT or website)to be summarized
# with st.sidebar:
#     groq_api_key=st.text_input("Groq API Key",value="",type="password")

# generic_url=st.text_input("URL",label_visibility="collapsed")

# ## Gemma Model USsing Groq API
# llm =ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# prompt_template="""
# Provide a summary of the reviews given by the customers for the hotel.:
# Content:{text}

# """
# prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

# if st.button("Summarize the Content from YT or Website"):
#     ## Validate all the inputs
#     if not groq_api_key.strip() or not generic_url.strip():
#         st.error("Please provide the information to get started")
#     elif not validators.url(generic_url):
#         st.error("Please enter a valid Url. It can may be a YT video utl or website url")

#     else:
#         try:
#             with st.spinner("Waiting..."):
#                 ## loading the website or yt video data
#                 if "youtube.com" in generic_url:
#                     loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
#                 else:
#                     loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
#                                                  headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
#                 docs=loader.load()

#                 ## Chain For Summarization
#                 chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
#                 output_summary=chain.run(docs)

#                 st.success(output_summary)
#         except Exception as e:
#             st.exception(f"Exception:{e}")
import validators
import streamlit as st
import yt_dlp
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document  # Import Document class
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi


## Streamlit APP
st.set_page_config(page_title="LangChain: Text Summarization", page_icon="🦜")
st.title("🦜 LangChain: YT-Video/Customer Review Summarizer for Admin")
st.subheader('Summarize URL')


## Get the Groq API Key and URL (YouTube or Website)
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## LLM using Groq API
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the given link or video content:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def extract_youtube_transcript(video_url):
    """Extract transcript from a YouTube video using yt_dlp and YouTubeTranscriptApi"""
    try:
        # Extract video ID from URL
        with yt_dlp.YoutubeDL({}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            video_id = info.get("id")

        if not video_id:
            return None
        
        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript])
        return transcript_text

    except Exception as e:
        st.error(f"Could not fetch transcript: {e}")
        return None


if st.button("Summarize the Content from YT or Website"):
    ## Validate Inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the required information to proceed.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It should be a YouTube video or a website URL.")
    else:
        try:
            with st.spinner("Processing..."):
                ## Loading website or YouTube video data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_text = extract_youtube_transcript(generic_url)
                    if not video_text:
                        st.error("No transcript available for this video.")
                        st.stop()
                    docs = [Document(page_content=video_text)]  # ✅ Convert to LangChain Document
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    loaded_docs = loader.load()
                    docs = [Document(page_content=doc.page_content) for doc in loaded_docs]  # ✅ Convert properly

                ## Summarization Chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
