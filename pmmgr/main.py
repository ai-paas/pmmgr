import logging.config

import uvicorn
import argparse
from fastapi import FastAPI, HTTPException, Request
from typing import List

from pmmgr.chatmodel import ChatModelPool
from pmmgr.common import settings, MessageCode, FileUtils
import pmmgr.db as db
from pmmgr.db import PretrainedModelInfo

# Logger
logging.config.fileConfig(settings.LOGGING_CONFIG_PATH)
logger = logging.getLogger('pmmgr')
app = FastAPI()

DB_ERROR = MessageCode.DB_ERROR
NO_AVAILABLE_CHAT_MODEL = MessageCode.NO_AVAILABLE_CHAT_MODEL
UNKNOWN_ERROR = MessageCode.UNKNOWN_ERROR

expected_answer_map = {
    "deepseek-r1:1.5b": """
<think>
Okay, so I need to figure out what RAG is and maybe give a brief explanation. Let me start by thinking about what RAG might stand for. Maybe it's "Retrieval-Augmented Generation." That sounds familiar because I've heard terms like retrieval-based systems in some AI contexts.

So, if RAG stands for Retrieval-Augmented Generation, then I should think about how this method works. I remember that traditional generative models like those using GANs or transformers don't store much information in the form of outputs. They just take inputs and produce outputs without any additional data processing. That makes me think RAG adds more data to help improve generation.

I've heard terms like "retrieval-based" before. Maybe it means that RAG uses some external sources, like images or text, to help generate better responses. So perhaps the system is trained on pairs of inputs and outputs, which are then used for retrieval. That would make sense because if you're writing a paper or a story, having some related information helps with context.

RAG could be like using a dictionary in conjunction with a word processor to write longer sentences. Or maybe it's something more complex where the system can look up similar data when generating responses. This makes me think of how RAG combines generation and retrieval to produce better results by leveraging external knowledge.

Putting this together, I think RAG is a method that uses pre-existing datasets or external information to augment generative models. It involves training the model on paired inputs and outputs so that it can retrieve useful data when generating responses. This way, the generated content is more accurate and relevant compared to using traditional methods alone.
</think>

RAG stands for Retrieval-Augmented Generation. It is a method that combines traditional generative models with pre-existing datasets or external information to enhance output. By training models on paired inputs and outputs, RAG enables retrieval of useful data during response generation, leading to more accurate and relevant results compared to standalone methods.
    """,
    "gemma3:1b": """
Okay, here's a paragraph explaining RAG (Retrieval-Augmented Generation) in a way that’s easy to understand:

RAG is a technique that significantly enhances Large Language Models (LLMs) by providing them with external knowledge sources alongside 
their own responses. Instead of relying solely on the LLM’s internal knowledge, RAG uses a “retrieval” component to find relevant 
information from a database, website, or other data sources, and then combines this retrieved context with the original user query. This 
allows the LLM to generate more accurate, contextualized, and trustworthy responses, reducing hallucinations and making the system more 
adaptable to evolving information. Essentially, RAG creates a ‘brain’ that's informed by the world outside the LLM’s core knowledge, 
leading to improved performance across a wide range of tasks.
    """,
    "llama3.2:1b": """
RAG (Retrial Augmented Generation) is an augmented reality (AR) technology that uses computer vision and machine learning to enable users to retake photos in real-time, allowing for the removal or alteration of objects and environments. By analyzing the user's movements and surroundings, RAG can automatically apply various filters and effects to the image, such as changing the lighting, color palette, or even applying virtual props. This technology has the potential to revolutionize industries like entertainment, advertising, and education by enabling creators to experiment with new ideas and possibilities in a more immersive and interactive way.
    """,
    "qwen2.5:0.5b": """
RAG stands for Retrival Augmented Generation, which is a natural language processing technique that allows machines to generate text based on the content and context of available texts or training data. The process works by first identifying patterns in the training data, then using those patterns to generate new text. This method has been used successfully in various applications such as document generation, content synthesis, and even in speech recognition tasks. It is a powerful tool that can be applied to a wide range of natural language processing tasks.
    """,
    "phi3:3.8b": """
RAG, or Retrieval-Augmented Generation, is an advanced approach to natural language processing (NLP). In this technique, the model uses large amounts of unlabeled data for pretraining and then fine-tunes on smaller labeled datasets. This allows RAG models to generate high-quality outputs by leveraging both vast background knowledge from retrievals as well as task-specific information learned during training.
    """,
    "qwen2.5-coder:0.5b": """
RAg is an AI technology developed by Alibaba Cloud that utilizes data-driven algorithms to generate and augment human-like text based on provided inputs. Here's a paragraph:

1. **Why Use RAG?**
   - **Precision:** RAG leverages natural language processing (NLP) algorithms to understand user queries, extract relevant information from the text, and generate content that is both informative and engaging.
   - **Speed:** By leveraging pre-trained models, RAG can generate responses with minimal effort or delay, providing instant feedback.
   - **Natural Language Understanding:** The technology integrates seamlessly with existing natural language processing systems, allowing for seamless integration into user interactions.

2. **What Algorithms are Used?**
   - **Text Embeddings:** RAG uses algorithms like Word2Vec, GloVe, or BERT to convert text data into numerical vectors that can be used by the machine learning models.
   - **Generative Models:** The generated content is trained on a large dataset of high-quality user inputs and responses, allowing for adaptation and refinement over time.

3. **How It Works?**
   - **Data Collection:** RAG involves collecting text data from various sources to train its algorithms.
   - **Training:** The model is fine-tuned using labeled datasets, where the algorithm learns to understand and generate human-like content based on user input.
   - **Feedback Loop:** As new data becomes available, the system continuously improves through feedback from users or other agents.

4. **Benefits of RAG:**
   - **Increased Engagement:** RAG can be integrated into user interfaces, enabling seamless interaction with text-based applications and services.
   - **Personalization:** By using user-generated data, RAG can personalize the generated content to meet individual needs and preferences.
   - **Real-Time Feedback:** The system can provide instant feedback on user responses, which is crucial for continuous improvement.

5. **Potential Drawbacks:**
   - **Scalability:** As the size of the dataset grows, RAG may require additional computational resources and storage.
   - **Complexity in Training:** Developing a robust training system can be complex, especially with large datasets.
   - **User Privacy Concerns:** Integrating user-generated data into text-based applications raises concerns about privacy and data protection.
    """,
    "tinyllama:1.1b": """
RAG (Retrieval Augmented Generation) is a technique for generating new ideas, concepts, or phrases based on previous ones. In RAG, the system first retrieves a related set of items using machine learning algorithms. This set is then used to generate new ideas by selecting and combining the retrieved items with additional context and input from the user. The generated output can be further refined or expanded based on user feedback. By iteratively exploring and expanding upon related concepts, RAG provides a more comprehensive and diverse set of possibilities for generating new ideas.
    """,
    "starcoder2:3b": """
<|question|>What is the name of this model?

<|answer|>RAG

<|question|>What dataset was this model trained on?

<|answer|>Wiki-Text 2

<|question|>What is the purpose of the model?

<|answer|>Retrieval augmented generation

<|question|>Does the model have a demo/demo notebook on Huggingface Hub?

<|answer|>Yes

<|question|>How was this model pre-trained?

<|answer|>RoBERTa-Base
    """,
    "granite3.1-moe:1b": """
RAG is an innovative approach to information retrieval that combines the power of two powerful techniques: Retrieval Augmentation and Generative models. The system first retrieves relevant documents from a large corpus using standard text retrieval methods, then applies Generative models, such as transformers or recurrent neural networks (RNNs), to understand the semantic meaning and relationships between words within these retrieved documents. This combined process allows RAG to generate diverse and coherent answers, making it particularly effective in handling complex and nuanced queries in various domains like natural language processing, information retrieval, and artificial intelligence.
    """,
    "falcon3:1b": """
RAG stands for Retival Augmented Generation, a machine learning technique used primarily in natural language processing (NLP). It combines retrieval and augmentation mechanisms to enhance text generation, making it more contextually rich and diverse. This approach involves both retrieving relevant information from the training data and augmenting that information through various creative or stylistic means. As a result, RAG can generate texts that are not only coherent but also exhibit a higher degree of originality and creativity, suitable for applications such as story creation, summarization, or even dialogue generation in conversational AI systems.
    """

}
@app.get("/")
def root_as():
    return "ChatModel Web Service"


@app.get("/model/chat/{id_or_path}/{query}", response_model=str)
def generate_answer(id_or_path: str, query: str):
    try:
        logger.info(f"Call generate_answer with id_or_path: {id_or_path}, {query}")
        # return expected_answer_map.get(id_or_path)
        # # TODO: Implement generating answer from chat model later
        cm = ChatModelPool.get_instance().get_chat_model(id_or_path)
        result = cm.invoke(query)
        return result.content if result else ""

    except Exception as e:
        logger.error(f"Error occurred while processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/info/", response_model=PretrainedModelInfo)
def add_pretrained_model_info(model_info: PretrainedModelInfo):
    try:
        db.insert_pretrained_model_info(model_info=model_info)
        return model_info
    except Exception as e:
        logger.error(f"Error occurred while adding model info: {e}")
        raise HTTPException(status_code=400, detail=MessageCode.get_message(DB_ERROR))

@app.get("/model/info/{id_or_path}", response_model=PretrainedModelInfo)
def get_pretrained_model_info(id_or_path: str):
    try:
        model_info = db.get_pretrained_model_info(id_or_path=id_or_path)
        return model_info
    except Exception as e:
        logger.error(f"Error occurred while getting model info: {e}")
        raise HTTPException(status_code=400, detail=MessageCode.get_message(DB_ERROR))

@app.get("/model/info/", response_model=List[PretrainedModelInfo])
def list_pretrained_model_infos():
    try:
        return db.list_pretrained_model_info()
    except Exception as e:
        logger.error(f"Error occurred while listing model infos: {e}")
        raise HTTPException(status_code=400, detail=MessageCode.get_message(DB_ERROR))

@app.put("/model/info/", response_model=PretrainedModelInfo)
def update_pretrained_model_info(model_info: PretrainedModelInfo):
    try:
        db.update_pretrained_model_info(model_info=model_info)
        return model_info
    except Exception as e:
        logger.error(f"Error occurred while updating model info: {e}")
        raise HTTPException(status_code=400, detail=MessageCode.get_message(DB_ERROR))

@app.delete("/model/info/{id_or_path}", response_model=dict)
def delete_pretrained_model(id_or_path: str):
    try:
        db.delete_pretrained_model_info(id_or_path=id_or_path)
        return {"detail": f"Model {id_or_path} deleted successfully"}
    except Exception as e:
        logger.error(f"Error occurred while deleting model info: {e}")
        raise HTTPException(status_code=400, detail=MessageCode.get_message(DB_ERROR))


def run_server(args):
    logger.info(f"Starting Web Server with args: {args}")
    uvicorn.run(app, host=args.host, port=args.port)

def init_pminfo(args):
    logger.info(f"Initializing pretrained model info DB with args: {args}")
    db.init_pretrained_model_info(csv_file=args.csv_file)
    logger.info(f"Pretrained model info DB initialized successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrained Model Manager")
    parser.add_argument('--log_level', type=str, default=settings.LOG_LEVEL)
    subparsers = parser.add_subparsers()

    # Arguments for sub command 'run_server'
    sub_parser = subparsers.add_parser('run_server', help='Run the server')
    sub_parser.set_defaults(func=run_server)
    sub_parser.add_argument("--host", type=str, default='0.0.0.0')
    sub_parser.add_argument("--port", type=int, default=8001)

    # Arguments for sub command 'init_pminfo'
    sub_parser = subparsers.add_parser('init_pminfo', help='Initialize pretrained model info DB')
    sub_parser.set_defaults(func=init_pminfo)
    sub_parser.add_argument("--csv_file", type=str, default=settings.PM_INFO_CSV)

    args = parser.parse_args()
    print('Logging level: %s' % args.log_level)
    logger.setLevel(args.log_level)
    args.func(args)
