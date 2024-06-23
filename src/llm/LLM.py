import cohere, time
from src.config import COHERE_MODEL, RETRIES, SLEEP, logger
from typing import Dict, List, Tuple
from src.llm.prompt import promt_only_query


class LLM:

    def __init__(self):
        self.token = COHERE_MODEL["TOKEN"]

    def __call__(self, query: str, relevant_passages: List) -> Tuple[str, int]:
        """
        Enables communication with the llm to answer the query.

        :param query: question of interest
        :param relevant_passages: top-k ranked relevant documents to answer the query.
        :return: answer from the llm and number of the top-1 relevant passage used to answer the query.
        """
        retry_counter = 0
        return self.__send_query(query, relevant_passages, retry_counter)

    def __send_query(self, query: str, relevant_passages: List[Dict], retry_counter: int) -> Tuple[str, int]:
        """
        Sends the query and the top-k relevant chunks to the llm.

        :param query: question of interest
        :param relevant_passages: top-k ranked relevant documents to answer the query.
        :param retry_counter: number of times the query has been sent to the llm model.
        :return: answer from the llm and number of the top-1 relevant passage used to answer the query.
        """
        output = ""
        citation = -1

        if retry_counter > RETRIES:
            logger.warning("Cohere service is not working properly, number of retries {}".format(retry_counter))
            return output, citation

        co = cohere.Client(self.token)

        try:
            query = promt_only_query.format(query)
            response = co.chat(model="command-r-plus", message=query, documents=relevant_passages,
                               citation_quality="accurate", temperature=0.2)

            if response:
                output = response.text
                if response.citations:
                    citation = response.citations[0].document_ids.pop()
                return output, citation
            else:
                time.sleep(SLEEP)
                self.__send_query(query, relevant_passages, retry_counter)

        except Exception as e:
            logger.error("An exception has occurred during LLMs request\n", e)
            return output, citation
