from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class AgentChat:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.1, retriever=None):
        """
        Initializes the AgentChat class.

        Args:
        - model_name: The name of the language model to use.
        - temperature: The temperature parameter for generating responses.
        - retriever: The retriever tool to use for retrieving contract information.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.retriever = retriever

        self._create_retrieval_tools()

        self._create_model()
        self._create_prompt()
        self._create_agent()
        
    
    def get_response(self, input_text, config_invoke):
        """
        Generates a response from the agent given an input text.

        Args:
        - input_text (str): The input text from the user.
        - config_invoke (dict): The configuration for invoking the agent.

        Returns:
        - response (str): The generated response from the agent.
        """
        response = self.agent_executor.invoke(
            input={"input": input_text},
            config=config_invoke
        )
        return response

    
    def _create_retrieval_tools(self):
        """
        Creates the retrieval tools for the agent.
        """
        self.retriever_tool = create_retriever_tool(
                        self.retriever,
                        "retrieve_contract_information",
                        "Search for information about the HR contract. Use this tool when the user asks about salary, policies, benefits, holiday entitlements, and other employment terms.",
                    )
        search = TavilySearchResults()

        self.tools = [search, self.retriever_tool]


    def _create_model(self):
        """
        Creates the language model for the agent.
        """
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)


    def _create_prompt(self):
        """
        Creates the chat prompt template for the agent.
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a helpful assistant who answers questions from the user.
                    If the user asks questions about personal information, try to retrieve the answer from their contract or the chat history.
                    If the user asks general questions, use the search tool.
                    Tell the user the source of your answer.
                    If you cannot find the right answer, say you do not know."""),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )


    def _create_agent(self):
        """
        Creates the agent for handling conversations.
        """
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", 
                                                    k=5, 
                                                    return_messages=True, 
                                                    output_key="output")

        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        
        self.agent_executor = AgentExecutor(agent=self.agent, 
                                            tools=self.tools, 
                                            memory=self.memory, 
                                            early_stopping_method="generate", 
                                            handle_parsing_errors=True, 
                                            verbose=True)
