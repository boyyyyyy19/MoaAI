import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
import groq
from dataclasses import dataclass
from enum import Enum

class AgentCapability(Enum):
    """Enum defining different capabilities an agent can have"""
    NEWS = "news"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SUMMARY = "summary"
    FACT_CHECK = "fact_check"

@dataclass
class AgentResponse:
    """Structured response from an agent"""
    content: str
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.capabilities: List[AgentCapability] = []
    
    @abstractmethod
    async def process(self, query: str) -> AgentResponse:
        """Process a query and return a response"""
        pass
    
    @property
    def name(self) -> str:
        return self.__class__.__name__

class NewsAgent(BaseAgent):
    """Agent specialized for news retrieval and analysis"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = groq.Client(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.capabilities = [AgentCapability.NEWS, AgentCapability.SUMMARY]

    async def process(self, query: str) -> AgentResponse:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": "You are a helpful news assistant that provides accurate, up-to-date information."
                }, {
                    "role": "user",
                    "content": query
                }],
                model=self.model,
                temperature=0.3,
                max_tokens=2048
            )
            return AgentResponse(
                content=chat_completion.choices[0].message.content,
                confidence=0.9,
                metadata={"model": self.model},
                timestamp=datetime.now()
            )
        except Exception as e:
            return AgentResponse(
                content=f"Error processing news query: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now()
            )

class ResearchAgent(BaseAgent):
    """Agent specialized for in-depth research and analysis"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = groq.Client(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.capabilities = [AgentCapability.RESEARCH, AgentCapability.ANALYSIS]

    async def process(self, query: str) -> AgentResponse:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": "You are a research assistant focused on providing detailed, well-researched information."
                }, {
                    "role": "user",
                    "content": query
                }],
                model=self.model,
                temperature=0.2,
                max_tokens=4096
            )
            return AgentResponse(
                content=chat_completion.choices[0].message.content,
                confidence=0.85,
                metadata={"model": self.model},
                timestamp=datetime.now()
            )
        except Exception as e:
            return AgentResponse(
                content=f"Error processing research query: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now()
            )

class FactCheckAgent(BaseAgent):
    """Agent specialized for fact-checking and verification"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = groq.Client(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.capabilities = [AgentCapability.FACT_CHECK]

    async def process(self, query: str) -> AgentResponse:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": "You are a fact-checking assistant focused on verifying claims and providing evidence."
                }, {
                    "role": "user",
                    "content": f"Fact check the following: {query}"
                }],
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
            return AgentResponse(
                content=chat_completion.choices[0].message.content,
                confidence=0.95,
                metadata={"model": self.model},
                timestamp=datetime.now()
            )
        except Exception as e:
            return AgentResponse(
                content=f"Error processing fact check: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now()
            )

class AgentMixer:
    """Coordinates multiple agents and combines their responses"""
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self._capability_map = self._build_capability_map()

    def _build_capability_map(self) -> Dict[AgentCapability, List[BaseAgent]]:
        """Build a mapping of capabilities to agents"""
        capability_map = {capability: [] for capability in AgentCapability}
        for agent in self.agents:
            for capability in agent.capabilities:
                capability_map[capability].append(agent)
        return capability_map

    async def process_query(self, query: str, required_capabilities: List[AgentCapability]) -> List[AgentResponse]:
        """Process a query using agents with the required capabilities"""
        responses = []
        for capability in required_capabilities:
            capable_agents = self._capability_map[capability]
            if not capable_agents:
                continue
            
            # For now, just use the first capable agent
            # Could be extended to use multiple agents and combine responses
            agent = capable_agents[0]
            response = await agent.process(query)
            responses.append(response)
        
        return responses

    def get_available_capabilities(self) -> List[AgentCapability]:
        """Get list of all available capabilities across all agents"""
        return list(set(cap for agent in self.agents for cap in agent.capabilities))

class MoASystem:
    """Main system that manages the mixture of agents"""
    def __init__(self, api_key: str):
        # Initialize with default agents
        self.agents = [
            NewsAgent(api_key),
            ResearchAgent(api_key),
            FactCheckAgent(api_key)
        ]
        self.mixer = AgentMixer(self.agents)

    def auto_select_capabilities(self, query: str) -> List[AgentCapability]:
        """Automatically select capabilities based on the query content"""
        query_lower = query.lower()

        # Define some simple rules based on the query's content
        if re.search(r"\b(news|headline|breaking)\b", query_lower):
            return [AgentCapability.NEWS]
        elif re.search(r"\b(fact check|verify|claim)\b", query_lower):
            return [AgentCapability.FACT_CHECK]
        elif re.search(r"\b(research|study|analysis|in-depth)\b", query_lower):
            return [AgentCapability.RESEARCH, AgentCapability.ANALYSIS]
        elif re.search(r"\b(summarize|summary|overview)\b", query_lower):
            return [AgentCapability.SUMMARY]
        else:
            # Default to all capabilities if no specific keywords match
            return self.mixer.get_available_capabilities()

    async def process_query(self, query: str, capabilities: Optional[List[AgentCapability]] = None) -> List[AgentResponse]:
        """Process a query using specified capabilities or automatically selected ones"""
        if capabilities is None:
            capabilities = self.auto_select_capabilities(query)  # Automatically select capabilities
        
        return await self.mixer.process_query(query, capabilities)

    def add_agent(self, agent: BaseAgent):
        """Add a new agent to the system"""
        self.agents.append(agent)
        self.mixer = AgentMixer(self.agents)  # Rebuild mixer with new agent

async def main():
    """Example usage of the MoA system with user input"""
    api_key = "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"
    if not api_key:
        print("Please set the GROQ_API_KEY environment variable")
        return

    # Initialize the system
    moa = MoASystem(api_key)

    # User input for the query
    query = input("Enter your query: ")

    # Process the query with automatically selected capabilities
    responses = await moa.process_query(query)
    
    # Print responses from each agent
    for response in responses:
        print(f"\n=== Response (Confidence: {response.confidence}) ===")
        print(response.content)
        print(f"Timestamp: {response.timestamp}")
        print(f"Metadata: {response.metadata}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
