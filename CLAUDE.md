# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a learning and demonstration repository for multi-agent AI frameworks. It contains practical implementations of various agent-based systems using different frameworks and LLM platforms. The project is structured around "Hello Agents" textbook chapters, with each subdirectory containing framework-specific implementations.

## Repository Structure

### Main Directories

- **chapter6/**: Complete framework implementations and demos
  - `AgentScopeDemo/`: Three Kingdoms Werewolf game using AgentScope (消息驱动架构, 并发协作)
  - `AutoGenDemo/`: Software development team collaboration using AutoGen
  - `CAMEL/`: Digital book writing collaboration using CAMEL role-playing framework
  - `Langgraph/`: Dialogue system with real search using LangGraph + Tavily API

- **the_chapter_4/**: Plan-and-Solve and Reflection patterns implementation
  - `LLMClient.py`: Unified LLM client for OpenAI-compatible APIs (used across projects)
  - `Plan_and_Solve.py`: Planning and step-by-step solving pattern
  - `Reflection.py`: Self-reflection and reasoning pattern

- **learing_agent/**: Initial agent learning experiments
- **transformer/**: Transformer model exploration
- **qwen-0.5b/**: Small language model (Qwen 0.5B) experiments

## Environment Setup

### Common Configuration

Each subdirectory may have its own `requirements.txt`. Create `.env` files for API credentials:

```bash
# Core LLM Configuration (common to most projects)
LLM_MODEL_ID=gpt-4o-mini        # or qwen-max, qwen-turbo, etc.
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1  # or Aliyun DashScope, etc.
LLM_TIMEOUT=60

# Framework-Specific APIs
DASHSCOPE_API_KEY=your-dashscope-key    # Required for AgentScope demos
TAVILY_API_KEY=your-tavily-key          # Required for Langgraph demos
```

### Installing Dependencies

```bash
# For specific demo (e.g., AgentScope):
cd chapter6/AgentScopeDemo
pip install -r requirements.txt

# For the_chapter_4 modules:
cd the_chapter_4
pip install -e .
# or manually install: python-dotenv, openai, requests, etc.
```

## Architecture Patterns

### Agent Frameworks

1. **AgentScope** (chapter6/AgentScopeDemo/)
   - Message-driven architecture with MsgHub for agent communication
   - Supports concurrent multi-agent interactions
   - Uses structured output models (Pydantic) to constrain agent behavior
   - ReActAgent for reasoning and action
   - Fault tolerance: individual agent failures don't affect overall flow

2. **AutoGen** (chapter6/AutoGenDemo/)
   - Multi-agent collaboration with human-in-the-loop
   - Agents pass tasks automatically between roles (product manager → engineer → reviewer → user)
   - Code generation and review capabilities
   - Conversation-based task orchestration

3. **CAMEL** (chapter6/CAMEL/)
   - Role-playing framework for agent collaboration
   - Task-driven conversation between two agents
   - Model factory pattern for flexible LLM provider support

4. **LangGraph** (chapter6/Langgraph/)
   - State graph-based workflow design
   - Supports asynchronous operations and real-world integrations (e.g., Tavily search)
   - Message annotations for conversation history management
   - Checkpoint system for conversation persistence

### Core Patterns (the_chapter_4/)

- **Plan-and-Solve**: Generate detailed plans before execution
- **Reflection**: Self-critique and iterative improvement patterns

### LLM Client Pattern

The `HelloAgentsLLM` client (the_chapter_4/LLMClient.py) provides a unified interface:
- Supports any OpenAI-compatible API endpoint
- Streaming response support
- Configuration via environment variables or constructor parameters
- Error handling and timeout management

## Running Demos

### AgentScope Demo

```bash
cd chapter6/AgentScopeDemo
python main_cn.py    # Run the Three Kingdoms Werewolf game
```

### AutoGen Demo

```bash
cd chapter6/AutoGenDemo
python autogen_software_team.py
```

### CAMEL Demo

```bash
cd chapter6/CAMEL
python DigitalBookWriting.py
```

### LangGraph Demo

```bash
cd chapter6/Langgraph
python Dialogue_System.py
```

## Key Implementation Details

### Agent Communication

- **AgentScope**: Uses MsgHub for message passing and fanout/sequential pipelines for orchestration
- **AutoGen**: Conversation-based with agents replying to each other's messages
- **CAMEL**: RolePlaying session manages two-agent interactions
- **LangGraph**: StateGraph defines transitions between nodes, with TypedDict state management

### Structured Output

Projects use Pydantic models to enforce response formats:
- AgentScope: structured_output_cn.py defines game action models (VoteModel, WitchActionModel, etc.)
- Parsing agent responses ensures valid game state transitions

### Error Handling

- Graceful degradation when individual agents fail
- Pydantic validation catches invalid responses
- Try-catch blocks around LLM calls with timeout management

## Testing and Debugging

- Check logs in console output for agent reasoning steps
- Verify `.env` files are properly configured before running demos
- Use Python debugger: `python -m pdb script.py`
- Examine agent role definitions and prompts in respective files

## Notes for Future Contributors

- Environment variables are loaded via `python-dotenv` - avoid hardcoding secrets
- Each framework demo is relatively self-contained; study one at a time for clarity
- The_chapter_4 modules serve as building blocks for chapter6 implementations
- All non-trivial projects use streaming responses for better UX
- Chinese language support is maintained throughout (prompts, output, documentation)
