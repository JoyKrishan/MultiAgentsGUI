# MultiAgentGui Documentation

## Overview

MultiAgentGui is a powerful toolset designed to enhance productivity and streamline workflows. This documentation provides an overview of the two main agentic tools available: `agenticpr` and `agenticwriter`.

## agenticpr

`APRGui` is a sophisticated tool designed for automated program repair. It identifies bugs in your code and, using the provided failed test cases, automatically generates fixes to repair the bugs. This tool significantly reduces the time and effort required for debugging and ensures that your codebase remains robust and error-free.

### Key Features

- **Bug Detection**: Identifies and isolates bugs in your code.
- **Automated Fixes**: Generates and applies fixes based on failed test cases.
- **Efficiency**: Reduces the time spent on manual debugging.
- **Reliability**: Ensures that the applied fixes pass all relevant test cases.
### Usage

To use `APRGui`, follow these steps:

1. **Running**: Run the requirements file, add the project path to the Python path, and then run the main script.
    ```bash
    pip install -r requirements.txt
    export PYTHONPATH=$PYTHONPATH:path/to/your/project
    python3 agenticpr main.py
    ```

By integrating `APRGui` into your development workflow, you can maintain a high-quality codebase with minimal manual intervention.


## agenticwriter

`agenticwriter` is a content generation tool that assists in writing documentation, reports, and other textual content. It leverages AI to provide suggestions and automate repetitive writing tasks.

### Features

- **Content Suggestions**: Provides context-aware suggestions to improve writing quality.
- **Automated Reports**: Generates reports based on data inputs.
- **Template Management**: Allows the creation and use of templates for consistent documentation.
- **Integration**: Works with various text editors and content management systems.

### Usage

To use `agenticwriter`, follow these steps:

1. **Installation**: Install the tool using the package manager.
    ```bash
    npm install agenticwriter
    ```
2. **Configuration**: Set up the tool with your preferred settings.
    ```json
    {
      "templates": "path/to/templates",
      "output": "path/to/output"
    }
    ```
3. **Running**: Run the requirements file, add the project path to the Python path, and then run the main script.    
    ```bash
    pip install -r requirements.txt
    export PYTHONPATH=$PYTHONPATH:path/to/your/project
    python3 agenticwriter main.py
    ```
    ```

## Conclusion

APRGui's `agenticpr` and `agenticwriter` tools are designed to enhance productivity by automating and simplifying key tasks. By integrating these tools into your workflow, you can save time and focus on more critical aspects of your projects.

For more detailed information, refer to the individual tool documentation and user guides.