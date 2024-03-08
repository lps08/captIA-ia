# captIA-ia

## Installation

Follow these steps to set up the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/lps08/captIA-ia.git
    ```

2. Navigate to the project directory:
    ```bash
    cd captIA-ia
    ```

3. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Environment Variables

Create a `.env` file in the root directory of the project and define the following environment variables:

```plaintext
GOOGLE_API_KEY=<api-key-value>
