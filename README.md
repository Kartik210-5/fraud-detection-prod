# Credit Card Fraud Detection

This repository, `fraud-detection-prod`, houses a robust and scalable solution for real-time credit card fraud detection. It features a Streamlit-powered interactive web application for user-friendly interaction, backed by a high-performance FastAPI backend. The system is designed to load, clean, and analyze transaction data, utilizing machine learning models to classify transactions as **Fraud** or **Legit** with associated confidence scores.

Built for an internship project, this solution is tailored for the `creditcard_2023.csv` dataset and prioritizes both accuracy and deployment efficiency.

## ✨ Key Features & Benefits

*   **Real-time Fraud Scoring:** Predicts fraudulent transactions with high accuracy and provides confidence scores.
*   **Interactive Streamlit UI:** A user-friendly web interface (`app.py`) for uploading data, visualizing results, and interacting with the fraud detection system.
*   **High-Performance FastAPI Backend:** A robust and scalable API (`api.py`) for serving fraud predictions, complete with:
    *   **API Key Authentication:** Secure access to prediction endpoints.
    *   **Rate Limiting:** Protects the API from abuse and ensures fair usage.
*   **Automated Data Quality Checks:** Performs checks for missing values, duplicate entries, and invalid labels during data processing.
*   **Advanced Machine Learning Models:** Integrates both Random Forest and XGBoost classifiers, with support for various data imbalance handling techniques (e.g., SMOTE, class weighting).
*   **Data Drift Monitoring:** Includes tools (`drift_analyzer.py`, `data_drift_report.html`) to detect and report on data drift, ensuring model relevance over time.
*   **Containerized Deployment:** Ready for deployment using Docker, ensuring consistency across different environments.
*   **Supabase Integration:** Seamlessly integrates with Supabase for potential data storage, user management, or other backend services.
*   **Development Container Support:** Pre-configured with `.devcontainer` for a consistent and isolated development environment using VS Code.

## 🛠️ Technologies Used

### Languages

*   Python (3.10+)

### Frameworks

*   **FastAPI:** For building the high-performance prediction API.
*   **Streamlit:** For creating the interactive web application.
*   **scikit-learn & XGBoost:** For machine learning model development.
*   **pandas:** For data manipulation and analysis.

### Tools & Technologies

*   **Docker:** For containerization and environment consistency.
*   **Uvicorn:** ASGI server for FastAPI.
*   **Supabase:** Backend-as-a-Service for database and other services.
*   **SlowAPI:** For API rate limiting.
*   **VS Code Dev Containers:** For streamlined development workflows.

##  Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/).
*   **pip**: Python package installer (usually comes with Python).
*   **Docker Desktop**: Download from [docker.com](https://www.docker.com/products/docker-desktop/).
*   (Optional) **Visual Studio Code**: Recommended for development, especially with the Remote - Containers extension.

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/spookycrow515/fraud-detection-prod.git
cd fraud-detection-prod
```

### 2. Local Development Setup

#### a. Create a Virtual Environment

```bash
python -m venv venv
# On Linux/macOS
source venv/bin/activate
# On Windows
.\venv\Scripts\activate
```

#### b. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Note: The `requirements.txt` content is not fully provided, but based on the code snippets, it would include `fastapi`, `uvicorn`, `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `pydantic`, `slowapi`, `supabase-py`, `requests`.)*

#### c. Configure Environment Variables

Create a `.env` file in the root directory or set them directly in your shell:

```env
API_URL="http://127.0.0.1:8000" # URL where the FastAPI app will run
API_KEY="dev-secret-key-123"   # Your secret API key for authentication
SUPABASE_URL="YOUR_SUPABASE_PROJECT_URL"
SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
```

#### d. Run the FastAPI Backend

Open a new terminal, activate your virtual environment, and run:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be accessible at `http://localhost:8000`.

#### e. Run the Streamlit Frontend

Open another terminal, activate your virtual environment, and run:

```bash
streamlit run app.py
```

The Streamlit application will open in your web browser, typically at `http://localhost:8501`.

### 3. Docker Setup (API Only)

To run the FastAPI backend using Docker:

#### a. Build the Docker Image

```bash
docker build -t fraud-detection-prod .
```

#### b. Run the Docker Container

```bash
docker run -p 8000:8000 -e API_KEY="dev-secret-key-123" -e SUPABASE_URL="YOUR_SUPABASE_PROJECT_URL" -e SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY" fraud-detection-prod
```

The FastAPI application will be available at `http://localhost:8000`.

### 4. Development Container (VS Code)

If you're using VS Code with the Remote - Containers extension:

1.  Open the project folder in VS Code.
2.  VS Code will prompt you to "Reopen in Container". Click this button.
3.  The development environment will be set up automatically inside a Docker container, including all dependencies.

## 📖 Usage Examples & API Documentation

### Streamlit Application

Once the Streamlit app is running (`http://localhost:8501`), you can:

*   Upload a CSV file containing transaction data.
*   View data quality reports.
*   Trigger fraud detection and see the results interactively.

### FastAPI Prediction API

The core prediction functionality is exposed via a FastAPI endpoint.

#### `POST /predict`

**Description:** Submits transaction data for fraud prediction.
**Rate Limit:** 30 requests per minute.

**Headers:**

*   `Content-Type: application/json`
*   `X-API-Key: <YOUR_API_KEY>` (e.g., `dev-secret-key-123`)

**Request Body (JSON):**

```json
{
    "Amount": 15.5,
    "V1": 0.0,
    "V2": 0.0,
    // ... V3 to V28 ...
    "V28": 0.0
}
```

**Example Request (using `curl`):**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: dev-secret-key-123" \
     -d '{
           "Amount": 15.5,
           "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0, "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0,
           "V10": 0.0, "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0, "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0,
           "V20": 0.0, "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0
         }'
```

**Example Request (using `burst_test.py`):**

The `burst_test.py` script demonstrates how to send multiple requests to the API and observe the rate limiting in action.

```bash
python burst_test.py
```

This script will send 35 rapid requests to breach the 30/min cap, demonstrating the rate limit handling.

### Available Models

The project includes pre-trained models using different techniques to address class imbalance:

*   `fraud_model_random_forest_None.pkl`: Random Forest without imbalance handling.
*   `fraud_model_random_forest_SMOTE.pkl`: Random Forest with SMOTE oversampling.
*   `fraud_model_random_forest_class_weight.pkl`: Random Forest with class weighting.
*   `fraud_model_xgboost_None.pkl`: XGBoost without imbalance handling.
*   `fraud_model_xgboost_SMOTE.pkl`: XGBoost with SMOTE oversampling.
*   `fraud_model_xgboost_class_weight.pkl`: XGBoost with class weighting.

The `api.py` and `model.py` files manage loading and using these models.

## ⚙️ Configuration Options

The primary configuration options are managed through environment variables:

*   `API_URL`: Specifies the endpoint for the FastAPI backend (used by the Streamlit app).
*   `API_KEY`: The secret key required to authenticate with the prediction API.
*   `SUPABASE_URL`: The URL of your Supabase project.
*   `SUPABASE_KEY`: Your Supabase anonymous key.

These variables should be set in your environment or within a `.env` file for local development.

## 🤝 Contributing Guidelines

We welcome contributions to enhance this fraud detection system!

To contribute:

1.  **Fork** the repository.
2.  **Clone** your forked repository: `git clone https://github.com/YOUR_USERNAME/fraud-detection-prod.git`
3.  **Create a new branch**: `git checkout -b feature/your-feature-name`
4.  **Make your changes**, ensuring code quality, test coverage, and documentation.
5.  **Commit your changes**: `git commit -m "feat: Add new feature"`
6.  **Push to your branch**: `git push origin feature/your-feature-name`
7.  **Open a Pull Request** to the `main` branch of the original repository, describing your changes and their benefits.

Please ensure your code adheres to standard Python best practices and includes appropriate tests.

## 📄 License

This project is currently **unlicensed**.

If you plan to use or distribute this software, please contact the owner, spookycrow515, for licensing information or consider adding a standard open-source license (e.g., MIT, Apache 2.0) to facilitate usage and collaboration.

## 🙏 Acknowledgments

*   **creditcard_2023.csv**: The dataset used for training and evaluating the fraud detection models.
*   The developers and maintainers of **FastAPI**, **Streamlit**, **scikit-learn**, **XGBoost**, **pandas**, **Docker**, and **Supabase** for providing excellent tools and libraries.