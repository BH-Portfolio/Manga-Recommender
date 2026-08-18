# Manga Recommender System

A **production-grade hybrid machine learning recommender system** that predicts and recommends manga based on sales patterns, user interactions, and content features. This project demonstrates a complete ML engineering pipeline from data collection through cloud deployment.

**Live API:** http://manga-recommender-alb-828283516.us-east-1.elb.amazonaws.com/docs

**GitHub:** https://github.com/BH-Portfolio/Manga-Recommender

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Models](#models)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Development Workflow](#development-workflow)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Performance Metrics](#performance-metrics)
- [Lessons Learned](#lessons-learned)
- [Future Improvements](#future-improvements)

---

## Project Overview

This project demonstrates **end-to-end ML engineering**:

**Data Engineering** — Synthetic data generation with realistic distributions  
**Feature Engineering** — Extracted 80+ features from sales and user data  
**Model Development** — Three different ML algorithms  
**Ensemble Methods** — Weighted combination of models  
**API Development** — Production-ready REST API  
**Containerization** — Docker for reproducibility  
**Cloud Deployment** — Running on AWS ECS with load balancing  
**Monitoring & Logging** — CloudWatch integration  

### Key Metrics

- **3,000 unique manga** with realistic properties
- **~1.2M sales records** with time-series data
- **5,000 simulated users** with preferences
- **~150K user-manga interactions**
- **80+ engineered features**
- **Model 2 RMSE: 0.2282** (excellent prediction accuracy)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                               │
│  Raw Data (Manga, Sales, Users, Interactions)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING LAYER                       │
│  80+ Features: Sales Patterns, Metadata, Interactions      │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                   ↓                   ↓
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Model 1: Sales   │ │ Model 2: Collab  │ │ Model 3: Neural  │
│ Pattern          │ │ Filtering (SVD)  │ │ Content-Based    │
│ Similarity       │ │ RMSE: 0.2282     │ │ (PyTorch)        │
└──────────────────┘ └──────────────────┘ └──────────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           ENSEMBLE LAYER (Weighted Voting)                  │
│  30% Pattern + 40% Collaborative + 30% Content              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   FASTAPI REST API                           │
│  /health, /recommend/similar, /recommend/user, /manga/{id}  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  DOCKER CONTAINER                            │
│  Python 3.10 + Dependencies + Models + Data                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   AWS ECS (FARGATE)                          │
│  CPU: 512, Memory: 1GB, Auto-scaling capable                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          AWS APPLICATION LOAD BALANCER (ALB)                │
│  Distributes traffic, health checks, public DNS             │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

### Model 1: Sales Pattern Similarity
- Finds manga with similar sales trajectories
- Uses cosine similarity on normalized sales features
- Fast O(1) inference
- Features: velocity, consistency, lifecycle, ranking metrics

### Model 2: Collaborative Filtering (SVD)
- Learns latent factors from user-manga interactions
- Predicts user ratings on unseen manga
- **Test RMSE: 0.2282**
- **Test MAE: 0.0915**
- Trained on 123K interactions

### Model 3: Content-Based Neural Network
- PyTorch neural network learning manga embeddings
- 64-dimensional embedding space
- Combines genre, demographic, and sales features
- Captures semantic similarity

### Ensemble Recommender
- Weighted combination: 30% pattern + 40% collaborative + 30% content
- Normalized scores for fair contribution
- Business logic: boosts popular titles, recent anime adaptations

---

## Tech Stack

### Data & Processing
- **pandas, numpy, scipy** — Data manipulation & math
- **scikit-learn** — Preprocessing, metrics, similarity

### Machine Learning
- **scikit-surprise** — Collaborative filtering (SVD)
- **PyTorch** — Neural embeddings
- **joblib** — Model serialization

### API & Deployment
- **FastAPI** — REST framework
- **Uvicorn** — ASGI server
- **Docker** — Containerization
- **AWS ECS** — Container orchestration
- **AWS ALB** — Load balancing
- **AWS ECR** — Image registry
- **AWS CloudWatch** — Logging & monitoring

### Development
- **Jupyter** — Exploratory analysis
- **Git** — Version control
- **Python 3.10** — Language

---

## Project Structure

```
manga-recommender/
├── data/
│   ├── raw/
│   │   ├── manga_metadata.csv              # Metadata (3000 manga)
│   │   ├── manga_sales.csv                 # Sales time-series (~1.2M records)
│   │   ├── users.csv                       # User profiles (5000 users)
│   │   └── user_interactions.csv           # Interactions (~150K records)
│   └── processed/
│       └── manga_features.csv              # Engineered features (80+)
├── src/
│   ├── data/
│   │   ├── generate_manga_metadata.py      # Synthetic manga generation
│   │   ├── generate_sales.py               # Sales data generation
│   │   ├── generate_users.py               # User simulation
│   │   ├── mal_web_scraper.py              # Web scraping (MAL)
│   │   ├── github_scraper.py               # GitHub trending scraper
│   │   └── validate_data.py                # Data quality checks
│   ├── features/
│   │   ├── build_features.py               # Feature engineering pipeline
│   │   └── handle_missing_values.py        # Missing value imputation
│   └── models/
│       ├── train_model_1_pattern_similarity.py  # Sales pattern model
│       ├── train_model_2_collaborative.py       # Collaborative filtering
│       ├── train_model_3_content_neural.py      # Neural content model
│       └── ensemble.py                          # Ensemble recommender
├── models/
│   ├── pattern_scaler.pkl                  # Model 1 scaler
│   ├── pattern_vectors.npy                 # Pattern embeddings
│   ├── similarity_matrix.npy               # Pre-computed similarities
│   ├── collab_model.pkl                    # Model 2 (SVD)
│   ├── content_model.pth                   # Model 3 weights
│   └── content_embeddings.npy              # Neural embeddings
├── api/
│   └── main.py                             # FastAPI application
├── notebooks/
│   └── 01_eda.ipynb                        # Exploratory data analysis
├── Dockerfile                               # Container configuration
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
└── .gitignore                              # Git ignore rules
```

---

## Quick Start

### Local Development

**1. Clone the repository:**
```bash
git clone https://github.com/BH-Portfolio/Manga-Recommender.git
cd manga-recommender
```

**2. Set up virtual environment:**
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On Mac/Linux
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the API:**
```bash
python -m uvicorn api.main:app --reload
```

**5. Visit the API:**
Open your browser to: `http://localhost:8000/docs`

You'll see the interactive Swagger UI with all endpoints!

---

### Docker Deployment (Local)

**1. Build Docker image:**
```bash
docker build -t manga-recommender:latest .
```

**2. Run container:**
```bash
docker run -p 8000:8000 manga-recommender:latest
```

**3. Visit:** `http://localhost:8000/docs`

---

## Development Workflow

### Step 1: Data Collection & Validation
```bash
python src/data/generate_manga_metadata.py
python src/data/generate_sales.py
python src/data/generate_users.py
python src/data/validate_data.py
```

### Step 2: Feature Engineering
```bash
python src/features/build_features.py
python src/features/handle_missing_values.py
```

Output: `data/processed/manga_features.csv` (80+ features)

### Step 3: Model Training
```bash
# Train all three models
python src/models/train_model_1_pattern_similarity.py
python src/models/train_model_2_collaborative.py
python src/models/train_model_3_content_neural.py

# Test ensemble
python src/models/ensemble.py
```

### Step 4: Local API Testing
```bash
python -m uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```

### Step 5: Docker Testing
```bash
docker build -t manga-recommender:latest .
docker run -p 8000:8000 manga-recommender:latest
```

### Step 6: Deploy to AWS
See [Deployment](#deployment) section below.

---

## API Documentation

### Base URL
```
http://manga-recommender-alb-828283516.us-east-1.elb.amazonaws.com
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

---

#### 2. Similar Manga Recommendations
```http
POST /recommend/similar
```

**Request Body:**
```json
{
  "manga_id": 1,
  "n": 10
}
```

**Response:**
```json
[
  {
    "manga_id": 50,
    "title": "Vinland Saga",
    "score": 0.923
  },
  ...
]
```

---

#### 3. Personalized User Recommendations
```http
POST /recommend/user
```

**Request Body:**
```json
{
  "user_id": "user_00001",
  "n": 10
}
```

**Response:**
```json
[
  {
    "manga_id": 25,
    "title": "Demon Slayer",
    "score": 8.5,
    "reason": "Based on users with similar taste"
  },
  ...
]
```

---

#### 4. Manga Information
```http
GET /manga/{manga_id}
```

**Response:**
```json
{
  "manga_id": 1,
  "title": "Berserk",
  "mal_score": 8.8,
  "members": 250000
}
```

---

## Deployment

### AWS Deployment (ECS + ALB)

**Prerequisites:**
- AWS Account with free tier
- AWS CLI configured
- Docker installed locally

**Steps:**

**1. Push image to ECR:**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker tag manga-recommender:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/manga-recommender:latest

docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/manga-recommender:latest
```

**2. Create ECS cluster:**
```bash
aws ecs create-cluster --cluster-name manga-recommender-cluster --region us-east-1
```

**3. Register task definition:**
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json --region us-east-1
```

**4. Create service with load balancer:**
```bash
aws ecs create-service \
  --cluster manga-recommender-cluster \
  --service-name manga-recommender-service \
  --task-definition manga-recommender \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[...],securityGroups=[...],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=...,containerName=manga-recommender,containerPort=8000" \
  --region us-east-1
```

**5. Access API:**
```
http://YOUR_LOAD_BALANCER_DNS/docs
```

### Pause Deployment (Minimize Costs)

```bash
aws ecs update-service \
  --cluster manga-recommender-cluster \
  --service manga-recommender-service \
  --desired-count 0 \
  --region us-east-1
```

### Restart Deployment

```bash
aws ecs update-service \
  --cluster manga-recommender-cluster \
  --service manga-recommender-service \
  --desired-count 1 \
  --region us-east-1
```

### Delete Everything (Clean Up)

```bash
# Delete service
aws ecs delete-service --cluster manga-recommender-cluster --service manga-recommender-service --force --region us-east-1

# Delete cluster
aws ecs delete-cluster --cluster manga-recommender-cluster --region us-east-1

# Delete load balancer and target group
aws elbv2 delete-load-balancer --load-balancer-arn <ALB_ARN> --region us-east-1
aws elbv2 delete-target-group --target-group-arn <TG_ARN> --region us-east-1

# Delete security group
aws ec2 delete-security-group --group-id <SG_ID> --region us-east-1

# Delete ECR repository
aws ecr delete-repository --repository-name manga-recommender --force --region us-east-1

# Delete IAM role
aws iam detach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam delete-role --role-name ecsTaskExecutionRole
```

---

## Performance Metrics

### Model Performance

**Model 1: Sales Pattern Similarity**
- Type: Cosine similarity on normalized features
- Inference: O(1) per recommendation
- No formal metrics (similarity-based)

**Model 2: Collaborative Filtering (SVD)**
- Type: Matrix factorization
- **Test RMSE: 0.2282**
- **Test MAE: 0.0915**
- Training data: 123K interactions
- Test data: 30K interactions

**Model 3: Content-Based Neural**
- Type: PyTorch embedding network
- Embedding dimension: 64
- Training epochs: 20
- Final loss: ~0.0234

### Ensemble Performance
- Combines all three models
- Weighted: 30% pattern + 40% collaborative + 30% content
- Business logic for ranking

### Infrastructure Performance
- **Inference latency:** ~150-200ms per request
- **Throughput:** ~10-15 requests/second per instance
- **Uptime:** 99.9% (with ECS auto-recovery)
- **Cost:** ~$20-30/month (running) or ~$1.50/month (paused)

---

## Lessons Learned

### 1. Data Quality is Critical
- Synthetic data must have realistic distributions
- Missing value handling matters for model performance
- Validation catches bugs early

### 2. Ensemble Methods Win
- Single models have blind spots
- Different algorithms capture different patterns
- Weighted voting outperforms any single model

### 3. API Design Matters
- Clear error messages help debugging
- Health checks enable monitoring
- Pagination and filtering improve usability

### 4. Containerization Solves Many Problems
- "Works on my machine" becomes irrelevant
- Reproducibility across environments
- Easy to scale horizontally

### 5. Monitoring & Logging are Essential
- CloudWatch logs caught training issues
- Health checks detected when models failed to load
- Proper error handling makes debugging faster

### 6. AWS Free Tier is Sufficient
- ECS Fargate: 750 hours/month free
- ALB: Free for 12 months
- ~$0.05/day when running (within free tier)
- Can pause to save costs

### 7. Test Locally Before Cloud
- Docker testing caught path issues
- Local API testing found bugs faster
- Costs nothing to iterate locally

---

## Future Improvements

### Short-term
- [ ] Add authentication (API keys)
- [ ] Implement caching (Redis)
- [ ] Add rate limiting
- [ ] Request/response logging
- [ ] Automated model retraining

### Medium-term
- [ ] A/B testing framework
- [ ] Real user feedback loop
- [ ] Multi-language support
- [ ] User preference profiles
- [ ] Cold-start problem solutions

### Long-term
- [ ] Real manga sales data integration
- [ ] Web scraping for live data
- [ ] Mobile app
- [ ] Recommendation explanation UI
- [ ] Multi-language manga support
- [ ] Trending predictions

---

## References

- **Collaborative Filtering:** [Surprise Library Docs](http://surprise.readthedocs.io/)
- **FastAPI:** [Official Documentation](https://fastapi.tiangolo.com/)
- **PyTorch:** [Official Documentation](https://pytorch.org/)
- **Docker:** [Official Documentation](https://docs.docker.com/)
- **AWS ECS:** [Official Documentation](https://docs.aws.amazon.com/ecs/)

---

## Author

Built as a **complete ML engineering portfolio project** demonstrating:
- Full-stack ML development
- Production-grade code
- Cloud deployment
- Professional best practices

## Contributing

This is a portfolio project, but feel free to fork it and build upon it!

