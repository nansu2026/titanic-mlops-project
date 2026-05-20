## MLOps Workflow

This project follows a complete MLOps workflow for Titanic survival prediction.

### CI/CD

When code is pushed to GitHub, the Continuous Integration pipeline runs automatically.  
It installs dependencies, trains the model, runs tests, and uploads MLflow experiment results.

### Continuous Training

The Continuous Training pipeline runs automatically when files inside the `data/` folder, `src/` folder, `config.yaml`, `requirements.txt`, or `tests/` are changed.

This means if the dataset changes, the model is retrained automatically.

The project also has scheduled retraining every Sunday at midnight using GitHub Actions cron.

### MLflow Tracking

MLflow records model parameters, metrics, artifacts, and model versions.  
The best model is saved as `models/best_model.pkl`.

### Kubernetes Deployment

The deployment pipeline builds a Docker image and deploys the Flask ML API to Kubernetes using KIND.

### Scalability

Scalability is handled using Kubernetes Horizontal Pod Autoscaler.

The HPA can scale the application from 2 pods to 5 pods based on CPU usage.

### Availability

Availability is improved using:

- 2 Kubernetes replicas
- readiness probe
- liveness probe
- service-based routing

If one pod fails, Kubernetes can route traffic to another healthy pod.

### Reusability

The project is reusable because configuration is separated into `config.yaml`, source code is modular, and the same workflow can be reused with another dataset or model.