The project focuses on CI/CD and Continuous Training.

CI/CD:
Whenever code is pushed to GitHub, the CI pipeline installs dependencies, trains the model, runs tests, and stores MLflow artifacts. If CI passes, the deployment workflow builds a Docker image, deploys the application into Kubernetes, checks pod status, and tests the API endpoint.

Continuous Training:
The training pipeline runs automatically when the data folder, source code, configuration file, or requirements file changes. In addition, a scheduled GitHub Actions cron job retrains the model every week. This ensures the model can be refreshed even if no manual training is started.

MLflow:
MLflow is used inside the training script to track experiments, metrics, parameters, and model artifacts. Each retraining run creates a new MLflow run, making model comparison and version tracking possible.

Kubernetes:
The application is containerized using Docker and deployed to Kubernetes. Kubernetes provides availability through multiple replicas and scalability through Horizontal Pod Autoscaler.