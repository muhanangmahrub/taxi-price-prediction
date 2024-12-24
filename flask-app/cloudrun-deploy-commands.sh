# Grant the necessary permissions to the Cloud Run Service Account:
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member=serviceAccount:<SERVICE_ACCOUNT_EMAIL> \
  --role='roles/aiplatform.admin'

# Step-1: Build the Docker image
docker build -t <IMAGE_NAME> .

# Push to Container Registry:
docker tag <IMAGE_NAME> gcr.io/<PROJECT_ID>/<IMAGE_NAME>
docker push gcr.io/<PROJECT_ID>/<IMAGE_NAME>

# Deploy the image to Cloud Run:
gcloud run deploy <SERVICE_NAME> --image gcr.io/<PROJECT_ID>/<IMAGE_NAME> --region <REGION>
