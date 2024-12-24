# Replace <PROJECT_ID> with your GCP project ID and <SERVICE_ACCOUNT_EMAIL> with your service account email

gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member=serviceAccount:<SERVICE_ACCOUNT_EMAIL> \
    --role=roles/aiplatform.customCodeServiceAgent

gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member=serviceAccount:<SERVICE_ACCOUNT_EMAIL> \
    --role=roles/aiplatform.admin

gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member=serviceAccount:<SERVICE_ACCOUNT_EMAIL> \
    --role=roles/storage.objectAdmin