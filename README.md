# FilingLens

Reads US public-company filings from SEC EDGAR and answers analyst questions about them with cited numbers.
See [docs/SPEC.md](docs/SPEC.md) for the full specification.

## Cloud resources and teardown ($0)

| Resource | Created by | Free because | Teardown |
|---|---|---|---|
| BigQuery datasets `filinglens`, `filinglens_raw` (project with the BigQuery sandbox) | `infra/terraform/gcp` (state in HCP Terraform, local execution) | Sandbox: no billing account; 10 GB storage, 1 TB queries/month; tables expire after 60 days | `cd infra/terraform/gcp && terraform destroy` |
| HCP Terraform workspace `filinglens-gcp` | `terraform init` | Free tier: up to 500 managed resources | Delete the workspace in HCP Terraform (Settings → Destruction and Deletion) |
| Google Cloud project | Google Cloud console | No billing attached | `gcloud projects delete <project-id>` |

Terraform needs `TF_CLOUD_ORGANIZATION` and `TF_VAR_project_id` (see `.env.example`); load them with `set -a; source .env; set +a`.
