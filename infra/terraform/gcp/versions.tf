terraform {
  required_version = ">= 1.10"

  # State lives in HCP Terraform (Free tier). The workspace uses Local execution, so plan and
  # apply run on this machine and Google credentials never leave it.
  # The organization comes from the TF_CLOUD_ORGANIZATION environment variable.
  cloud {
    workspaces {
      name = "filinglens-gcp"
    }
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0"
    }
  }
}
