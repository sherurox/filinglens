provider "google" {
  project = var.project_id
}

locals {
  # The sandbox expires tables and partitions after 60 days. Declaring the same value keeps
  # Terraform's view equal to what Google enforces, so plans show no drift.
  sixty_days_ms = 60 * 24 * 60 * 60 * 1000

  labels = {
    project    = "filinglens"
    milestone  = "fl-m1"
    managed_by = "terraform"
  }
}

resource "google_bigquery_dataset" "raw" {
  dataset_id                      = "filinglens_raw"
  friendly_name                   = "FilingLens raw"
  description                     = "FSDS rows for the 300-company universe, loaded from Parquet with bq load."
  location                        = var.location
  default_table_expiration_ms     = local.sixty_days_ms
  default_partition_expiration_ms = local.sixty_days_ms
  delete_contents_on_destroy      = true # terraform destroy is the full teardown
  labels                          = local.labels
}

resource "google_bigquery_dataset" "dbt" {
  dataset_id                      = "filinglens"
  friendly_name                   = "FilingLens dbt models"
  description                     = "dbt staging, intermediate and marts (dbt build --target bigquery)."
  location                        = var.location
  default_table_expiration_ms     = local.sixty_days_ms
  default_partition_expiration_ms = local.sixty_days_ms
  delete_contents_on_destroy      = true
  labels                          = local.labels
}
