output "datasets" {
  description = "Fully qualified dataset IDs."
  value = {
    raw = "${var.project_id}.${google_bigquery_dataset.raw.dataset_id}"
    dbt = "${var.project_id}.${google_bigquery_dataset.dbt.dataset_id}"
  }
}
