variable "project_id" {
  description = "Google Cloud project with the BigQuery sandbox (set TF_VAR_project_id)."
  type        = string
}

variable "location" {
  description = "BigQuery location for both datasets."
  type        = string
  default     = "US"
}
