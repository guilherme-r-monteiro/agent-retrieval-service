terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "image" {
  type = string
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "services" {
  repository_id = "services"
  format        = "DOCKER"
  location      = var.region
}

resource "google_service_account" "retrieval" {
  account_id   = "retrieval-service"
  display_name = "Retrieval service runtime"
}

resource "google_cloud_run_v2_service" "retrieval" {
  name     = "retrieval-service"
  location = var.region
  template {
    service_account = google_service_account.retrieval.email
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
    containers {
      image = var.image
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
  }
}

output "service_url" {
  value = google_cloud_run_v2_service.retrieval.uri
}
