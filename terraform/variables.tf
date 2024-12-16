variable "resource_group_name" {
  description = "The name of the Azure Resource Group"
  type        = string
  default     = "askmeai-webapp"  # Replace with your actual Resource Group name
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "Germany West Central"  # Choose your preferred region
}

variable "resource_group_location" {
  description = "The location of the Azure Resource Group"
  type        = string
  default     = "Germany West Central"  # Replace with your actual Resource Group location
}

variable "storage_account_name" {
  description = "Name of the Storage account"
  type = string
  default = "askmeaiwebappstorage"
}

variable "storage_account_contaier_name" {
  description = "Name of the storage container"
  type = string
  default = "askmeaiwebappstorage-container"
}

variable "blob_storage_name" {
  description = "Name of the Blob storage"
  type = string
  default = "backend-models"
}