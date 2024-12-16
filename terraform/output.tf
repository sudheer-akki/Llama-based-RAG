output "resource_group_name" {
  description = "Name of the Resource Group"
  value = azurerm_resource_group.rg.name
}

output "resource_group_id" {
  description = "Name of the Resource Group ID"
  value = azurerm_resource_group.rg.id
}

output "resource_group_location" {
  description = "Name of the Resource Group Location"
  value =  azurerm_resource_group.rg.location
}


output "azure_storage_account_name" {
  description = "Name of the azurerm storage account"
  value =  azurerm_storage_account.storage.name
}


output "azurerm_storage_container_name" {
  description = "Name of the azurerm storage container name"
  value =  azurerm_storage_container.container.name
}


output "azurerm_storage_blob_name" {
  description = "Name of the Azure storage blob name"
  value =  azurerm_storage_blob.blob_storage.name
}