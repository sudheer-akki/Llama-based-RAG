resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.resource_group_location
}

resource "azurerm_storage_account" "storage" {
    name = var.storage_account_name
    resource_group_name = azurerm_resource_group.rg.name
    location = azurerm_resource_group.rg.location
    account_tier = "Standard"
    account_replication_type = "LRS" #locally redundant storage
    # Enable hierarchical namespace for improved performance
    is_hns_enabled = true
}

resource "azurerm_storage_container" "container" {
    name = var.storage_account_contaier_name
    storage_account_id = azurerm_storage_account.storage.id
    container_access_type = "private"
}
