from pytorch_tutorial.VAE import load_data, train

if __name__ == "__main__":
    # Load the data
    train_dataloader, valid_dataloader, test_dataloader = load_data()
    # Train the model
    train(train_dataloader, valid_dataloader)
    # Test the model
