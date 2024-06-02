from pytorch_tutorial import VAE


def VariationalAutoencoder() -> None:
    # Load the data
    train_dataloader, valid_dataloader, test_dataloader = VAE.load_data()
    # Train the model
    VAE.train(train_dataloader, valid_dataloader)
    # Test the model
