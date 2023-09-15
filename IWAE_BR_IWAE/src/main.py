from VAE import *
from IWAE import *
from BR_IWAE import *


#Parameters of the model
params = {
    'input_size': 784,
    'hidden_size': 400,
    'latent_size': 20,
    'K': 5,
    'k_0': 0,
    'k_max': 3,
    'batch_size': 256,
    'lr': 0.01,
    'num_epochs': 100
}
model_names = ['VAE']   #['VAE', 'IWAE', 'BR-IWAE']

# MNIST data transformations
transform = transforms.Compose([transforms.ToTensor()])
transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
binarize = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.bernoulli(x))])

# Load MNIST train dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)

# Load MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)


def create_model(model_name, params):
    if model_name == 'VAE':
        model = VAE(params['input_size'], params['hidden_size'], params['latent_size'])
    elif model_name == 'IWAE':
        model = IWAE(params['input_size'], params['hidden_size'], params['latent_size'], params['K'])
    elif model_name == 'BR-IWAE':
        model = BR_IWAE(params['input_size'], params['hidden_size'], params['latent_size'], params['K'], params['k_0'], params['k_max'])
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return model



if __name__ == "__main__":
    # Initialize models
    models = {}
    for model_name in model_names:
        models[model_name] = create_model(model_name, params)


    # Training loop

    # Create a dictionary to store loss values for each model
    losses_dict = {model_name: {'train': [], 'test': []} for model_name in models}

    for model_name, model in models.items():

        #optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        #optimizer = optim.SGD(model.parameters(), lr=params['lr'])
        optimizer = optim.Adagrad(model.parameters(), lr=params['lr'])

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)

        print(model_name)

        for epoch in range(params['num_epochs']):
            model.train()
            train_loss = 0

            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.view(-1, params['input_size'])
                optimizer.zero_grad()
                if model_name == 'VAE':
                    x_hat, mu, logvar = model(data)
                    loss = model.loss(x_hat, data, mu, logvar)
                if model_name == 'IWAE':
                    _, loss = model(data)
                if model_name == 'BR-IWAE':
                    loss = model(data)
                loss.backward()

                optimizer.step()
                train_loss += loss.item()

            # Step the learning rate scheduler at the end of each epoch
            scheduler.step()

            # Calculate the average training loss for the epoch and append it to the corresponding model's loss list
            avg_train_loss = train_loss / len(train_loader.dataset)
            losses_dict[model_name]['train'].append(avg_train_loss)

            print('Epoch: [{}/{}], Training Loss: {:.3f}'.format(epoch+1, params['num_epochs'], avg_train_loss))

            # Evaluate the model on the test dataset
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.view(-1, params['input_size'])
                    if model_name == 'VAE':
                        x_hat, mu, logvar = model(data)
                        loss = model.loss(x_hat, data, mu, logvar)
                    if model_name == 'IWAE':
                        _, loss = model(data)
                    if model_name == 'BR-IWAE':
                        loss = model(data)
                    test_loss += loss.item()

            # Calculate the average test loss for the epoch and append it to the corresponding model's loss list
            avg_test_loss = test_loss / len(test_loader.dataset)
            losses_dict[model_name]['test'].append(avg_test_loss)

            print('Epoch: [{}/{}], Test Loss: {:.3f}'.format(epoch+1, params['num_epochs'], avg_test_loss))

        # Saving the dictionary
        with open('VAE_Adagrad_test_loss.pkl', 'wb') as f:
            pickle.dump(losses_dict, f)